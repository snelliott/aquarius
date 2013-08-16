/* Copyright (c) 2013, Devin Matthews
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL DEVIN MATTHEWS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. */

#include "task.hpp"

#include <set>

using namespace std;
using namespace aquarius;
using namespace aquarius::task;

Requirement::Requirement(const string& type, const string& name)
: type(type), name(name) {}

void Requirement::fulfil(const Product& product)
{
    this->product.reset(new Product(product));
}

bool Requirement::exists() const
{
    return product->exists();
}

Product& Requirement::get()
{
    if (!product) throw logic_error("Requirement " + name + " not fulfilled");
    return *product;
}

Product::Product(const string& type, const string& name)
: type(type), name(name), requirements(), used(new bool(false)) {}

Product::Product(const string& type, const string& name, const vector<Requirement>& reqs)
: type(type), name(name), requirements(reqs), used(new bool(false)) {}

void Product::addRequirement(const Requirement& req)
{
    requirements.push_back(req);
}

void Product::addRequirements(const std::vector<Requirement>& reqs)
{
    requirements.insert(requirements.end(), reqs.begin(), reqs.end());
}

Task::Task(const string& type, const string& name)
: type(type), name(name) {}

map<string,Task::factory_func>& Task::tasks()
{
    static std::map<std::string,factory_func> tasks_;
    return tasks_;
}

void Task::addProduct(const Product& product)
{
    products.push_back(product);
}

bool Task::registerTask(const string& name, factory_func create)
{
    tasks()[name] = create;
    return true;
}

Product& Task::getProduct(const string& name)
{
    for (vector<Product>::iterator i = products.begin();i != products.end();i++)
    {
        if (i->getName() == name) return *i;
    }
    throw logic_error("Product " + name + " not found on task " + this->name);
}

Task* Task::createTask(const string& type, const string& name, const input::Config& config)
{
    map<string,factory_func>::iterator i = tasks().find(type);

    if (i == tasks().end()) throw logic_error("Task type " + type + " not found");

    return i->second(name, config);
}

TaskDAG::~TaskDAG()
{
    for (vector<Task*>::iterator i = tasks.begin();i != tasks.end();++i) delete *i;
    tasks.clear();
}

void TaskDAG::addTask(Task* task)
{
    tasks.push_back(task);
}

void TaskDAG::execute(Arena& world)
{
    /*
     * Attempy to satisfy task requirements greedily. If we are not careful, this could produce cycles.
     */
    for (vector<Task*>::iterator t1 = tasks.begin();t1 != tasks.end();++t1)
    {
        for (vector<Product>::iterator p1 = (*t1)->getProducts().begin();p1 != (*t1)->getProducts().end();++p1)
        {
            for (vector<Requirement>::iterator r = p1->getRequirements().begin();r != p1->getRequirements().end();++r)
            {
                if (r->isFulfilled()) continue;
                for (vector<Task*>::iterator t2 = tasks.begin();t2 != tasks.end();++t2)
                {
                    for (vector<Product>::iterator p2 = (*t2)->getProducts().begin();p2 != (*t2)->getProducts().end();++p2)
                    {
                        if (r->getType() == p2->getType())
                        {
                            r->fulfil(*p2);
                        }
                    }
                }
                if (!r->isFulfilled()) ERROR("Could not fulfil requirement %s of task %s", r->getName().c_str(),(*t1)->getName().c_str());
            }
        }
    }

    //TODO: check for cycles

    /*
     * Successively search for executable tasks
     */
    while (true)
    {
        set<Task*> to_execute;

        for (vector<Task*>::iterator t = tasks.begin();t != tasks.end();++t)
        {
            bool can_execute = true;

            for (vector<Product>::iterator p = (*t)->getProducts().begin();p != (*t)->getProducts().end();++p)
            {
                for (vector<Requirement>::iterator r = p->getRequirements().begin();r != p->getRequirements().end();++r)
                {
                    if (!r->exists())
                    {
                        can_execute = false;
                        break;
                    }
                }
                if (!can_execute) break;
            }

            if (can_execute) to_execute.insert(*t);
        }

        if (to_execute.empty()) break;

        for (set<Task*>::iterator t = to_execute.begin();t != to_execute.end();++t)
        {
            PRINT("Executing task %s...", (*t)->getName().c_str());
            (*t)->run(*this, world);
            PRINT("done\n");

            for (vector<Product>::iterator p = (*t)->getProducts().begin();p != (*t)->getProducts().end();++p)
            {
                if (p->isUsed() && !p->exists()) ERROR("Product %s of task %s was not successfully produced", p->getName().c_str(), (*t)->getName().c_str());
            }

            tasks.erase(std::find(tasks.begin(), tasks.end(), *t));
            delete *t;
        }
    }

    if (!tasks.empty())
    {
        ERROR("Some tasks were not executed");
    }
}
