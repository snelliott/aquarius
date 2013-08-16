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

#ifndef _AQUARIUS_INPUT_MOLECULE_HPP_
#define _AQUARIUS_INPUT_MOLECULE_HPP_

#include <vector>
#include <iterator>

#include "integrals/shell.hpp"
#include "symmetry/symmetry.hpp"
#include "util/util.h"
#include "task/task.hpp"

#include "config.hpp"

namespace aquarius
{
namespace input
{

class Atom;

class Molecule : public task::Resource
{
    private:
        std::vector<Atom> atoms;
        int multiplicity;
        int nelec;
        int norb;
        double nucrep;

        void addAtom(const Atom& atom);

    public:
        Molecule(const Arena& arena, const Config& config);

        void print(task::Printer& p) const {}

        int getNumElectrons() const;

        int getNumAlphaElectrons() const;

        int getNumBetaElectrons() const;

        int getMultiplicity() const;

        int getNumOrbitals() const;

        double getNuclearRepulsion() const;

        const symmetry::PointGroup& getPointGroup() const { return symmetry::PointGroup::C1(); }

        class shell_iterator : public std::iterator<std::forward_iterator_tag, integrals::Shell>
        {
            friend class Molecule;

            private:
                std::vector<Atom>::iterator atom_it;
                std::vector<Atom>::iterator atom_it_end;
                std::vector<integrals::Shell>::iterator shell_it;

            protected:
                shell_iterator(const std::vector<Atom>::iterator& atom_it,
                               const std::vector<Atom>::iterator& atom_it_end,
                               const std::vector<integrals::Shell>::iterator& shell_it);

            public:
                shell_iterator();

                shell_iterator(const shell_iterator& other);

                shell_iterator& operator=(const shell_iterator& other);

                shell_iterator& operator++();

                shell_iterator operator++(int x);

                shell_iterator operator+(int x);

                shell_iterator operator-(int x);

                integrals::Shell& operator*();

                integrals::Shell* operator->();

                bool operator<(const shell_iterator& other) const;

                bool operator==(const shell_iterator& other) const;

                bool operator!=(const shell_iterator& other) const;
        };

        class const_shell_iterator : public std::iterator<std::forward_iterator_tag, const integrals::Shell>
        {
            friend class Molecule;

            private:
                std::vector<Atom>::const_iterator atom_it;
                std::vector<Atom>::const_iterator atom_it_end;
                std::vector<integrals::Shell>::const_iterator shell_it;

            protected:
                const_shell_iterator(const std::vector<Atom>::const_iterator& atom_it,
                               const std::vector<Atom>::const_iterator& atom_it_end,
                               const std::vector<integrals::Shell>::const_iterator& shell_it);

            public:
                const_shell_iterator();

                const_shell_iterator(const const_shell_iterator& other);

                const_shell_iterator& operator=(const const_shell_iterator& other);

                const_shell_iterator& operator++();

                const_shell_iterator operator++(int x);

                const_shell_iterator operator+(int x);

                const_shell_iterator operator-(int x);

                const integrals::Shell& operator*();

                const integrals::Shell* operator->();

                bool operator<(const const_shell_iterator& other) const;

                bool operator==(const const_shell_iterator& other) const;

                bool operator!=(const const_shell_iterator& other) const;
        };

        shell_iterator getShellsBegin();

        shell_iterator getShellsEnd();

        const_shell_iterator getShellsBegin() const;

        const_shell_iterator getShellsEnd() const;

        std::vector<Atom>::iterator getAtomsBegin();

        std::vector<Atom>::iterator getAtomsEnd();

        std::vector<Atom>::const_iterator getAtomsBegin() const;

        std::vector<Atom>::const_iterator getAtomsEnd() const;
};

class Atom
{
    private:
        integrals::Center center;
        std::vector<integrals::Shell> shells;

    public:
        Atom(const integrals::Center& center);

        void addShell(const integrals::Shell& shell);

        integrals::Center& getCenter();

        const integrals::Center& getCenter() const;

        std::vector<integrals::Shell>::iterator getShellsBegin();

        std::vector<integrals::Shell>::iterator getShellsEnd();

        std::vector<integrals::Shell>::const_iterator getShellsBegin() const;

        std::vector<integrals::Shell>::const_iterator getShellsEnd() const;
};

}
}

#endif
