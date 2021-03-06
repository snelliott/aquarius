#ifndef _AQUARIUS_OPERATOR_FAKEMOINTS_HPP_
#define _AQUARIUS_OPERATOR_FAKEMOINTS_HPP_

#include "util/global.hpp"

#include "moints.hpp"

namespace aquarius
{
namespace op
{

template <typename T>
class FakeMOIntegrals : public task::Task
{
    public:
        FakeMOIntegrals(const string& name, input::Config& config);

    protected:
        int noa, nob, nva, nvb;

        bool run(task::TaskDAG& dag, const Arena& arena);
};

}
}

#endif
