#ifndef _AQUARIUS_CC_CCSDTQ_1A_HPP_
#define _AQUARIUS_CC_CCSDTQ_1A_HPP_

#include "util/global.hpp"

#include "task/task.hpp"
#include "time/time.hpp"
#include "util/subiterative.hpp"
#include "operator/2eoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "convergence/diis.hpp"

#include "ccsd.hpp"

namespace aquarius
{
namespace cc
{

template <typename U>
class CCSDTQ_1a : public Subiterative<U>
{
    protected:
        convergence::DIIS<op::ExcitationOperator<U,2>> diis;

    public:
        CCSDTQ_1a(const string& name, input::Config& config);

        bool run(task::TaskDAG& dag, const Arena& arena);

        void iterate(const Arena& arena);
        void subiterate(const Arena& arena);
        void microiterate(const Arena& arena);
};

}
}

#endif
