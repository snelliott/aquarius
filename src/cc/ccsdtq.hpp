#ifndef _AQUARIUS_CC_CCSDTQ_HPP_
#define _AQUARIUS_CC_CCSDTQ_HPP_

#include "util/global.hpp"

#include "task/task.hpp"
#include "time/time.hpp"
#include "util/iterative.hpp"
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
class CCSDTQ : public Subiterative<U>
{
    protected:
        convergence::DIIS<op::ExcitationOperator<U,4>> diis;
        string guess;

    public:
        CCSDTQ(const string& name, input::Config& config);

        bool run(task::TaskDAG& dag, const Arena& arena);

        void iterate(const Arena& arena);
        void subiterate(const Arena& arena);
        void microiterate(const Arena& arena);

        /*
        double getProjectedS2() const;

        double getProjectedMultiplicity() const;
        */
};

}
}

#endif
