#ifndef _AQUARIUS_CC_LAMBDACCSDT_HPP_
#define _AQUARIUS_CC_LAMBDACCSDT_HPP_

#include "util/global.hpp"

#include "operator/st2eoperator.hpp"
#include "operator/deexcitationoperator.hpp"
#include "operator/excitationoperator.hpp"
#include "convergence/diis.hpp"
#include "util/subiterative.hpp"
#include "task/task.hpp"

namespace aquarius
{
namespace cc
{

/*
 * Solve the left-hand coupled cluster eigenvalue equation:
 *
 *               _
 * <0|L|Phi><Phi|H    |Phi> = 0
 *                open
 *
 *       _    -T   T       T
 * where X = e  X e  = (X e )
 *                           c
 */
template <typename U>
class LambdaCCSDT : public Subiterative<U>
{
    protected:
        convergence::DIIS<op::DeexcitationOperator<U,2>> diis;
//        convergence::DIIS<op::DeexcitationOperator<U,3>> diis;

    public:
        LambdaCCSDT(const string& name, input::Config& config);

        bool run(task::TaskDAG& dag, const Arena& arena);

        void iterate(const Arena& arena);
        void subiterate(const Arena& arena);
};

}
}

#endif
