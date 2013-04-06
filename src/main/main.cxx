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

#include "input/config.hpp"
#include "input/molecule.hpp"
#include "scf/aoints.hpp"
#include "scf/aoscf.hpp"
#include "scf/aomoints.hpp"
#include "scf/choleskyscf.hpp"
#include "scf/choleskymoints.hpp"
#include "cc/ccsd.hpp"
#include "cc/ccsdt.hpp"
#include "cc/lambdaccsd.hpp"
#include "cc/2edensity.hpp"
#include "operator/st2eoperator.hpp"
#include "time/time.hpp"
#include "tensor/dist_tensor.hpp"
#include "tensor/spinorbital.hpp"

using namespace std;
using namespace elem;
using namespace MPI;
using namespace aquarius::slide;
using namespace aquarius::input;
using namespace aquarius::scf;
using namespace aquarius::cc;
using namespace aquarius::op;
using namespace aquarius::time;
using namespace aquarius::tensor;

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    SLIDE::init();
    elem::Initialize(argc, argv);

    {
        int i;
        double dt;
        tCTF_World<double> ctf;

        assert(argc > 1);
        Schema schema(TOPDIR "/input_schema");
        Config config(argv[1]);
        schema.apply(config);

        Molecule mol(config);

        tic();
        CholeskyIntegrals<double> chol(ctf, config.get("cholesky"), mol);
        dt = todouble(toc());
        PRINT("Cholesky integrals took: %8.3f s\n\n", dt);

        tic();
        CholeskyUHF<double> scf(config.get("scf"), chol);

        PRINT("nA: %d\n", mol.getNumOrbitals()-mol.getNumAlphaElectrons());
        PRINT("na: %d\n", mol.getNumOrbitals()-mol.getNumBetaElectrons());
        PRINT("nI: %d\n", mol.getNumAlphaElectrons());
        PRINT("ni: %d\n", mol.getNumBetaElectrons());

        //chol.test();

        PRINT("\nUHF-SCF\n\n");
        PRINT("It.            SCF Energy     Residual Walltime\n");
        tic();
        for (i = 0;scf.iterate();i++)
        {
            dt = todouble(toc());
            PRINT("%3d % 21.15f %12.6e %8.3f\n", i+1, scf.getEnergy(), scf.getConvergence(), dt);
            tic();
        }
        toc();

        PRINT("\nUHF Orbital Energies\n\n");
        for (int i = 0;i < mol.getNumOrbitals();i++)
        {
            PRINT("%4d ", i+1);

            if (i < mol.getNumAlphaElectrons())
            {
                PRINT("%21.15f a ", scf.getAlphaEigenvalues()[i]);
            }
            else
            {
                PRINT("%21.15f   ", scf.getAlphaEigenvalues()[i]);
            }

            if (i < mol.getNumBetaElectrons())
            {
                PRINT("%21.15f b ", scf.getBetaEigenvalues()[i]);
            }
            else
            {
                PRINT("%21.15f   ", scf.getBetaEigenvalues()[i]);
            }

            PRINT("\n");
        }

        dt = todouble(toc());
        PRINT("\nCholesky SCF took: %8.3f s (%8.3f s/it.)\n", dt, dt/i);

        double s2 = scf.getS2();
        double mult = scf.getMultiplicity();
        double na = scf.getAvgNumAlpha();
        double nb = scf.getAvgNumBeta();

        PRINT("\n");
        PRINT("<0|S^2|0>     = %f\n", s2);
        PRINT("<0|2S+1|0>    = %f\n", mult);
        PRINT("<0|n_alpha|0> = %f\n", na);
        PRINT("<0|n_beta|0>  = %f\n", nb);
        PRINT("\n");

        tic();
        AOIntegrals<double> ao(ctf, mol);
        dt = todouble(toc());
        PRINT("AO integrals took: %8.3f s\n", dt);

        tic();
        AOUHF<double> aoscf(config.get("scf"), ao);

        PRINT("\nUHF-SCF\n\n");
        PRINT("It.            SCF Energy     Residual Walltime\n");
        tic();
        for (i = 0;aoscf.iterate();i++)
        {
            dt = todouble(toc());
            PRINT("%3d % 21.15f %12.6e %8.3f\n", i+1, aoscf.getEnergy(), aoscf.getConvergence(), dt);
            tic();
        }
        toc();

        PRINT("\nUHF Orbital Energies\n\n");
        for (int i = 0;i < mol.getNumOrbitals();i++)
        {
            PRINT("%4d ", i+1);

            if (i < mol.getNumAlphaElectrons())
            {
                PRINT("%21.15f a ", aoscf.getAlphaEigenvalues()[i]);
            }
            else
            {
                PRINT("%21.15f   ", aoscf.getAlphaEigenvalues()[i]);
            }

            if (i < mol.getNumBetaElectrons())
            {
                PRINT("%21.15f b ", aoscf.getBetaEigenvalues()[i]);
            }
            else
            {
                PRINT("%21.15f   ", aoscf.getBetaEigenvalues()[i]);
            }

            PRINT("\n");
        }

        dt = todouble(toc());
        PRINT("\nAO SCF took: %8.3f s (%8.3f s/it.)\n", dt, dt/i);

        s2 = aoscf.getS2();
        mult = aoscf.getMultiplicity();
        na = aoscf.getAvgNumAlpha();
        nb = aoscf.getAvgNumBeta();

        PRINT("\n");
        PRINT("<0|S^2|0>     = %f\n", s2);
        PRINT("<0|2S+1|0>    = %f\n", mult);
        PRINT("<0|n_alpha|0> = %f\n", na);
        PRINT("<0|n_beta|0>  = %f\n", nb);
        PRINT("\n");

        {
            tic();
            CholeskyMOIntegrals<double> moints(scf);
            dt = todouble(toc());
            PRINT("Cholesky MO took: %8.3f s\n\n", dt);

            CCSD<double> cholccsd(config.get("cc"), moints);

            PRINT("\nCholesky UHF-MP2 Energy: %.15f\n", cholccsd.getEnergy());

            s2 = cholccsd.getProjectedS2();
            mult = cholccsd.getProjectedMultiplicity();

            PRINT("\n");
            PRINT("<0|S^2|MP2>  = %f\n", s2);
            PRINT("<0|2S+1|MP2> = %f\n", mult);
            PRINT("\n");

            PRINT("\nCholesky UHF-CCSD\n\n");
            PRINT("It.   Correlation Energy     Residual Walltime\n");
            tic();
            tic();
            for (i = 0;cholccsd.iterate();i++)
            {
                dt = todouble(toc());
                PRINT("%3d % 20.15f %12.6e %8.3f\n", i+1, cholccsd.getEnergy(), cholccsd.getConvergence(), dt);
                tic();
            }
            toc();

            dt = todouble(toc());
            PRINT("\nCholesky CCSD took: %8.3f s (%8.3f s/it.)\n", dt, dt/i);

            s2 = cholccsd.getProjectedS2();
            mult = cholccsd.getProjectedMultiplicity();

            PRINT("\n");
            PRINT("<0|S^2|CC>  = %f\n", s2);
            PRINT("<0|2S+1|CC> = %f\n", mult);
            PRINT("\n");

            STTwoElectronOperator<double,2> cholH(moints, cholccsd);
            LambdaCCSD<double> chollambda(config.get("cc"), cholH, cholccsd, cholccsd.getEnergy());

            PRINT("Cholesky UHF-Lambda-CCSD\n\n");
            PRINT("It.   Correlation Energy     Residual Walltime\n");
            tic();
            tic();
            for (i = 0;chollambda.iterate();i++)
            {
                dt = todouble(toc());
                PRINT("%3d % 20.15f %12.6e %8.3f\n", i+1, chollambda.getEnergy(), chollambda.getConvergence(), dt);
                tic();
            }
            toc();

            dt = todouble(toc());
            PRINT("\nCholesky Lambda took: %8.3f s (%8.3f s/it.)\n", dt, dt/i);

            PRINT("\nFinal Energy: %.15f\n\n", scf.getEnergy()+cholccsd.getEnergy());
        }

        {
            tic();
            AOMOIntegrals<double> aomo(aoscf);
            dt = todouble(toc());
            PRINT("AO MO took: %8.3f s\n\n", dt);

            CCSD<double> aoccsd(config.get("cc"), aomo);

            PRINT("AO UHF-MP2 Energy: %.15f\n", aoccsd.getEnergy());

            s2 = aoccsd.getProjectedS2();
            mult = aoccsd.getProjectedMultiplicity();

            PRINT("\n");
            PRINT("<0|S^2|MP2>  = %f\n", s2);
            PRINT("<0|2S+1|MP2> = %f\n", mult);
            PRINT("\n");

            PRINT("\nAO UHF-CCSD\n\n");
            PRINT("It.   Correlation Energy     Residual Walltime\n");
            tic();
            tic();
            for (i = 0;aoccsd.iterate();i++)
            {
                dt = todouble(toc());
                PRINT("%3d % 20.15f %12.6e %8.3f\n", i+1, aoccsd.getEnergy(), aoccsd.getConvergence(), dt);
                tic();
            }
            toc();

            dt = todouble(toc());
            PRINT("\nAO CCSD took: %8.3f s (%8.3f s/it.)\n", dt, dt/i);

            s2 = aoccsd.getProjectedS2();
            mult = aoccsd.getProjectedMultiplicity();

            PRINT("\n");
            PRINT("<0|S^2|CC>  = %f\n", s2);
            PRINT("<0|2S+1|CC> = %f\n", mult);
            PRINT("\n");

            STTwoElectronOperator<double,2> aoH(aomo, aoccsd);
            LambdaCCSD<double> aolambda(config.get("cc"), aoH, aoccsd, aoccsd.getEnergy());

            PRINT("AO UHF-Lambda-CCSD\n\n");
            PRINT("It.   Correlation Energy     Residual Walltime\n");
            tic();
            tic();
            for (i = 0;aolambda.iterate();i++)
            {
                dt = todouble(toc());
                PRINT("%3d % 20.15f %12.6e %8.3f\n", i+1, aolambda.getEnergy(), aolambda.getConvergence(), dt);
                tic();
            }
            toc();

            dt = todouble(toc());
            PRINT("\nAO Lambda took: %8.3f s (%8.3f s/it.)\n", dt, dt/i);

            PRINT("\nFinal Energy: %.15f\n\n", aoscf.getEnergy()+aoccsd.getEnergy());
        }

        print_timers();
    }

    elem::Finalize();
    SLIDE::finish();
    MPI_Finalize();
}
