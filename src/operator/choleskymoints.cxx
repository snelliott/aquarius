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

#include "choleskymoints.hpp"

using namespace std;
using namespace aquarius;
using namespace aquarius::scf;
using namespace aquarius::tensor;
using namespace aquarius::input;
using namespace aquarius::integrals;
using namespace aquarius::op;
using namespace aquarius::task;

template <typename T>
CholeskyMOIntegrals<T>::CholeskyMOIntegrals(const string& name, const Config& config)
: MOIntegrals<T>("choleskymoints", name, config)
{
    this->getProduct("H").addRequirement(Requirement("cholesky","cholesky"));
}

template <typename T>
void CholeskyMOIntegrals<T>::run(TaskDAG& dag, const Arena& arena)
{
    const MOSpace<T>& occ = this->template get<MOSpace<T> >("occ");
    const MOSpace<T>& vrt = this->template get<MOSpace<T> >("vrt");

    const DistTensor<T>& Fa = this->template get<DistTensor<T> >("Fa");
    const DistTensor<T>& Fb = this->template get<DistTensor<T> >("Fb");

    this->put("H", new TwoElectronOperator<T>(OneElectronOperator<T>(occ, vrt, Fa, Fb)));
    TwoElectronOperator<T>& H = this->template get<TwoElectronOperator<T> >("H");

    const CholeskyIntegrals<T>& chol = this->template get<CholeskyIntegrals<T> >("cholesky");

    const DistTensor<T>& cA = vrt.Calpha;
    const DistTensor<T>& ca = vrt.Cbeta;
    const DistTensor<T>& cI = occ.Calpha;
    const DistTensor<T>& ci = occ.Cbeta;
    const DistTensor<T>& Lpq = chol.getL();
    const DistTensor<T>& D = chol.getD();

    int N = occ.nao;
    int nI = occ.nalpha;
    int ni = occ.nbeta;
    int nA = vrt.nalpha;
    int na = vrt.nbeta;
    int R = chol.getRank();

    vector<int> sizeIIR = vec(nI, nI, R);
    vector<int> sizeiiR = vec(ni, ni, R);
    vector<int> sizeAAR = vec(nA, nA, R);
    vector<int> sizeaaR = vec(na, na, R);
    vector<int> sizeAIR = vec(nA, nI, R);
    vector<int> sizeaiR = vec(na, ni, R);

    vector<int> shapeNNN = vec(NS, NS, NS);

    DistTensor<T> LIJ(arena, 3, sizeIIR, shapeNNN, false);
    DistTensor<T> Lij(arena, 3, sizeiiR, shapeNNN, false);
    DistTensor<T> LAB(arena, 3, sizeAAR, shapeNNN, false);
    DistTensor<T> Lab(arena, 3, sizeaaR, shapeNNN, false);
    DistTensor<T> LAI(arena, 3, sizeAIR, shapeNNN, false);
    DistTensor<T> Lai(arena, 3, sizeaiR, shapeNNN, false);

    {
        vector<int> sizeNIR = vec(N, nI, R);
        vector<int> sizeNiR = vec(N, ni, R);
        vector<int> sizeNAR = vec(N, nA, R);
        vector<int> sizeNaR = vec(N, na, R);

        DistTensor<T> LpI(arena, 3, sizeNIR, shapeNNN, false);
        DistTensor<T> Lpi(arena, 3, sizeNiR, shapeNNN, false);
        DistTensor<T> LpA(arena, 3, sizeNAR, shapeNNN, false);
        DistTensor<T> Lpa(arena, 3, sizeNaR, shapeNNN, false);

        LpI["pIR"] = Lpq["pqR"]*cI["qI"];
        Lpi["piR"] = Lpq["pqR"]*ci["qi"];
        LpA["pAR"] = Lpq["pqR"]*cA["qA"];
        Lpa["paR"] = Lpq["pqR"]*ca["qa"];

        LIJ["IJR"] = LpI["pJR"]*cI["pI"];
        Lij["ijR"] = Lpi["pjR"]*ci["pi"];
        LAI["AIR"] = LpI["pIR"]*cA["pA"];
        Lai["aiR"] = Lpi["piR"]*ca["pa"];
        LAB["ABR"] = LpA["pBR"]*cA["pA"];
        Lab["abR"] = Lpa["pbR"]*ca["pa"];
    }

    DistTensor<T> LDIJ(arena, 3, sizeIIR, shapeNNN, false);
    DistTensor<T> LDij(arena, 3, sizeiiR, shapeNNN, false);
    DistTensor<T> LDAB(arena, 3, sizeAAR, shapeNNN, false);
    DistTensor<T> LDab(arena, 3, sizeaaR, shapeNNN, false);
    DistTensor<T> LDAI(arena, 3, sizeAIR, shapeNNN, false);
    DistTensor<T> LDai(arena, 3, sizeaiR, shapeNNN, false);

    LDIJ["IJR"] = D["R"]*LIJ["IJR"];
    LDij["ijR"] = D["R"]*Lij["ijR"];
    LDAI["AIR"] = D["R"]*LAI["AIR"];
    LDai["aiR"] = D["R"]*Lai["aiR"];
    LDAB["ABR"] = D["R"]*LAB["ABR"];
    LDab["abR"] = D["R"]*Lab["abR"];

    H.getIJKL()(vec(0,2),vec(0,2))["IJKL"] = 0.5*LDIJ["IKR"]*LIJ["JLR"];
    H.getIJKL()(vec(0,1),vec(0,1))["IjKl"] =     LDIJ["IKR"]*Lij["jlR"];
    H.getIJKL()(vec(0,0),vec(0,0))["ijkl"] = 0.5*LDij["ikR"]*Lij["jlR"];

    H.getIJAK()(vec(0,2),vec(1,1))["IJAK"] =  LDIJ["JKR"]*LAI["AIR"];
    H.getIJAK()(vec(0,1),vec(1,0))["IjAk"] =  LDij["jkR"]*LAI["AIR"];
    H.getIJAK()(vec(0,1),vec(0,1))["IjaK"] = -LDIJ["IKR"]*Lai["ajR"];
    H.getIJAK()(vec(0,0),vec(0,0))["ijak"] =  LDij["jkR"]*Lai["aiR"];

    H.getAIJK()(vec(1,1),vec(0,2))["AIJK"] = H.getIJAK()(vec(0,2),vec(1,1))["JKAI"];
    H.getAIJK()(vec(1,0),vec(0,1))["AiJk"] = H.getIJAK()(vec(0,1),vec(1,0))["JkAi"];
    H.getAIJK()(vec(0,1),vec(0,1))["aIJk"] = H.getIJAK()(vec(0,1),vec(0,1))["JkaI"];
    H.getAIJK()(vec(0,0),vec(0,0))["aijk"] = H.getIJAK()(vec(0,0),vec(0,0))["jkai"];

    H.getABIJ()(vec(2,0),vec(0,2))["ABIJ"] = 0.5*LDAI["AIR"]*LAI["BJR"];
    H.getABIJ()(vec(1,0),vec(0,1))["AbIj"] =     LDAI["AIR"]*Lai["bjR"];
    H.getABIJ()(vec(0,0),vec(0,0))["abij"] = 0.5*LDai["aiR"]*Lai["bjR"];

    H.getIJAB()(vec(0,2),vec(2,0))["IJAB"] = H.getABIJ()(vec(2,0),vec(0,2))["ABIJ"];
    H.getIJAB()(vec(0,1),vec(1,0))["IjAb"] = H.getABIJ()(vec(1,0),vec(0,1))["AbIj"];
    H.getIJAB()(vec(0,0),vec(0,0))["ijab"] = H.getABIJ()(vec(0,0),vec(0,0))["abij"];

    H.getAIBJ()(vec(1,1),vec(1,1))["AIBJ"]  = LDAB["ABR"]*LIJ["IJR"];
    H.getAIBJ()(vec(1,1),vec(1,1))["AIBJ"] -= LDAI["AJR"]*LAI["BIR"];
    H.getAIBJ()(vec(1,0),vec(1,0))["AiBj"]  = LDAB["ABR"]*Lij["ijR"];
    H.getAIBJ()(vec(0,1),vec(0,1))["aIbJ"]  = LDab["abR"]*LIJ["IJR"];
    H.getAIBJ()(vec(0,0),vec(0,0))["aibj"]  = LDab["abR"]*Lij["ijR"];
    H.getAIBJ()(vec(0,0),vec(0,0))["aibj"] -= LDai["ajR"]*Lai["biR"];
    H.getAIBJ()(vec(1,0),vec(0,1))["AibJ"]  = -H.getABIJ()(vec(1,0),vec(0,1))["AbJi"];
    H.getAIBJ()(vec(0,1),vec(1,0))["aIBj"]  = -H.getABIJ()(vec(1,0),vec(0,1))["BaIj"];

    H.getABCI()(vec(2,0),vec(1,1))["ABCI"] =  LDAB["ACR"]*LAI["BIR"];
    H.getABCI()(vec(1,0),vec(1,0))["AbCi"] =  LDAB["ACR"]*Lai["biR"];
    H.getABCI()(vec(1,0),vec(0,1))["AbcI"] = -LDab["bcR"]*LAI["AIR"];
    H.getABCI()(vec(0,0),vec(0,0))["abci"] =  LDab["acR"]*Lai["biR"];

    H.getAIBC()(vec(1,1),vec(2,0))["AIBC"] = H.getABCI()(vec(2,0),vec(1,1))["BCAI"];
    H.getAIBC()(vec(1,0),vec(1,0))["AiBc"] = H.getABCI()(vec(1,0),vec(1,0))["BcAi"];
    H.getAIBC()(vec(0,1),vec(1,0))["aIBc"] = H.getABCI()(vec(1,0),vec(0,1))["BcaI"];
    H.getAIBC()(vec(0,0),vec(0,0))["aibc"] = H.getABCI()(vec(0,0),vec(0,0))["bcai"];

    H.getABCD()(vec(2,0),vec(2,0))["ABCD"] = 0.5*LDAB["ACR"]*LAB["BDR"];
    H.getABCD()(vec(1,0),vec(1,0))["AbCd"] =     LDAB["ACR"]*Lab["bdR"];
    H.getABCD()(vec(0,0),vec(0,0))["abcd"] = 0.5*LDab["acR"]*Lab["bdR"];
}

INSTANTIATE_SPECIALIZATIONS(CholeskyMOIntegrals);
REGISTER_TASK(CholeskyMOIntegrals<double>,"choleskymoints");