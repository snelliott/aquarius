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

#include "mp4dq.hpp"

using namespace std;
using namespace aquarius::op;
using namespace aquarius::cc;
using namespace aquarius::input;
using namespace aquarius::tensor;
using namespace aquarius::time;
using namespace aquarius::task;

template <typename U>
MP4DQ<U>::MP4DQ(const string& name, const Config& config)
: NonIterative("mp4dq", name, config)
{
    vector<Requirement> reqs;
    reqs.push_back(Requirement("moints", "H"));
    addProduct(Product("double", "mp2", reqs));
    addProduct(Product("double", "mp3", reqs));
    addProduct(Product("double", "mp4d", reqs));
    addProduct(Product("double", "mp4q", reqs));
    addProduct(Product("double", "energy", reqs));
    addProduct(Product("double", "S2", reqs));
    addProduct(Product("double", "multiplicity", reqs));
    addProduct(Product("mp4dq.T", "T", reqs));
    addProduct(Product("mp4dq.Hbar", "Hbar", reqs));
}

template <typename U>
void MP4DQ<U>::run(TaskDAG& dag, const Arena& arena)
{
    const TwoElectronOperator<U>& H = get<TwoElectronOperator<U> >("H");

    const Space& occ = H.occ;
    const Space& vrt = H.vrt;

    put("T", new ExcitationOperator<U,2>("T", arena, occ, vrt));
    puttmp("D", new Denominator<U>(H));
    puttmp("Z", new ExcitationOperator<U,2>("Z", arena, occ, vrt));

    ExcitationOperator<U,2>& T = get<ExcitationOperator<U,2> >("T");
    Denominator<U>& D = gettmp<Denominator<U> >("D");
    ExcitationOperator<U,2>& Z = gettmp<ExcitationOperator<U,2> >("Z");

    Z(0) = (U)0.0;
    T(0) = (U)0.0;
    T(1) = (U)0.0;
    T(2) = H.getABIJ();

    T.weight(D);

    energy = 0.25*real(scalar(H.getABIJ()*T(2)));
    double mp2energy = energy;


    Logger::log(arena) << "MP2 energy = " << setprecision(15) << energy << endl;
    put("mp2", new Scalar(arena, energy));

    TwoElectronOperator<U>& H_ = get<TwoElectronOperator<U> >("H");

    TwoElectronOperator<U> Hnew("W", H_, TwoElectronOperator<U>::AB|
                                 TwoElectronOperator<U>::IJ|
                                 TwoElectronOperator<U>::IJKL|
                                 TwoElectronOperator<U>::AIBJ);

    SpinorbitalTensor<U>& FAE = Hnew.getAB();
    SpinorbitalTensor<U>& FMI = Hnew.getIJ();
    SpinorbitalTensor<U>& WMNEF = Hnew.getIJAB();
    SpinorbitalTensor<U>& WABEF = Hnew.getABCD();
    SpinorbitalTensor<U>& WMNIJ = Hnew.getIJKL();
    SpinorbitalTensor<U>& WAMEI = Hnew.getAIBJ();

    Z(2)["abij"] = WMNEF["ijab"];
    Z(2)["abij"] += FAE["af"]*T(2)["fbij"];
    Z(2)["abij"] -= FMI["ni"]*T(2)["abnj"];
    Z(2)["abij"] += 0.5*WABEF["abef"]*T(2)["efij"];
    Z(2)["abij"] += 0.5*WMNIJ["mnij"]*T(2)["abmn"];
    Z(2)["abij"] -= WAMEI["amei"]*T(2)["ebmj"];

    Z.weight(D);
    T += Z;

    energy = 0.25*real(scalar(H.getABIJ()*T(2)));
    double mp3energy = energy;

    //Logger::log(arena) << "MP3 energy = " << setprecision(15) << energy << endl;
    Logger::log(arena) << "MP3 correlation energy = " << setprecision(15) << energy - mp2energy << endl;
    put("mp3", new Scalar(arena, energy));



    ExcitationOperator<U,2>& Znew = gettmp<ExcitationOperator<U,2> >("Z");

    Znew(2)["abij"] = WMNEF["ijab"];
    Znew(2)["abij"] += FAE["af"]*T(2)["fbij"];
    Znew(2)["abij"] -= FMI["ni"]*T(2)["abnj"];
    Znew(2)["abij"] += 0.5*WABEF["abef"]*T(2)["efij"];
    Znew(2)["abij"] += 0.5*WMNIJ["mnij"]*T(2)["abmn"];
    Znew(2)["abij"] -= WAMEI["amei"]*T(2)["ebmj"];

    Znew.weight(D);
    T += Znew;

    energy = 0.25*real(scalar(H.getABIJ()*T(2)));
    double mp4denergy = energy - mp3energy;

    //Logger::log(arena) << "LCCD(2) energy = " << setprecision(15) << energy << endl;
    Logger::log(arena) << "MP4D correlation energy = " << setprecision(15) << energy - mp3energy << endl;
    put("mp4d", new Scalar(arena, energy));



    ExcitationOperator<U,2>& Tccd = get<ExcitationOperator<U,2> >("T");
    Denominator<U>& Dccd = gettmp<Denominator<U> >("D");
    ExcitationOperator<U,2>& Zccd = gettmp<ExcitationOperator<U,2> >("Z");

    Zccd(0) = (U)0.0;
    Tccd(0) = (U)0.0;
    Tccd(1) = (U)0.0;
    Tccd(2) = H.getABIJ();

    Tccd.weight(Dccd);


    FMI["mi"] += 0.5*WMNEF["mnef"]*T(2)["efin"];
    WMNIJ["mnij"] += 0.5*WMNEF["mnef"]*T(2)["efij"];
    FAE["ae"] -= 0.5*WMNEF["mnef"]*T(2)["afmn"];
    WAMEI["amei"] -= 0.5*WMNEF["mnef"]*T(2)["afin"];
    Z(2)["abij"] = WMNEF["ijab"];
    Z(2)["abij"] += FAE["af"]*T(2)["fbij"];
    Z(2)["abij"] -= FMI["ni"]*T(2)["abnj"];
    Z(2)["abij"] += 0.5*WABEF["abef"]*T(2)["efij"];
    Z(2)["abij"] += 0.5*WMNIJ["mnij"]*T(2)["abmn"];
    Z(2)["abij"] -= WAMEI["amei"]*T(2)["ebmj"];

    Z.weight(D);
    T += Z;

    energy = 0.25*real(scalar(H.getABIJ()*T(2)));
    double mp4qenergy = energy - mp3energy;

    //Logger::log(arena) << "CCD(1) energy = " << setprecision(15) << energy << endl;
    Logger::log(arena) << "MP4Q correlation energy = " << setprecision(15) << energy - mp3energy << endl;
    Logger::log(arena) << "MP4DQ correlation energy = " << setprecision(15) << mp4denergy + mp4qenergy << endl;
    put("mp4q", new Scalar(arena, energy));

    /*
    if (isUsed("S2") || isUsed("multiplicity"))
    {
        double s2 = getProjectedS2(occ, vrt, T(1), T(2));
        double mult = sqrt(4*s2+1);

        put("S2", new Scalar(arena, s2));
        put("multiplicity", new Scalar(arena, mult));
    }
    */

    put("energy", new Scalar(arena, energy));

    if (isUsed("Hbar"))
    {
        put("Hbar", new STTwoElectronOperator<U,2>("Hbar", H, T, true));
    }
}

#if 0
template <typename U>
void MP4DQ<U>::iterate(const Arena& arena)
{
    TwoElectronOperator<U>& H_ = get<TwoElectronOperator<U> >("H");

    ExcitationOperator<U,2>& T = get<ExcitationOperator<U,2> >("T");
    Denominator<U>& D = gettmp<Denominator<U> >("D");
    ExcitationOperator<U,2>& Z = gettmp<ExcitationOperator<U,2> >("Z");

    TwoElectronOperator<U> H("W", H_, TwoElectronOperator<U>::AB|
                                 TwoElectronOperator<U>::IJ|
                                 TwoElectronOperator<U>::IJKL|
                                 TwoElectronOperator<U>::AIBJ);

    SpinorbitalTensor<U>& FAE = H.getAB();
    SpinorbitalTensor<U>& FMI = H.getIJ();
    SpinorbitalTensor<U>& WMNEF = H.getIJAB();
    SpinorbitalTensor<U>& WABEF = H.getABCD();
    SpinorbitalTensor<U>& WMNIJ = H.getIJKL();
    SpinorbitalTensor<U>& WAMEI = H.getAIBJ();

//    sched.set_max_partitions(1);
    /**************************************************************************
     *
     * Intermediates
     */
    // FMI["mi"] += 0.5*WMNEF["mnef"]*T(2)["efin"];


    // WMNIJ["mnij"] += 0.5*WMNEF["mnef"]*T(2)["efij"];
    // FAE["ae"] -= 0.5*WMNEF["mnef"]*T(2)["afmn"];
    // WAMEI["amei"] -= 0.5*WMNEF["mnef"]*T(2)["afin"];
    /*
     *************************************************************************/

    /**************************************************************************
     *
     * T(1)->T(2) and T(2)->T(2)
     */
    Z(2)["abij"] = WMNEF["ijab"];
    Z(2)["abij"] += FAE["af"]*T(2)["fbij"];
    Z(2)["abij"] -= FMI["ni"]*T(2)["abnj"];
    Z(2)["abij"] += 0.5*WABEF["abef"]*T(2)["efij"];
    Z(2)["abij"] += 0.5*WMNIJ["mnij"]*T(2)["abmn"];
    Z(2)["abij"] -= WAMEI["amei"]*T(2)["ebmj"];
    /*
     *************************************************************************/

    Z.weight(D);
    T += Z;

    energy = 0.25*real(scalar(H.getABIJ()*T(2)));

    conv = Z.norm(00);

    diis.extrapolate(T, Z);
}
#endif

INSTANTIATE_SPECIALIZATIONS(MP4DQ);
REGISTER_TASK(MP4DQ<double>,"mp4dq");
