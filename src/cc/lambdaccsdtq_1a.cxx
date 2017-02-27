#include "lambdaccsdtq_1a.hpp"

using namespace aquarius::op;
using namespace aquarius::input;
using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::time;
using namespace aquarius::symmetry;

namespace aquarius
{
namespace cc
{

template <typename U>
LambdaCCSDTQ_1a<U>::LambdaCCSDTQ_1a(const string& name, Config& config)
: Subiterative<U>(name, config), diis(config.get("diis"))
{
    vector<Requirement> reqs;
    reqs.push_back(Requirement("moints", "H"));
    reqs.push_back(Requirement("ccsdtq-1a.Hbar", "Hbar"));
    reqs.push_back(Requirement("ccsdtq-1a.T", "T"));
    this->addProduct(Product("double", "energy", reqs));
    this->addProduct(Product("double", "convergence", reqs));
    this->addProduct(Product("ccsdtq-1a.L", "L", reqs));
    this->addProduct(Product("ccsdtq-1a.L3", "L3", reqs));
}

template <typename U>
bool LambdaCCSDTQ_1a<U>::run(TaskDAG& dag, const Arena& arena)
{
    const auto& H    = this->template get<  TwoElectronOperator<U>>("H");
    const auto& Hbar = this->template get<STTwoElectronOperator<U>>("Hbar");

    const Space& occ = H.occ;
    const Space& vrt = H.vrt;
    const PointGroup& group = occ.group;

    this->put   (      "L", new DeexcitationOperator<U,2>("L", arena, occ, vrt));
    this->puttmp(      "D", new Denominator         <U  >(H));
    this->puttmp(      "Z", new DeexcitationOperator<U,2>("Z", arena, occ, vrt));
    this->puttmp(      "Q", new DeexcitationOperator<U,3>("Q", arena, occ, vrt));
    this->puttmp(      "q", new DeexcitationOperator<U,2>("q", arena, occ, vrt));
    this->puttmp(   "Z3", new SpinorbitalTensor   <U>("Z3", arena,
                                                  H.getABIJ().getGroup(),
                                                  {vrt, occ}, {0, 3},
                                                  {3, 0}));
    this->put   (   "L3", new SpinorbitalTensor   <U>("L3", arena,
                                                  H.getABIJ().getGroup(),
                                                  {vrt, occ}, {0, 3},
                                                  {3, 0}));
    this->puttmp(    "DAB", new SpinorbitalTensor   <U  >(     "D(ab)", arena, group, {vrt,occ}, {1,0}, {1,0}));
    this->puttmp(    "DIJ", new SpinorbitalTensor   <U  >(     "D(ij)", arena, group, {vrt,occ}, {0,1}, {0,1}));
    this->puttmp(    "DAI", new SpinorbitalTensor   <U  >(     "D(ai)", arena, group, {vrt,occ}, {1,0}, {0,1}));
    this->puttmp(  "GABCD", new SpinorbitalTensor   <U  >(  "G(ab,cd)", arena, group, {vrt,occ}, {2,0}, {2,0}));
    this->puttmp(  "GAIBJ", new SpinorbitalTensor   <U  >(  "G(ai,bj)", arena, group, {vrt,occ}, {1,1}, {1,1}));
    this->puttmp(  "GIJKL", new SpinorbitalTensor   <U  >(  "G(ij,kl)", arena, group, {vrt,occ}, {0,2}, {0,2}));
    this->puttmp(  "GAIBC", new SpinorbitalTensor   <U  >(  "G(ai,bc)", arena, group, {vrt,occ}, {1,1}, {2,0}));
    this->puttmp(  "GIJAK", new SpinorbitalTensor   <U  >(  "G(ij,ak)", arena, group, {vrt,occ}, {0,2}, {1,1}));
    this->puttmp(  "GABCI", new SpinorbitalTensor   <U  >(  "G(ab,ci)", arena, group, {vrt,occ}, {2,0}, {1,1}));
    this->puttmp(  "GAIJK", new SpinorbitalTensor   <U  >(  "G(ai,jk)", arena, group, {vrt,occ}, {1,1}, {0,2}));
    this->puttmp("WABCEJK", new SpinorbitalTensor   <U  >("W(abc,ejk)", arena, group, {vrt,occ}, {3,0}, {1,2}));
    this->puttmp("WABMIJK", new SpinorbitalTensor   <U  >("W(abm,ijk)", arena, group, {vrt,occ}, {2,1}, {0,3}));

    this->puttmp("L4", new SpinorbitalTensor<U>("L(ijkl,abcd)", arena, group, {vrt, occ}, {0, 4}, {4, 0}));

    auto& T = this->template get   <ExcitationOperator  <U,3>>("T");
    auto& L = this->template get   <DeexcitationOperator<U,2>>("L");
    auto& L3 = this->template get   <SpinorbitalTensor   <U >>("L3");
    auto& Z = this->template gettmp<DeexcitationOperator<U,2>>("Z");
    auto& D = this->template gettmp<Denominator         <U  >>("D");

    Z(0) = 0;
    L(0) = 1;
    L(1)[    "ia"] = T(1)[  "ai"];
    L(2)[  "ijab"] = T(2)["abij"];
    L3["ijkabc"] = 0;

    const SpinorbitalTensor<U>& VABEF = H.getABCD();
    const SpinorbitalTensor<U>& VMNIJ = H.getIJKL();
    const SpinorbitalTensor<U>& VAMEI = H.getAIBJ();

    auto& WABCEJK = this->template gettmp<SpinorbitalTensor<U>>("WABCEJK");
    auto& WABMIJK = this->template gettmp<SpinorbitalTensor<U>>("WABMIJK");

    WABCEJK["abcejk"]  = VABEF["abef"]*T(2)[    "fcjk"];
    WABCEJK["abcejk"] -= VAMEI["amej"]*T(2)[    "bcmk"];

    WABMIJK["abmijk"]  = VAMEI["amek"]*T(2)[    "ebij"];
    WABMIJK["abmijk"] -= VMNIJ["nmjk"]*T(2)[    "abin"];

    Subiterative<U>::run(dag, arena);

    this->put("energy", new U(this->energy()));
    this->put("convergence", new U(this->conv()));

    return true;
}

template <typename U>
void LambdaCCSDTQ_1a<U>::iterate(const Arena& arena)
{
    const auto& H    = this->template get<  TwoElectronOperator<U>>("H");
    const auto& Hbar = this->template get<STTwoElectronOperator<U>>("Hbar");

    const SpinorbitalTensor<U>&   FME =   Hbar.getIA();
    const SpinorbitalTensor<U>&   FAE =   Hbar.getAB();
    const SpinorbitalTensor<U>&   FMI =   Hbar.getIJ();
    const SpinorbitalTensor<U>& WMNEF = Hbar.getIJAB();
    const SpinorbitalTensor<U>& WAMEF = Hbar.getAIBC();
    const SpinorbitalTensor<U>& WABEJ = Hbar.getABCI();
    const SpinorbitalTensor<U>& WABEF = Hbar.getABCD();
    const SpinorbitalTensor<U>& WMNIJ = Hbar.getIJKL();
    const SpinorbitalTensor<U>& WMNEJ = Hbar.getIJAK();
    const SpinorbitalTensor<U>& WAMIJ = Hbar.getAIJK();
    const SpinorbitalTensor<U>& WAMEI = Hbar.getAIBJ();

    const SpinorbitalTensor<U>&   fME =   H.getIA();
    const SpinorbitalTensor<U>& VMNEF = H.getIJAB();
    const SpinorbitalTensor<U>& VABEJ = H.getABCI();
    const SpinorbitalTensor<U>& VAMIJ = H.getAIJK();

    auto& T = this->template get   <ExcitationOperator  <U,3>>("T");
    auto& L = this->template get   <DeexcitationOperator<U,2>>("L");
    auto& D = this->template gettmp<Denominator         <U  >>("D");
    auto& Z = this->template gettmp<DeexcitationOperator<U,2>>("Z");
    auto& Q = this->template gettmp<DeexcitationOperator<U,3>>("Q");
    auto& L3 = this->template get   <SpinorbitalTensor<U  >>("L3");
    auto& Z3 = this->template gettmp<SpinorbitalTensor<U  >>("Z3");

    auto& L4 = this->template gettmp<SpinorbitalTensor<U>>("L4");

    auto&     DIJ = this->template gettmp<SpinorbitalTensor<U>>(    "DIJ");
    auto&     DAB = this->template gettmp<SpinorbitalTensor<U>>(    "DAB");
    auto&     DAI = this->template gettmp<SpinorbitalTensor<U>>(    "DAI");
    auto&   GABCD = this->template gettmp<SpinorbitalTensor<U>>(  "GABCD");
    auto&   GAIBJ = this->template gettmp<SpinorbitalTensor<U>>(  "GAIBJ");
    auto&   GIJKL = this->template gettmp<SpinorbitalTensor<U>>(  "GIJKL");
    auto&   GAIBC = this->template gettmp<SpinorbitalTensor<U>>(  "GAIBC");
    auto&   GIJAK = this->template gettmp<SpinorbitalTensor<U>>(  "GIJAK");
    auto&   GABCI = this->template gettmp<SpinorbitalTensor<U>>(  "GABCI");
    auto&   GAIJK = this->template gettmp<SpinorbitalTensor<U>>(  "GAIJK");
    auto& WABCEJK = this->template gettmp<SpinorbitalTensor<U>>("WABCEJK");
    auto& WABMIJK = this->template gettmp<SpinorbitalTensor<U>>("WABMIJK");

    /***************************************************************************
     *
     * Intermediates for Lambda-CCSD
     */
    DIJ["ij"]  =  0.5*T(2)["efjm"]*L(2)["imef"];
    DAB["ab"]  = -0.5*T(2)["aemn"]*L(2)["mnbe"];
    /*
     **************************************************************************/

    /***************************************************************************
     *
     * Lambda-CCSD iteration
     */
    Z(1)[  "ia"]  =       FME[  "ia"];
    Z(1)[  "ia"] +=       FAE[  "ea"]*L(1)[  "ie"];
    Z(1)[  "ia"] -=       FMI[  "im"]*L(1)[  "ma"];
    Z(1)[  "ia"] -=     WAMEI["eiam"]*L(1)[  "me"];
    Z(1)[  "ia"] += 0.5*WABEJ["efam"]*L(2)["imef"];
    Z(1)[  "ia"] -= 0.5*WAMIJ["eimn"]*L(2)["mnea"];
    Z(1)[  "ia"] -=     WMNEJ["inam"]* DIJ[  "mn"];
    Z(1)[  "ia"] -=     WAMEF["fiea"]* DAB[  "ef"];

    Z(2)["ijab"]  =     WMNEF["ijab"];
    Z(2)["ijab"] +=       FME[  "ia"]*L(1)[  "jb"];
    Z(2)["ijab"] +=     WAMEF["ejab"]*L(1)[  "ie"];
    Z(2)["ijab"] -=     WMNEJ["ijam"]*L(1)[  "mb"];
    Z(2)["ijab"] +=       FAE[  "ea"]*L(2)["ijeb"];
    Z(2)["ijab"] -=       FMI[  "im"]*L(2)["mjab"];
    Z(2)["ijab"] += 0.5*WABEF["efab"]*L(2)["ijef"];
    Z(2)["ijab"] += 0.5*WMNIJ["ijmn"]*L(2)["mnab"];
    Z(2)["ijab"] +=     WAMEI["eiam"]*L(2)["mjbe"];
    Z(2)["ijab"] -=     WMNEF["mjab"]* DIJ[  "im"];
    Z(2)["ijab"] +=     WMNEF["ijeb"]* DAB[  "ea"];
    /*
     **************************************************************************/

    /***************************************************************************
     *
     * Intermediates for Lambda-CCSDT
     */
      DIJ[  "ij"]  =  (1.0/12.0)*T(3)["efgjmn"]* L3["imnefg"];
      DAB[  "ab"]  = -(1.0/12.0)*T(3)["aefmno"]* L3["mnobef"];

    GABCD["abcd"]  =   (1.0/6.0)*T(3)["abemno"]* L3["mnocde"];
    GAIBJ["aibj"]  =       -0.25*T(3)["aefjmn"]* L3["imnbef"];
    GIJKL["ijkl"]  =   (1.0/6.0)*T(3)["efgklm"]* L3["ijmefg"];

    GIJAK["ijak"]  =         0.5*T(2)[  "efkm"]* L3["ijmaef"];
    GAIBC["aibc"]  =        -0.5*T(2)[  "aemn"]* L3["minbce"];

      DAI[  "ai"]  =        0.25*T(3)["aefimn"]* L(2)[  "mnef"];
      DAI[  "ai"] -=         0.5*T(2)[  "eamn"]*GIJAK[  "mnei"];
    /*
     **************************************************************************/

    /***************************************************************************
     *
     * Lambda-CCSDT iteration
     */
    Z(1)[    "ia"] +=       FME[  "ie"]*  DAB[    "ea"];
    Z(1)[    "ia"] -=       FME[  "ma"]*  DIJ[    "im"];
    Z(1)[    "ia"] -=     WMNEJ["inam"]*  DIJ[    "mn"];
    Z(1)[    "ia"] -=     WAMEF["fiea"]*  DAB[    "ef"];
    Z(1)[    "ia"] +=     WMNEF["miea"]*  DAI[    "em"];
    Z(1)[    "ia"] -= 0.5*WABEF["efga"]*GAIBC[  "gief"];
    Z(1)[    "ia"] +=     WAMEI["eifm"]*GAIBC[  "fmea"];
    Z(1)[    "ia"] -=     WAMEI["eman"]*GIJAK[  "inem"];
    Z(1)[    "ia"] += 0.5*WMNIJ["imno"]*GIJAK[  "noam"];
    Z(1)[    "ia"] -= 0.5*WAMEF["gief"]*GABCD[  "efga"];
    Z(1)[    "ia"] +=     WAMEF["fmea"]*GAIBJ[  "eifm"];
    Z(1)[    "ia"] -=     WMNEJ["inem"]*GAIBJ[  "eman"];
    Z(1)[    "ia"] += 0.5*WMNEJ["noam"]*GIJKL[  "imno"];

    Z(2)[  "ijab"] -=     WMNEF["mjab"]*  DIJ[    "im"];
    Z(2)[  "ijab"] +=     WMNEF["ijeb"]*  DAB[    "ea"];
    Z(2)[  "ijab"] += 0.5*WMNEF["ijef"]*GABCD[  "efab"];
    Z(2)[  "ijab"] +=     WMNEF["imea"]*GAIBJ[  "ejbm"];
    Z(2)[  "ijab"] += 0.5*WMNEF["mnab"]*GIJKL[  "ijmn"];
    Z(2)[  "ijab"] -=     WAMEF["fiae"]*GAIBC[  "ejbf"];
    Z(2)[  "ijab"] -=     WMNEJ["ijem"]*GAIBC[  "emab"];
    Z(2)[  "ijab"] -=     WAMEF["emab"]*GIJAK[  "ijem"];
    Z(2)[  "ijab"] -=     WMNEJ["niam"]*GIJAK[  "mjbn"];
    Z(2)[  "ijab"] += 0.5*WABEJ["efbm"]* L3["ijmaef"];
    Z(2)[  "ijab"] -= 0.5*WAMIJ["ejnm"]* L3["imnabe"];

    Z3["ijkabc"]  =     WMNEF["ijab"]* L(1)[    "kc"];
    Z3["ijkabc"] +=       FME[  "ia"]* L(2)[  "jkbc"];
    Z3["ijkabc"] +=     WAMEF["ekbc"]* L(2)[  "ijae"];
    Z3["ijkabc"] -=     WMNEJ["ijam"]* L(2)[  "mkbc"];
    Z3["ijkabc"] +=     WMNEF["ijae"]*GAIBC[  "ekbc"];
    Z3["ijkabc"] -=     WMNEF["mkbc"]*GIJAK[  "ijam"];
    Z3["ijkabc"] +=       FAE[  "ea"]* L3["ijkebc"];
    Z3["ijkabc"] -=       FMI[  "im"]* L3["mjkabc"];
    Z3["ijkabc"] += 0.5*WABEF["efab"]* L3["ijkefc"];
    Z3["ijkabc"] += 0.5*WMNIJ["ijmn"]* L3["mnkabc"];
    Z3["ijkabc"] +=     WAMEI["eiam"]* L3["mjkbec"];
    /*
     **************************************************************************/

    /***************************************************************************
     *
     * Lambda-CCSDTQ-1a iteration
     */
      L4["ijklabcd"]  =              VMNEF[  "ijab"]*L(2)[    "klcd"];
      L4["ijklabcd"] +=                fME[    "ia"]*L3[  "jklbcd"];

    L4.weight({&D.getDA(), &D.getDI()}, {&D.getDa(), &D.getDi()});

    Q(2)[    "ijab"]  = (1.0/12.0)*WABCEJK["efgamn"]*  L4["ijmnebfg"];
    Q(2)[    "ijab"] -= (1.0/12.0)*WABMIJK["efjmno"]*  L4["mnioefab"];

    Q(3)[  "ijkabc"]  = (1.0/ 2.0)*  VABEJ[  "efam"]*  L4["ijkmebcf"];
    Q(3)[  "ijkabc"] -= (1.0/ 2.0)*  VAMIJ[  "eknm"]*  L4["ijmnabce"];
      
    Z(2)["ijab"] +=                                           Q(2)["ijab"];
    Z3["ijkabc"] +=                                         Q(3)["ijkabc"];
    /*
     **************************************************************************/
    
    Z3.weight({&D.getDA(),&D.getDI()},{&D.getDA(),&D.getDI()});
    L3 += Z3;

    Z.weight(D);
    L += Z;

    this->energy() = 0.25*real(scalar(conj(WMNEF)*L(2)));
    this->conv() = Z.norm(00);

    diis.extrapolate(L, Z);
}

template <typename U>
void LambdaCCSDTQ_1a<U>::subiterate(const Arena& arena)
{
    const auto& H    = this->template get<  TwoElectronOperator<U>>("H");
    const auto& Hbar = this->template get<STTwoElectronOperator<U>>("Hbar");

    const SpinorbitalTensor<U>&   FME =   Hbar.getIA();
    const SpinorbitalTensor<U>&   FAE =   Hbar.getAB();
    const SpinorbitalTensor<U>&   FMI =   Hbar.getIJ();
    const SpinorbitalTensor<U>& WMNEF = Hbar.getIJAB();
    const SpinorbitalTensor<U>& WAMEF = Hbar.getAIBC();
    const SpinorbitalTensor<U>& WABEJ = Hbar.getABCI();
    const SpinorbitalTensor<U>& WABEF = Hbar.getABCD();
    const SpinorbitalTensor<U>& WMNIJ = Hbar.getIJKL();
    const SpinorbitalTensor<U>& WMNEJ = Hbar.getIJAK();
    const SpinorbitalTensor<U>& WAMIJ = Hbar.getAIJK();
    const SpinorbitalTensor<U>& WAMEI = Hbar.getAIBJ();

    const SpinorbitalTensor<U>&   fME =   H.getIA();
    const SpinorbitalTensor<U>& VMNEF = H.getIJAB();
    const SpinorbitalTensor<U>& VABEJ = H.getABCI();
    const SpinorbitalTensor<U>& VAMIJ = H.getAIJK();

    auto& T = this->template get   <ExcitationOperator  <U,3>>("T");
    auto& L = this->template get   <DeexcitationOperator<U,2>>("L");
    auto& D = this->template gettmp<Denominator         <U  >>("D");
    auto& Z = this->template gettmp<DeexcitationOperator<U,2>>("Z");
    auto& Q = this->template gettmp<DeexcitationOperator<U,3>>("Q");
    auto& q = this->template gettmp<DeexcitationOperator<U,2>>("q");
    auto& L3 = this->template get   <SpinorbitalTensor<U  >>("L3");
    auto& Z3 = this->template gettmp<SpinorbitalTensor<U  >>("Z3");


    auto&     DIJ = this->template gettmp<SpinorbitalTensor<U>>(    "DIJ");
    auto&     DAB = this->template gettmp<SpinorbitalTensor<U>>(    "DAB");
    auto&     DAI = this->template gettmp<SpinorbitalTensor<U>>(    "DAI");
    auto&   GABCD = this->template gettmp<SpinorbitalTensor<U>>(  "GABCD");
    auto&   GAIBJ = this->template gettmp<SpinorbitalTensor<U>>(  "GAIBJ");
    auto&   GIJKL = this->template gettmp<SpinorbitalTensor<U>>(  "GIJKL");
    auto&   GAIBC = this->template gettmp<SpinorbitalTensor<U>>(  "GAIBC");
    auto&   GIJAK = this->template gettmp<SpinorbitalTensor<U>>(  "GIJAK");
    auto&   GABCI = this->template gettmp<SpinorbitalTensor<U>>(  "GABCI");
    auto&   GAIJK = this->template gettmp<SpinorbitalTensor<U>>(  "GAIJK");

    /***************************************************************************
     *
     * Intermediates for Lambda-CCSD
     */
    DIJ["ij"]  =  0.5*T(2)["efjm"]*L(2)["imef"];
    DAB["ab"]  = -0.5*T(2)["aemn"]*L(2)["mnbe"];
    /*
     **************************************************************************/

    /***************************************************************************
     *
     * Lambda-CCSD iteration
     */
    Z(1)[  "ia"]  =       FME[  "ia"];
    Z(1)[  "ia"] +=       FAE[  "ea"]*L(1)[  "ie"];
    Z(1)[  "ia"] -=       FMI[  "im"]*L(1)[  "ma"];
    Z(1)[  "ia"] -=     WAMEI["eiam"]*L(1)[  "me"];
    Z(1)[  "ia"] += 0.5*WABEJ["efam"]*L(2)["imef"];
    Z(1)[  "ia"] -= 0.5*WAMIJ["eimn"]*L(2)["mnea"];
    Z(1)[  "ia"] -=     WMNEJ["inam"]* DIJ[  "mn"];
    Z(1)[  "ia"] -=     WAMEF["fiea"]* DAB[  "ef"];

    Z(2)["ijab"]  =     WMNEF["ijab"];
    Z(2)["ijab"] +=       FME[  "ia"]*L(1)[  "jb"];
    Z(2)["ijab"] +=     WAMEF["ejab"]*L(1)[  "ie"];
    Z(2)["ijab"] -=     WMNEJ["ijam"]*L(1)[  "mb"];
    Z(2)["ijab"] +=       FAE[  "ea"]*L(2)["ijeb"];
    Z(2)["ijab"] -=       FMI[  "im"]*L(2)["mjab"];
    Z(2)["ijab"] += 0.5*WABEF["efab"]*L(2)["ijef"];
    Z(2)["ijab"] += 0.5*WMNIJ["ijmn"]*L(2)["mnab"];
    Z(2)["ijab"] +=     WAMEI["eiam"]*L(2)["mjbe"];
    Z(2)["ijab"] -=     WMNEF["mjab"]* DIJ[  "im"];
    Z(2)["ijab"] +=     WMNEF["ijeb"]* DAB[  "ea"];
    /*
     **************************************************************************/

    /***************************************************************************
     *
     * Intermediates for Lambda-CCSDT
     */
      DIJ[  "ij"]  =  (1.0/12.0)*T(3)["efgjmn"]* L3["imnefg"];
      DAB[  "ab"]  = -(1.0/12.0)*T(3)["aefmno"]* L3["mnobef"];

    GABCD["abcd"]  =   (1.0/6.0)*T(3)["abemno"]* L3["mnocde"];
    GAIBJ["aibj"]  =       -0.25*T(3)["aefjmn"]* L3["imnbef"];
    GIJKL["ijkl"]  =   (1.0/6.0)*T(3)["efgklm"]* L3["ijmefg"];

    GIJAK["ijak"]  =         0.5*T(2)[  "efkm"]* L3["ijmaef"];
    GAIBC["aibc"]  =        -0.5*T(2)[  "aemn"]* L3["minbce"];

      DAI[  "ai"]  =        0.25*T(3)["aefimn"]* L(2)[  "mnef"];
      DAI[  "ai"] -=         0.5*T(2)[  "eamn"]*GIJAK[  "mnei"];
    /*
     **************************************************************************/

    /***************************************************************************
     *
     * Lambda-CCSDT iteration
     */
    if (this->config.get<int>("micro_iterations") == 0 )
    {
        Z(1)[    "ia"] +=       FME[  "ie"]*  DAB[    "ea"];
        Z(1)[    "ia"] -=       FME[  "ma"]*  DIJ[    "im"];
        Z(1)[    "ia"] -=     WMNEJ["inam"]*  DIJ[    "mn"];
        Z(1)[    "ia"] -=     WAMEF["fiea"]*  DAB[    "ef"];
        Z(1)[    "ia"] +=     WMNEF["miea"]*  DAI[    "em"];
        Z(1)[    "ia"] -= 0.5*WABEF["efga"]*GAIBC[  "gief"];
        Z(1)[    "ia"] +=     WAMEI["eifm"]*GAIBC[  "fmea"];
        Z(1)[    "ia"] -=     WAMEI["eman"]*GIJAK[  "inem"];
        Z(1)[    "ia"] += 0.5*WMNIJ["imno"]*GIJAK[  "noam"];
        Z(1)[    "ia"] -= 0.5*WAMEF["gief"]*GABCD[  "efga"];
        Z(1)[    "ia"] +=     WAMEF["fmea"]*GAIBJ[  "eifm"];
        Z(1)[    "ia"] -=     WMNEJ["inem"]*GAIBJ[  "eman"];
        Z(1)[    "ia"] += 0.5*WMNEJ["noam"]*GIJKL[  "imno"];

        Z(2)[  "ijab"] -=     WMNEF["mjab"]*  DIJ[    "im"];
        Z(2)[  "ijab"] +=     WMNEF["ijeb"]*  DAB[    "ea"];
        Z(2)[  "ijab"] += 0.5*WMNEF["ijef"]*GABCD[  "efab"];
        Z(2)[  "ijab"] +=     WMNEF["imea"]*GAIBJ[  "ejbm"];
        Z(2)[  "ijab"] += 0.5*WMNEF["mnab"]*GIJKL[  "ijmn"];
        Z(2)[  "ijab"] -=     WAMEF["fiae"]*GAIBC[  "ejbf"];
        Z(2)[  "ijab"] -=     WMNEJ["ijem"]*GAIBC[  "emab"];
        Z(2)[  "ijab"] -=     WAMEF["emab"]*GIJAK[  "ijem"];
        Z(2)[  "ijab"] -=     WMNEJ["niam"]*GIJAK[  "mjbn"];
        Z(2)[  "ijab"] += 0.5*WABEJ["efbm"]* L3["ijmaef"];
        Z(2)[  "ijab"] -= 0.5*WAMIJ["ejnm"]* L3["imnabe"];
    }

    Z3["ijkabc"]  =     WMNEF["ijab"]* L(1)[    "kc"];
    Z3["ijkabc"] +=       FME[  "ia"]* L(2)[  "jkbc"];
    Z3["ijkabc"] +=     WAMEF["ekbc"]* L(2)[  "ijae"];
    Z3["ijkabc"] -=     WMNEJ["ijam"]* L(2)[  "mkbc"];
    Z3["ijkabc"] +=     WMNEF["ijae"]*GAIBC[  "ekbc"];
    Z3["ijkabc"] -=     WMNEF["mkbc"]*GIJAK[  "ijam"];
    Z3["ijkabc"] +=       FAE[  "ea"]* L3["ijkebc"];
    Z3["ijkabc"] -=       FMI[  "im"]* L3["mjkabc"];
    Z3["ijkabc"] += 0.5*WABEF["efab"]* L3["ijkefc"];
    Z3["ijkabc"] += 0.5*WMNIJ["ijmn"]* L3["mnkabc"];
    Z3["ijkabc"] +=     WAMEI["eiam"]* L3["mjkbec"];
    
    Z(1)[    "ia"] +=                                   Q(1)[  "ia"];
    Z(2)[  "ijab"] +=                                   Q(2)["ijab"];
      Z3["ijkabc"] +=                                 Q(3)["ijkabc"];
    /*
     **************************************************************************/
    
    Z3.weight({&D.getDA(),&D.getDI()},{&D.getDA(),&D.getDI()});
    L3 += Z3;

    if (this->config.get<int>("micro_iterations") != 0 )
    {
        q(1)[    "ia"]  =       FME[  "ie"]*  DAB[    "ea"]; 
        q(1)[    "ia"] -=       FME[  "ma"]*  DIJ[    "im"]; 
        q(1)[    "ia"] -=     WMNEJ["inam"]*  DIJ[    "mn"]; 
        q(1)[    "ia"] -=     WAMEF["fiea"]*  DAB[    "ef"]; 
        q(1)[    "ia"] +=     WMNEF["miea"]*  DAI[    "em"]; 
        q(1)[    "ia"] -= 0.5*WABEF["efga"]*GAIBC[  "gief"]; 
        q(1)[    "ia"] +=     WAMEI["eifm"]*GAIBC[  "fmea"]; 
        q(1)[    "ia"] -=     WAMEI["eman"]*GIJAK[  "inem"]; 
        q(1)[    "ia"] += 0.5*WMNIJ["imno"]*GIJAK[  "noam"]; 
        q(1)[    "ia"] -= 0.5*WAMEF["gief"]*GABCD[  "efga"]; 
        q(1)[    "ia"] +=     WAMEF["fmea"]*GAIBJ[  "eifm"]; 
        q(1)[    "ia"] -=     WMNEJ["inem"]*GAIBJ[  "eman"]; 
        q(1)[    "ia"] += 0.5*WMNEJ["noam"]*GIJKL[  "imno"]; 
                                                             
        q(2)[  "ijab"]  =     WMNEF["ijeb"]*  DAB[    "ea"]; 
        q(2)[  "ijab"] -=     WMNEF["mjab"]*  DIJ[    "im"]; 
        q(2)[  "ijab"] += 0.5*WMNEF["ijef"]*GABCD[  "efab"]; 
        q(2)[  "ijab"] +=     WMNEF["imea"]*GAIBJ[  "ejbm"]; 
        q(2)[  "ijab"] += 0.5*WMNEF["mnab"]*GIJKL[  "ijmn"]; 
        q(2)[  "ijab"] -=     WAMEF["fiae"]*GAIBC[  "ejbf"]; 
        q(2)[  "ijab"] -=     WMNEJ["ijem"]*GAIBC[  "emab"]; 
        q(2)[  "ijab"] -=     WAMEF["emab"]*GIJAK[  "ijem"]; 
        q(2)[  "ijab"] -=     WMNEJ["niam"]*GIJAK[  "mjbn"]; 
        q(2)[  "ijab"] +=   0.5*WABEJ["efbm"]* L3["ijmaef"]; 
        q(2)[  "ijab"] -=   0.5*WAMIJ["ejnm"]* L3["imnabe"]; 

        Z(1)[  "ia"] +=                        q(1)[  "ia"];
        Z(2)["ijab"] +=                        q(2)["ijab"];
    }

    Z.weight(D);
    L += Z;

    if (this->config.get<int>("print_subiterations")>0)
    {
        this->energy() = 0.25*real(scalar(conj(WMNEF)*L(2)));
    }
}

template <typename U>
void LambdaCCSDTQ_1a<U>::microiterate(const Arena& arena)
{
    const auto& H    = this->template get<  TwoElectronOperator<U>>("H");
    const auto& Hbar = this->template get<STTwoElectronOperator<U>>("Hbar");

    const SpinorbitalTensor<U>&   FME =   Hbar.getIA();
    const SpinorbitalTensor<U>&   FAE =   Hbar.getAB();
    const SpinorbitalTensor<U>&   FMI =   Hbar.getIJ();
    const SpinorbitalTensor<U>& WMNEF = Hbar.getIJAB();
    const SpinorbitalTensor<U>& WAMEF = Hbar.getAIBC();
    const SpinorbitalTensor<U>& WABEJ = Hbar.getABCI();
    const SpinorbitalTensor<U>& WABEF = Hbar.getABCD();
    const SpinorbitalTensor<U>& WMNIJ = Hbar.getIJKL();
    const SpinorbitalTensor<U>& WMNEJ = Hbar.getIJAK();
    const SpinorbitalTensor<U>& WAMIJ = Hbar.getAIJK();
    const SpinorbitalTensor<U>& WAMEI = Hbar.getAIBJ();

    const SpinorbitalTensor<U>&   fME =   H.getIA();
    const SpinorbitalTensor<U>& VMNEF = H.getIJAB();
    const SpinorbitalTensor<U>& VABEJ = H.getABCI();
    const SpinorbitalTensor<U>& VAMIJ = H.getAIJK();

    auto& T = this->template get   <ExcitationOperator  <U,3>>("T");
    auto& L = this->template get   <DeexcitationOperator<U,2>>("L");
    auto& D = this->template gettmp<Denominator         <U  >>("D");
    auto& Z = this->template gettmp<DeexcitationOperator<U,2>>("Z");
    auto& Q = this->template gettmp<DeexcitationOperator<U,3>>("Q");
    auto& q = this->template gettmp<DeexcitationOperator<U,2>>("q");

    auto&     DIJ = this->template gettmp<SpinorbitalTensor<U>>(    "DIJ");
    auto&     DAB = this->template gettmp<SpinorbitalTensor<U>>(    "DAB");

    /***************************************************************************
     *
     * Intermediates for Lambda-CCSD
     */
    DIJ["ij"]  =  0.5*T(2)["efjm"]*L(2)["imef"];
    DAB["ab"]  = -0.5*T(2)["aemn"]*L(2)["mnbe"];
    /*
     **************************************************************************/

    /***************************************************************************
     *
     * Lambda-CCSD iteration
     */
    Z(1)[  "ia"]  =       FME[  "ia"];
    Z(1)[  "ia"] +=       FAE[  "ea"]*L(1)[  "ie"];
    Z(1)[  "ia"] -=       FMI[  "im"]*L(1)[  "ma"];
    Z(1)[  "ia"] -=     WAMEI["eiam"]*L(1)[  "me"];
    Z(1)[  "ia"] += 0.5*WABEJ["efam"]*L(2)["imef"];
    Z(1)[  "ia"] -= 0.5*WAMIJ["eimn"]*L(2)["mnea"];
    Z(1)[  "ia"] -=     WMNEJ["inam"]* DIJ[  "mn"];
    Z(1)[  "ia"] -=     WAMEF["fiea"]* DAB[  "ef"];

    Z(2)["ijab"]  =     WMNEF["ijab"];
    Z(2)["ijab"] +=       FME[  "ia"]*L(1)[  "jb"];
    Z(2)["ijab"] +=     WAMEF["ejab"]*L(1)[  "ie"];
    Z(2)["ijab"] -=     WMNEJ["ijam"]*L(1)[  "mb"];
    Z(2)["ijab"] +=       FAE[  "ea"]*L(2)["ijeb"];
    Z(2)["ijab"] -=       FMI[  "im"]*L(2)["mjab"];
    Z(2)["ijab"] += 0.5*WABEF["efab"]*L(2)["ijef"];
    Z(2)["ijab"] += 0.5*WMNIJ["ijmn"]*L(2)["mnab"];
    Z(2)["ijab"] +=     WAMEI["eiam"]*L(2)["mjbe"];
    Z(2)["ijab"] -=     WMNEF["mjab"]* DIJ[  "im"];
    Z(2)["ijab"] +=     WMNEF["ijeb"]* DAB[  "ea"];
    /*
     **************************************************************************/
    Z(1)[  "ia"] +=                   q(1)[  "ia"];
    Z(2)["ijab"] +=                   Q(2)["ijab"];
    Z(2)["ijab"] +=                   q(2)["ijab"];
    /***************************************************************************
     */
    Z.weight(D);
    L += Z;
    if (this->config.get<int>("print_subiterations")>0)
    {
        this->energy() = 0.25*real(scalar(conj(WMNEF)*L(2)));
    }
}

}
}

static const char* spec = R"!(

convergence?
    double 1e-9,
max_iterations?
    int 50,
sub_iterations?
    int 2,
micro_iterations?
    int 2,
print_subiterations?
    int 0,
conv_type?
    enum { MAXE, RMSE, MAE },
diis?
{
    damping?
        double 0.0,
    start?
        int 1,
    order?
        int 5,
    jacobi?
        bool false
}

)!";

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::LambdaCCSDTQ_1a);
REGISTER_TASK(aquarius::cc::LambdaCCSDTQ_1a<double>,"lambdaccsdtq-1a",spec);
