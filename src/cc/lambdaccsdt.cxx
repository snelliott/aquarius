#include "lambdaccsdt.hpp"

using namespace aquarius::op;
using namespace aquarius::input;
using namespace aquarius::tensor;
using namespace aquarius::task;
using namespace aquarius::time;

namespace aquarius
{
namespace cc
{

template <typename U>
LambdaCCSDT<U>::LambdaCCSDT(const string& name, Config& config)
: Subiterative<U>(name, config), diis(config.get("diis"))
{
    vector<Requirement> reqs;
    reqs.push_back(Requirement("ccsdt.Hbar", "Hbar"));
    reqs.push_back(Requirement("ccsdt.T", "T"));
    this->addProduct(Product("double", "energy", reqs));
    this->addProduct(Product("double", "convergence", reqs));
    this->addProduct(Product("ccsdt.L", "L", reqs));
    this->addProduct(Product("ccsdt.L3","L3", reqs));
}

template <typename U>
bool LambdaCCSDT<U>::run(TaskDAG& dag, const Arena& arena)
{
    const auto& H = this->template get<STTwoElectronOperator<U>>("Hbar");

    const Space& occ = H.occ;
    const Space& vrt = H.vrt;

    this->put   (    "L", new DeexcitationOperator<U,3>("L", arena, occ, vrt));
    this->puttmp(    "Z", new DeexcitationOperator<U,3>("Z", arena, occ, vrt));
    this->puttmp(    "Q", new DeexcitationOperator<U,2>("Q", arena, occ, vrt));
    this->puttmp(    "D", new Denominator         <U  >(H));
    this->puttmp(  "DIJ", new SpinorbitalTensor   <U  >(   "D(ij)",   H.getIJ()));
    this->puttmp(  "DAB", new SpinorbitalTensor   <U  >(   "D(ab)",   H.getAB()));
    this->puttmp(  "DAI", new SpinorbitalTensor   <U  >(   "D(ai)",   H.getAI()));
    this->puttmp("GABCD", new SpinorbitalTensor   <U  >("G(ab,cd)", H.getABCD()));
    this->puttmp("GAIBJ", new SpinorbitalTensor   <U  >("G(ai,bj)", H.getAIBJ()));
    this->puttmp("GIJKL", new SpinorbitalTensor   <U  >("G(ij,kl)", H.getIJKL()));
    this->puttmp("GAIBC", new SpinorbitalTensor   <U  >("G(ai,bc)", H.getAIBC()));
    this->puttmp("GIJAK", new SpinorbitalTensor   <U  >("G(ij,ak)", H.getIJAK()));
    auto& QEFIAMN = this->puttmp("QEFIAMN", new SpinorbitalTensor<U>("Q)efi,amn)", arena,
                                               H.getABIJ().getGroup(),
                                               {vrt, occ}, {2,1},
                                               {1,2}));

    auto& T  = this->template get   <ExcitationOperator  <U,3>>( "T");
    auto& L  = this->template get   <DeexcitationOperator<U,3>>( "L");
    auto& Z  = this->template gettmp<DeexcitationOperator<U,3>>( "Z");
    auto& D  = this->template gettmp<Denominator         <U  >>( "D");

    Z(0) = 0;
    L(0) = 1;
    L(1)[    "ia"] = T(1)[    "ai"];
    L(2)[  "ijab"] = T(2)[  "abij"];
    L(3)["ijkabc"] = T(3)["abcijk"]; 

    const SpinorbitalTensor<U>& WMNEF = H.getIJAB();
    QEFIAMN["efiamn"]  =  0.25*WMNEF[  "ijab"]* T(3)["efbjmn"];

    Subiterative<U>::run(dag, arena);

    this->put("energy", new U(this->energy()));
    this->put("convergence", new U(this->conv()));

    return true;
}

template <typename U>
void LambdaCCSDT<U>::iterate(const Arena& arena)
{
    const auto& H = this->template get<STTwoElectronOperator<U>>("Hbar");
    const SpinorbitalTensor<U>&   FME =   H.getIA();
    const SpinorbitalTensor<U>&   FAE =   H.getAB();
    const SpinorbitalTensor<U>&   FMI =   H.getIJ();
    const SpinorbitalTensor<U>& WMNEF = H.getIJAB();
    const SpinorbitalTensor<U>& WAMEF = H.getAIBC();
    const SpinorbitalTensor<U>& WABEJ = H.getABCI();
    const SpinorbitalTensor<U>& WABEF = H.getABCD();
    const SpinorbitalTensor<U>& WMNIJ = H.getIJKL();
    const SpinorbitalTensor<U>& WMNEJ = H.getIJAK();
    const SpinorbitalTensor<U>& WAMIJ = H.getAIJK();
    const SpinorbitalTensor<U>& WAMEI = H.getAIBJ();

    auto& T  = this->template get   <ExcitationOperator  <U,3>>( "T");
    auto& L  = this->template get   <DeexcitationOperator<U,3>>( "L");
    auto& Z  = this->template gettmp<DeexcitationOperator<U,3>>( "Z");
    auto& Q  = this->template gettmp<DeexcitationOperator<U,2>>( "Q");
    auto& D  = this->template gettmp<Denominator         <U  >>( "D");

    auto&   DIJ = this->template gettmp<SpinorbitalTensor<U>>(  "DIJ");
    auto&   DAB = this->template gettmp<SpinorbitalTensor<U>>(  "DAB");
    auto&   DAI = this->template gettmp<SpinorbitalTensor<U>>(  "DAI");
    auto& GABCD = this->template gettmp<SpinorbitalTensor<U>>("GABCD");
    auto& GAIBJ = this->template gettmp<SpinorbitalTensor<U>>("GAIBJ");
    auto& GIJKL = this->template gettmp<SpinorbitalTensor<U>>("GIJKL");
    auto& GAIBC = this->template gettmp<SpinorbitalTensor<U>>("GAIBC");
    auto& GIJAK = this->template gettmp<SpinorbitalTensor<U>>("GIJAK");
    auto& QEFIAMN = this->template gettmp<SpinorbitalTensor<U>>(  "QEFIAMN");

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
      DIJ[  "ij"]  =  (1.0/12.0)*T(3)["efgjmn"]* L(3)["imnefg"];
      DAB[  "ab"]  = -(1.0/12.0)*T(3)["aefmno"]* L(3)["mnobef"];
                                                           
    GABCD["abcd"]  =   (1.0/6.0)*T(3)["abemno"]* L(3)["mnocde"];
    GAIBJ["aibj"]  =       -0.25*T(3)["aefjmn"]* L(3)["imnbef"];
    GIJKL["ijkl"]  =   (1.0/6.0)*T(3)["efgklm"]* L(3)["ijmefg"];

    GIJAK["ijak"]  =         0.5*T(2)[  "efkm"]* L(3)["ijmaef"];
    GAIBC["aibc"]  =        -0.5*T(2)[  "aemn"]* L(3)["minbce"];

      DAI[  "ai"]  =        0.25*T(3)["aefimn"]* L(2)[  "mnef"];
      DAI[  "ai"] -=         0.5*T(2)[  "eamn"]*GIJAK[  "mnei"];
    /*
     **************************************************************************/

    /***************************************************************************
     *
     * Lambda-CCSDT iteration
     */
     if (this->config.get<int>("sub_iterations") == 0 )
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
        Z(2)[  "ijab"] += 0.5*WABEJ["efbm"]* L(3)["ijmaef"];
        Z(2)[  "ijab"] -= 0.5*WAMIJ["ejnm"]* L(3)["imnabe"];
    }
    Z(3)["ijkabc"]  =     WMNEF["ijab"]* L(1)[    "kc"];
    Z(3)["ijkabc"] +=       FME[  "ia"]* L(2)[  "jkbc"];
    Z(3)["ijkabc"] +=     WAMEF["ekbc"]* L(2)[  "ijae"];
    Z(3)["ijkabc"] -=     WMNEJ["ijam"]* L(2)[  "mkbc"];
    Z(3)["ijkabc"] +=     WMNEF["ijae"]*GAIBC[  "ekbc"];
    Z(3)["ijkabc"] -=     WMNEF["mkbc"]*GIJAK[  "ijam"];
    Z(3)["ijkabc"] +=       FAE[  "ea"]* L(3)["ijkebc"];
    Z(3)["ijkabc"] -=       FMI[  "im"]* L(3)["mjkabc"];
    Z(3)["ijkabc"] += 0.5*WABEF["efab"]* L(3)["ijkefc"];
    Z(3)["ijkabc"] += 0.5*WMNIJ["ijmn"]* L(3)["mnkabc"];
    Z(3)["ijkabc"] +=     WAMEI["eiam"]* L(3)["mjkbec"];
    /*
     **************************************************************************/

    Z(3).weight({&D.getDA(),&D.getDI()},{&D.getDA(),&D.getDI()});
    L(3) += Z(3);

    DAI[  "ai"] =       - 0.5*T(2)[  "eamn"]*GIJAK[  "mnei"];
   
    if (this->config.get<int>("sub_iterations") != 0 )
    {
        Q(1)[    "ia"]  =       FME[  "ie"]*  DAB[    "ea"];
        Q(1)[    "ia"] -=       FME[  "ma"]*  DIJ[    "im"];
        Q(1)[    "ia"] -=     WMNEJ["inam"]*  DIJ[    "mn"];
        Q(1)[    "ia"] -=     WAMEF["fiea"]*  DAB[    "ef"];
        Q(1)[    "ia"] +=     WMNEF["miea"]*  DAI[    "em"];
        Q(1)[    "ia"] -= 0.5*WABEF["efga"]*GAIBC[  "gief"];
        Q(1)[    "ia"] +=     WAMEI["eifm"]*GAIBC[  "fmea"];
        Q(1)[    "ia"] -=     WAMEI["eman"]*GIJAK[  "inem"];
        Q(1)[    "ia"] += 0.5*WMNIJ["imno"]*GIJAK[  "noam"];
        Q(1)[    "ia"] -= 0.5*WAMEF["gief"]*GABCD[  "efga"];
        Q(1)[    "ia"] +=     WAMEF["fmea"]*GAIBJ[  "eifm"];
        Q(1)[    "ia"] -=     WMNEJ["inem"]*GAIBJ[  "eman"];
        Q(1)[    "ia"] += 0.5*WMNEJ["noam"]*GIJKL[  "imno"];

        Q(2)[  "ijab"]  =     WMNEF["ijeb"]*  DAB[    "ea"];
        Q(2)[  "ijab"] -=     WMNEF["mjab"]*  DIJ[    "im"];
        Q(2)[  "ijab"] += 0.5*WMNEF["ijef"]*GABCD[  "efab"];
        Q(2)[  "ijab"] +=     WMNEF["imea"]*GAIBJ[  "ejbm"];
        Q(2)[  "ijab"] += 0.5*WMNEF["mnab"]*GIJKL[  "ijmn"];
        Q(2)[  "ijab"] -=     WAMEF["fiae"]*GAIBC[  "ejbf"];
        Q(2)[  "ijab"] -=     WMNEJ["ijem"]*GAIBC[  "emab"];
        Q(2)[  "ijab"] -=     WAMEF["emab"]*GIJAK[  "ijem"];
        Q(2)[  "ijab"] -=     WMNEJ["niam"]*GIJAK[  "mjbn"];
        Q(2)[  "ijab"] += 0.5*WABEJ["efbm"]* L(3)["ijmaef"];
        Q(2)[  "ijab"] -= 0.5*WAMIJ["ejnm"]* L(3)["imnabe"];

        Z(1)[    "ia"] +=                    Q(1)[    "ia"];
        Z(2)[  "ijab"] +=                    Q(2)[  "ijab"];
      
        Z(1)[    "ia"] +=     QEFIAMN["efiamn"]* L(2)[  "mnef"];
    }

    Z(2).weight({&D.getDA(),&D.getDI()},{&D.getDA(),&D.getDI()});
    L(2) += Z(2);
    Z(1).weight({&D.getDA(),&D.getDI()},{&D.getDA(),&D.getDI()});
    L(1) += Z(1);

    this->energy() = 0.25*real(scalar(conj(WMNEF)*L(2)));
    this->conv() = Z.norm(00);

    auto& Ldiis  = this->template get   <DeexcitationOperator<U,2>>(  "L");
    auto& Zdiis  = this->template gettmp<DeexcitationOperator<U,2>>(  "Z");
    diis.extrapolate(Ldiis, Zdiis);
}


template <typename U>
void LambdaCCSDT<U>::subiterate(const Arena& arena)
{
    const auto& H = this->template get<STTwoElectronOperator<U>>("Hbar");

    const SpinorbitalTensor<U>&   FME =   H.getIA();
    const SpinorbitalTensor<U>&   FAE =   H.getAB();
    const SpinorbitalTensor<U>&   FMI =   H.getIJ();
    const SpinorbitalTensor<U>& WMNEF = H.getIJAB();
    const SpinorbitalTensor<U>& WAMEF = H.getAIBC();
    const SpinorbitalTensor<U>& WABEJ = H.getABCI();
    const SpinorbitalTensor<U>& WABEF = H.getABCD();
    const SpinorbitalTensor<U>& WMNIJ = H.getIJKL();
    const SpinorbitalTensor<U>& WMNEJ = H.getIJAK();
    const SpinorbitalTensor<U>& WAMIJ = H.getAIJK();
    const SpinorbitalTensor<U>& WAMEI = H.getAIBJ();

    auto& T  = this->template get   <ExcitationOperator  <U,2>>( "T");
    auto& L  = this->template get   <DeexcitationOperator<U,2>>( "L");
    auto& Q  = this->template gettmp<DeexcitationOperator<U,2>>( "Q");
    auto& Z  = this->template gettmp<DeexcitationOperator<U,2>>( "Z");
    auto& D  = this->template gettmp<Denominator         <U  >>( "D");

    auto&   DIJ = this->template gettmp<SpinorbitalTensor<U>>(  "DIJ");
    auto&   DAB = this->template gettmp<SpinorbitalTensor<U>>(  "DAB");
    auto&   DAI = this->template gettmp<SpinorbitalTensor<U>>(  "DAI");
    auto& QEFIAMN = this->template gettmp<SpinorbitalTensor<U>>("QEFIAMN");

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
   
     
    /*L(3) -> T(2) and L(3) -> T(1)  */
    Z(1)[    "ia"] +=                                Q(1)[    "ia"];
    Z(2)[  "ijab"] +=                                Q(2)[  "ijab"];
    Z(1)[    "ia"] +=         QEFIAMN[    "efiamn"]* L(2)[  "mnef"];
  
     /*
      **************************************************************************/
    Z(1).weight({&D.getDA(),&D.getDI()},{&D.getDa(),&D.getDi()});
    Z(2).weight({&D.getDa(),&D.getDi()},{&D.getDa(),&D.getDi()});
    L(1) += Z(1);
    L(2) += Z(2);

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
    int 3,
micro_iterations?
    int 0,
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

INSTANTIATE_SPECIALIZATIONS(aquarius::cc::LambdaCCSDT);
REGISTER_TASK(aquarius::cc::LambdaCCSDT<double>,"lambdaccsdt",spec);
