// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Ngoc Cuong Nguyen (MIT) and Dionysios Sema (MIT)
------------------------------------------------------------------------- */

// LAMMPS header files

#include "comm.h"
#include "error.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "safe_pointers.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>

// header file. Moved down here to avoid polluting other headers with its defines
#include "eapod.h"

using namespace LAMMPS_NS;
using MathConst::MY_PI;
using MathSpecial::powint;
static const double e_v = std::exp(1.0);
static constexpr int MAXLINE=1024;

// constructor
EAPOD::EAPOD(LAMMPS *_lmp, const std::string &pod_file, const std::string &coeff_file) :
    Pointers(_lmp), elemindex(nullptr), Phi(nullptr), coeff(nullptr), tmpmem(nullptr),
    rin(nullptr), rcut(nullptr), Proj(nullptr), Centroids(nullptr),
    invLeftClusterRcut2(nullptr), invRightClusterRcut2(nullptr),
    leftClusterEdges(nullptr), rightClusterEdges(nullptr),
    invPcaSpan(nullptr), clusterOccupancy(nullptr),
    bd(nullptr), bdd(nullptr), pd(nullptr), pdd(nullptr),
    pn3(nullptr), pq3(nullptr), pc3(nullptr), pa4(nullptr),
    pb4(nullptr), pc4(nullptr), tmpint(nullptr), ind33l(nullptr), ind33r(nullptr),
    ind34l(nullptr), ind34r(nullptr), ind44l(nullptr), ind44r(nullptr),
    p3_active(nullptr), dabf3_active(nullptr),
    p4_active(nullptr), dabf4_active(nullptr), deg4_full(nullptr),
    rbf_spline_coeffs(nullptr), spline_r0(nullptr), spline_invdr(nullptr)
{
  nelements = 1;
  nClusters = 1;
  eapod = 0;
  localeapod = 0;
  nActiveClusters = 0;
  clusterSearchBox = 0.5*nActiveClusters + 1;
  hat_p = 2;
  hat_q = 2;
  hat_p1 = hat_p - 1;
  hat_q1 = hat_q - 1;
  h_pq = hat_p*hat_q;
  nComponents = 1;
  besseldegree = 4;
  inversedegree = 8;
  nbesselpars = 3;
  nbesselrbf = nbesselpars*besseldegree;
  ns = nbesselrbf + inversedegree;
  Njmax = 100;
  onebody = 1;
  nrbf2 = 8;
  nrbf3 = 6;
  nrbf4 = 0;
  nabf3 = 5;
  nabf4 = 0;
  nrbf33 = 0;
  nrbf34 = 0;
  nrbf44 = 0;
  P3 = 4;
  P4 = 0;
  P33 = 0;
  P34 = 0;
  P44 = 0;
  L3min = 0; L3max = -1;   // -1 => use full [0,P3]
  L4min = 0; L4max = -1;   // -1 => use full [0,P4]
  pbc[0] = 1;
  pbc[1] = 1;
  pbc[2] = 1;

  use_spline   = false;
  nspline_grid = 1000;
  nspline_bins = nspline_grid-1;
  
  uncertaintyflag = false;

  // read pod input file to podstruct
  read_pod_file(pod_file);

  if (!coeff_file.empty()) {
    read_model_coeff_file(coeff_file);
  }
}

// destructor
EAPOD::~EAPOD()
{
  memory->destroy(elemindex);
  memory->destroy(Phi);
  memory->destroy(rin);
  memory->destroy(rcut);
  memory->destroy(rcutsq);
  memory->destroy(invrdiff);
  memory->destroy(Proj);
  memory->destroy(Centroids);
  memory->destroy(invLeftClusterRcut2);
  memory->destroy(invRightClusterRcut2);
  memory->destroy(leftClusterEdges);
  memory->destroy(rightClusterEdges);
  memory->destroy(invPcaSpan);
  memory->destroy(clusterOccupancy);
  memory->destroy(bd);
  memory->destroy(bdd);
  memory->destroy(pd);
  memory->destroy(pdd);
  memory->destroy(coeff);
  memory->destroy(tmpmem);
  memory->destroy(tmpint);
  memory->destroy(pn3);
  memory->destroy(pq3);
  memory->destroy(pc3);
  memory->destroy(pa4);
  memory->destroy(pb4);
  memory->destroy(pc4);
  memory->destroy(ind33l);
  memory->destroy(ind34l);
  memory->destroy(ind44l);
  memory->destroy(ind33r);
  memory->destroy(ind34r);
  memory->destroy(ind44r);
  memory->destroy(besselparams);
  memory->destroy(p3_active);
  memory->destroy(p4_active);
  memory->destroy(dabf3_active);
  memory->destroy(dabf4_active);
  memory->destroy(deg4_full);
  memory->destroy(rbf_spline_coeffs);
  memory->destroy(spline_r0);
  memory->destroy(spline_invdr);
}

void EAPOD::read_pod_file(const std::string &pod_file)
{
  std::string podfilename = pod_file;
  SafeFilePtr fppod;
  if (comm->me == 0) {

    fppod = utils::open_potential(podfilename,lmp,nullptr);
    if (fppod == nullptr)
      error->one(FLERR,"Cannot open POD coefficient file {}: ",
                                   podfilename, utils::getsyserror());
  }

  // loop through lines of POD file and parse keywords

  char line[MAXLINE],*ptr;
  int eof = 0;

  while (true) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fppod);
      if (ptr == nullptr) {
        eof = 1;
      }
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

    // words = ptrs to all words in line
    // strip single and double quotes from words

    std::vector<std::string> words;
    try {
      words = Tokenizer(utils::trim_comment(line),"\"' \t\n\r\f").as_vector();
    } catch (TokenizerException &) {
      // ignore
    }

    if (words.empty()) continue;

    const auto &keywd = words[0];

    if (keywd == "species") {
      nelements = words.size()-1;
      for (int ielem = 1; ielem <= nelements; ielem++) {
        species.push_back(words[ielem]);
      }
    }

    if (keywd == "pbc") {
      if (words.size() != 4)
        error->one(FLERR,"Improper POD file.", utils::getsyserror());
      pbc[0] = utils::inumeric(FLERR,words[1],false,lmp);
      pbc[1] = utils::inumeric(FLERR,words[2],false,lmp);
      pbc[2] = utils::inumeric(FLERR,words[3],false,lmp);
    }

    int Ne = nelements;

    if (keywd == "rin") {
      int wsize = words.size()-1;
      if ( (wsize != Ne*Ne) && (wsize != 1) )
        error->one(FLERR,"Improper POD file. Provide outer cut-off radius for each element pair", utils::getsyserror());
      
      memory->create(rin, Ne, Ne, "rin");
      if (wsize != Ne*Ne) {
        double r = utils::numeric(FLERR,words[1],false,lmp);
        for (int i = 0; i < Ne; i++)
          for (int j = 0; j < Ne; j++)
            rin[i][j] = r;
      }
      else {
        for (int i = 0; i < Ne; i++)
          for (int j = 0; j < Ne; j++) {
            int ij = j + i*Ne;
            rin[i][j] = utils::numeric(FLERR,words[ij+1],false,lmp);
          }
      }
    }

    if (keywd == "rcut") {
      int wsize = words.size()-1;
      if ( (wsize != Ne*Ne) && (wsize != 1) )
        error->one(FLERR,"Improper POD file. Provide outer cut-off radius for each element pair", utils::getsyserror());
      
      memory->create(rcut, Ne, Ne, "rcut");
      if (wsize != Ne*Ne) {
        double r = utils::numeric(FLERR,words[1],false,lmp);
        for (int i = 0; i < Ne; i++)
          for (int j = 0; j < Ne; j++)
            rcut[i][j] = r;
      }
      else {
        for (int i = 0; i < Ne; i++)
          for (int j = 0; j < Ne; j++) {
            int ij = j + i*Ne;
            rcut[i][j] = utils::numeric(FLERR,words[ij+1],false,lmp);
          }
      }
    }

    // settings for the base POD potential and descriptors
    if (keywd == "bessel_parameters") {
      int wsize = words.size()-1;
      if ( wsize < nbesselpars )
        error->one(FLERR,"Improper POD file. Less number of bessel parameters available than requested", utils::getsyserror());
      
      memory->create(besselparams, nbesselpars, "besselparams");
      for (int i = 0; i < nbesselpars; i++)
        besselparams[i] = utils::numeric(FLERR,words[i+1],false,lmp);
    }

    if ((keywd != "#") && (keywd != "species") && (keywd != "pbc") &&
        (keywd != "rin") && (keywd != "rcut") && (keywd != "bessel_parameters")) {

      if (words.size() != 2)
        error->one(FLERR,"Improper POD file.", utils::getsyserror());
      
      if (keywd == "number_of_environment_clusters")
        nClusters = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "number_of_active_clusters")
        nActiveClusters = utils::numeric(FLERR,words[1],false,lmp);
      if (keywd == "enable_active_learning")
        uncertaintyflag = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "order_of_cluster_activation_basis")
        hat_p = utils::numeric(FLERR,words[1],false,lmp);
      if (keywd == "order_of_cluster_activation_hat")
        hat_q = utils::numeric(FLERR,words[1],false,lmp);
      if (keywd == "number_of_principal_components")
        nComponents = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "number_of_bessel_parameters")
        nbesselpars = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "bessel_polynomial_degree")
        besseldegree = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "inverse_polynomial_degree")
        inversedegree = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "threebody_angular_degree_min")
        L3min = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "threebody_angular_degree_max")
        L3max = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "fourbody_angular_degree_min")
        L4min = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "fourbody_angular_degree_max")
        L4max = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "enable_radial_spline")
        use_spline = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "radial_spline_grid")
        nspline_grid = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "onebody")
        onebody = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "twobody_number_radial_basis_functions")
        nrbf2 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "threebody_number_radial_basis_functions")
        nrbf3 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "threebody_angular_degree")
        P3 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "fourbody_number_radial_basis_functions")
        nrbf4 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "fourbody_angular_degree")
        P4 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "fivebody_number_radial_basis_functions")
        nrbf33 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "fivebody_angular_degree")
        P33 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "sixbody_number_radial_basis_functions")
        nrbf34 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "sixbody_angular_degree")
        P34 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "sevenbody_number_radial_basis_functions")
        nrbf44 = utils::inumeric(FLERR,words[1],false,lmp);
      if (keywd == "sevenbody_angular_degree")
        P44 = utils::inumeric(FLERR,words[1],false,lmp);
    }
  }
  if (nrbf2 < nrbf3) error->all(FLERR,"number of three-body radial basis functions must be equal or less than number of two-body radial basis functions");
  if (nrbf3 < nrbf4) error->all(FLERR,"number of four-body radial basis functions must be equal or less than number of three-body radial basis functions");
  if (nrbf4 < nrbf33) error->all(FLERR,"number of five-body radial basis functions must be equal or less than number of four-body radial basis functions");
  if (nrbf4 < nrbf34) error->all(FLERR,"number of six-body radial basis functions must be equal or less than number of four-body radial basis functions");
  if (nrbf4 < nrbf44) error->all(FLERR,"number of seven-body radial basis functions must be equal or less than number of four-body radial basis functions");

  nbesselrbf = besseldegree*nbesselpars;
  ns = nbesselrbf + inversedegree;
  if (ns < nrbf2) {
    inversedegree = nrbf2 - nbesselrbf;
    ns = nrbf2;
  }

  for (int i = 0; i < nbesselpars; i++)
    if (fabs(besselparams[i]) <= 1.0e-6) besselparams[i] = 1e-3;

  if (P3 < P4) error->all(FLERR,"four-body angular degree must be equal or less than three-body angular degree");
  if (P4 < P33) error->all(FLERR,"five-body angular degree must be equal or less than four-body angular degree");
  if (P4 < P34) error->all(FLERR,"six-body angular degree must be equal or less than four-body angular degree");
  if (P4 < P44) error->all(FLERR,"seven-body angular degree must be equal or less than four-body angular degree");

  if (P3 > 12) error->all(FLERR,"three-body angular degree must be equal or less than 12");
  if (P4 > 6) error->all(FLERR,"four-body angular degree must be equal or less than 6");

  if (nActiveClusters == 1.0) nActiveClusters += 1e-3;
  if (nClusters < 1) nClusters = 1;
  if (nClusters == 1) nActiveClusters = 0;
  if ((nActiveClusters < 1.0) && (static_cast<int>(nActiveClusters) != 0))
    error->all(FLERR,"average number of active clusters must be greater or equal to 1.0");
  if (nClusters > 1) eapod = 1;
  if (nActiveClusters >= 1.0) localeapod = 1;
  if ( (localeapod) && (nComponents != 1))
    error->all(FLERR,"local EA-POD with multiple PCA components is not supported yet. Please use one principal component.");
  clusterSearchBox = 0.5 * nActiveClusters + 1;
  nMaxActiveClusters = clusterSearchBox + 1;

  if (uncertaintyflag && !eapod) uncertaintyflag = false;

  h_pq = hat_p * hat_q;
  hat_p1 = hat_p - 1;
  hat_q1 = hat_q - 1;

  int Ne = nelements;
  memory->create(elemindex, Ne, Ne, "elemindex");
  int k = 0;
  for (int i1 = 0; i1<Ne; i1++)
    for (int i2 = i1; i2<Ne; i2++) {
      elemindex[i1][i2] = k;
      elemindex[i2][i1] = k;
      k += 1;
    }
  
  // Compute the maximum and minimum distances between two atoms for each element pair type
  memory->create(rcutsq, Ne, Ne, "rcutsq");
  memory->create(invrdiff, Ne, Ne, "invrdiff");
  rcutmax = rcut[0][0];
  for (int i = 0; i < nelements; i++) {
    for (int j = 0; j < nelements; j++) {
      double rcut_ij = rcut[i][j];
      double rin_ij = rin[i][j];
      invrdiff[i][j] = 1.0 / (rcut_ij - rin_ij);
      rcutsq[i][j] = rcut_ij * rcut_ij;
      if (rcut_ij > rcutmax) rcutmax = rcut_ij;
    }
  }

  init_bessel_const();

  init2body();
  init3body(P3);
  init4body(P4);

  if (use_spline) init_spline_radialbasis();

  if (L3max < 0) L3max = P3;
  if (L4max < 0) L4max = P4;

  if (L3min < 0 || L3min > L3max || L3max > P3)
    error->all(FLERR,"Invalid 3-body angular degree range");

  if (L4min < 0 || L4min > L4max || L4max > P4)
    error->all(FLERR,"Invalid 4-body angular degree range");

  init_active_angular_ranges();

  int nebf3 = Ne*(Ne+1)/2;
  int nebf4 = Ne*(Ne+1)*(Ne+2)/6;

  if (onebody==0)
    nd1 = 0;
  else {
    nd1 = Ne;
    onebody = 1;
  }

  nl1 = onebody;
  nl2 = nrbf2*Ne;

  nl3 = nabf3_active*nrbf3*nebf3;
  nl4 = nabf4_active*nrbf4*nebf4;

  nl33 = 0;
  nl34 = 0;
  nl44 = 0;
  if (nrbf33>0) {
    nl33 = crossindices(dabf3_active, nabf3_active, nrbf3, nebf3, dabf3_active, nabf3_active, nrbf3, nebf3, P33, nrbf33);
    memory->create(ind33l, nl33, "ind33l");
    memory->create(ind33r, nl33, "ind33r");
    crossindices(ind33l, ind33r, dabf3_active, nabf3_active, nrbf3, nebf3, dabf3_active, nabf3_active, nrbf3, nebf3, P33, nrbf33);
  }
  if (nrbf34>0) {
    nl34 = crossindices(dabf3_active, nabf3_active, nrbf3, nebf3, dabf4_active, nabf4_active, nrbf4, nebf4, P34, nrbf34);
    memory->create(ind34l, nl34, "ind34l");
    memory->create(ind34r, nl34, "ind34r");
    crossindices(ind34l, ind34r, dabf3_active, nabf3_active, nrbf3, nebf3, dabf4_active, nabf4_active, nrbf4, nebf4, P34, nrbf34);
  }
  if (nrbf44>0) {
    nl44 = crossindices(dabf4_active, nabf4_active, nrbf4, nebf4, dabf4_active, nabf4_active, nrbf4, nebf4, P44, nrbf44);
    memory->create(ind44l, nl44, "ind44l");
    memory->create(ind44r, nl44, "ind44r");
    crossindices(ind44l, ind44r, dabf4_active, nabf4_active, nrbf4, nebf4, dabf4_active, nabf4_active, nrbf4, nebf4, P44, nrbf44);
  }

  nd2 = nl2*Ne;
  nd3 = nl3*Ne;
  nd4 = nl4*Ne;
  nd33 = nl33*Ne;
  nd34 = nl34*Ne;
  nd44 = nl44*Ne;

  Mdesc = nl2 + nl3 + nl4 + nl33 + nl34 + nl44;
  nl = nl1 + nl2 + nl3 + nl4 + nl33 + nl34 + nl44;
  nd = nd1 + nd2 + nd3 + nd4 + nd33 + nd34 + nd44;
  nCoeffPerElement = nl1 + Mdesc*nClusters;
  nCoeffAll = nCoeffPerElement*nelements;

  allocate_temp_memory(Njmax);

  if (comm->me == 0) {
    utils::logmesg(lmp, "**************** Begin of POD Potentials ****************\n");

    utils::logmesg(lmp, "periodic boundary conditions: {} {} {}\n", pbc[0], pbc[1], pbc[2]);
    utils::logmesg(lmp, "species:");
    for (int i=0; i<nelements; i++)
      utils::logmesg(lmp, " {}", species[i]);
    utils::logmesg(lmp, "\n");
    utils::logmesg(lmp, "inner cut-off radius for element pairs:\n");
    for (int i = 0; i < nelements; i++) {
      for (int j = 0; j < nelements; j++) {
        utils::logmesg(lmp, "  {}-{}: {}\n", species[i], species[j], rin[i][j]);
      }
    }
    utils::logmesg(lmp, "outer cut-off radius for element pairs:\n");
    for (int i = 0; i < nelements; i++) {
      for (int j = 0; j < nelements; j++) {
        utils::logmesg(lmp, "  {}-{}: {}\n", species[i], species[j], rcut[i][j]);
      }
    }
    utils::logmesg(lmp, "enable active learning: {}\n", (int)uncertaintyflag);
    utils::logmesg(lmp, "number of environment clusters: {}\n", nClusters);
    utils::logmesg(lmp, "enable local ea-pod: {}\n", (int)localeapod);
    utils::logmesg(lmp, "number of active clusters: {}\n", nActiveClusters);
    utils::logmesg(lmp, "order of cluster activation basis: {}\n", hat_p);
    utils::logmesg(lmp, "order of cluster activation hat function: {}\n", hat_q);
    utils::logmesg(lmp, "number of principal components: {}\n", nComponents);

    utils::logmesg(lmp, "number of bessel parameters: {}\n", nbesselpars);
    utils::logmesg(lmp, "bessel polynomial degree: {}\n", besseldegree);
    utils::logmesg(lmp, "inverse polynomial degree: {}\n", inversedegree);

    utils::logmesg(lmp, "bessel parameters:");
    for (int i = 0; i < nbesselpars; i++)
      utils::logmesg(lmp, " {}", besselparams[i]);
    utils::logmesg(lmp, "\n");

    utils::logmesg(lmp,"3-body angular range: [{}:{}], active channels: {}\n",
                   L3min,L3max,nabf3_active);
    utils::logmesg(lmp,"4-body angular range: [{}:{}], active channels: {}\n",
                   L4min,L4max,nabf4_active);
    
    utils::logmesg(lmp, "enable radial spline: {}\n", (int)use_spline);
    if (use_spline == 1) utils::logmesg(lmp, "radial spline grid resolution: {}\n", nspline_grid);
    utils::logmesg(lmp, "one-body potential: {}\n", onebody);
    utils::logmesg(lmp, "two-body radial basis functions: {}\n", nrbf2);
    utils::logmesg(lmp, "three-body radial basis functions: {}\n", nrbf3);
    utils::logmesg(lmp, "three-body angular degree: {}\n", P3);
    utils::logmesg(lmp, "four-body radial basis functions: {}\n", nrbf4);
    utils::logmesg(lmp, "four-body angular degree: {}\n", P4);
    utils::logmesg(lmp, "five-body radial basis functions: {}\n", nrbf33);
    utils::logmesg(lmp, "five-body angular degree: {}\n", P33);
    utils::logmesg(lmp, "six-body radial basis functions: {}\n", nrbf34);
    utils::logmesg(lmp, "six-body angular degree: {}\n", P34);
    utils::logmesg(lmp, "seven-body radial basis functions: {}\n", nrbf44);
    utils::logmesg(lmp, "seven-body angular degree: {}\n", P44);
    utils::logmesg(lmp, "number of local descriptors per element for one-body potential: {}\n", nl1);
    utils::logmesg(lmp, "number of local descriptors per element for two-body potential: {}\n", nl2);
    utils::logmesg(lmp, "number of local descriptors per element for three-body potential: {}\n", nl3);
    utils::logmesg(lmp, "number of local descriptors per element for four-body potential: {}\n", nl4);
    utils::logmesg(lmp, "number of local descriptors per element for five-body potential: {}\n", nl33);
    utils::logmesg(lmp, "number of local descriptors per element for six-body potential: {}\n", nl34);
    utils::logmesg(lmp, "number of local descriptors per element for seven-body potential: {}\n", nl44);
    utils::logmesg(lmp, "number of local descriptors per element for all potentials: {}\n", nl);
    utils::logmesg(lmp, "number of global descriptors: {}\n", nCoeffAll);
    utils::logmesg(lmp, "**************** End of POD Potentials ****************\n\n");
  }
}

void EAPOD::read_model_coeff_file(const std::string &coeff_file)
{
  std::string coefffilename = coeff_file;
  SafeFilePtr fpcoeff;
  if (comm->me == 0) {

    fpcoeff = utils::open_potential(coefffilename,lmp,nullptr);
    if (fpcoeff == nullptr)
      error->one(FLERR,"Cannot open model coefficient file {}: ", coefffilename, utils::getsyserror());
  }

  // check format for first line of file

  char line[MAXLINE],*ptr;
  int eof = 0;
  int nwords = 0;
  while (nwords == 0) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpcoeff);
      if (ptr == nullptr) {
        eof = 1;
      }
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

    // strip comment, skip line if blank
    nwords = utils::count_words(utils::trim_comment(line));
  }

  if (nwords != 4)
    error->all(FLERR,"Incorrect format in POD coefficient file");

  // strip single and double quotes from words

  int ncoeffall, nprojall, ncentall;
  std::string tmp_str;
  try {
    ValueTokenizer words(utils::trim_comment(line),"\"' \t\n\r\f");
    tmp_str = words.next_string();
    ncoeffall = words.next_int();
    nprojall = words.next_int();
    ncentall = words.next_int();
  } catch (TokenizerException &e) {
    error->all(FLERR,"Incorrect format in POD coefficient file: {}", e.what());
  }

  // loop over single block of coefficients and insert values in coeff

  memory->create(coeff, ncoeffall, "pod:pod_coeff");

  for (int icoeff = 0; icoeff < ncoeffall; icoeff++) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpcoeff);
      if (ptr == nullptr) {
        eof = 1;
      }
    }

    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) error->all(FLERR,"Incorrect format in model coefficient file");
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

    try {
      ValueTokenizer cff(utils::trim_comment(line));
      if (cff.count() != 1) error->all(FLERR,"Incorrect format in model coefficient file");

      coeff[icoeff] = cff.next_double();
    } catch (TokenizerException &e) {
      error->all(FLERR,"Incorrect format in model coefficient file: {}", e.what());
    }
  }

  memory->create(Proj, nprojall, "pod:pca_proj");

  for (int iproj = 0; iproj < nprojall; iproj++) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpcoeff);
      if (ptr == nullptr) {
        eof = 1;
      }
    }

    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) error->all(FLERR,"Incorrect format in model coefficient file");
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

    try {
      ValueTokenizer cff(utils::trim_comment(line));
      if (cff.count() != 1) error->all(FLERR,"Incorrect format in model coefficient file");

      Proj[iproj] = cff.next_double();
    } catch (TokenizerException &e) {
      error->all(FLERR,"Incorrect format in model coefficient file: {}", e.what());
    }
  }

  memory->create(Centroids, ncentall, "pod:pca_cent");

  for (int icent = 0; icent < ncentall; icent++) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fpcoeff);
      if (ptr == nullptr) {
        eof = 1;
      }
    }

    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) error->all(FLERR,"Incorrect format in model coefficient file");
    MPI_Bcast(line,MAXLINE,MPI_CHAR,0,world);

    try {
      ValueTokenizer cff(utils::trim_comment(line));
      if (cff.count() != 1) error->all(FLERR,"Incorrect format in model coefficient file");

      Centroids[icent] = cff.next_double();
    } catch (TokenizerException &e) {
      error->all(FLERR,"Incorrect format in model coefficient file: {}", e.what());
    }
  }


  if (ncoeffall != nCoeffAll)
    error->all(FLERR,"number of coefficients in the coefficient file is not correct");

  if (eapod) {
    if (nprojall != nComponents*Mdesc*nelements)
      error->all(FLERR,"number of coefficients in the projection file is not correct");

    if (ncentall != nComponents*nClusters*nelements)
        error->all(FLERR,"number of coefficients in the projection file is not correct");

    if (uncertaintyflag) {
      memory->create(invPcaSpan, nComponents*nelements, "pod:invPcaSpan");
      calculatePcaSpan();
      read_cluster_occupancy_file(coeff_file);
    }

    if (localeapod) {
      memory->create(invLeftClusterRcut2, ncentall, "pod:invLeftClusterRcut2");
      memory->create(invRightClusterRcut2, ncentall, "pod:invRightClusterRcut2");
      memory->create(leftClusterEdges, ncentall, "pod:leftClusterEdges");
      memory->create(rightClusterEdges, ncentall, "pod:rightClusterEdges");
      calculateClusterEdges(nClusters, nActiveClusters, nComponents, nelements);
    }
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "**************** Begin of Model Coefficients ****************\n");
    utils::logmesg(lmp, "total number of coefficients for POD potential: {}\n", ncoeffall);
    utils::logmesg(lmp, "total number of elements for PCA projection matrix: {}\n", nprojall);
    utils::logmesg(lmp, "total number of elements for PCA centroids: {}\n", ncentall);
    utils::logmesg(lmp, "**************** End of Model Coefficients ****************\n\n");
  }
}

void EAPOD::read_cluster_occupancy_file(const std::string &coeff_file)
{
  std::string occfilename = coeff_file;
  auto pos = occfilename.rfind("_coefficients");
  if (pos != std::string::npos) {
    occfilename.replace(pos, std::string("_coefficients").size(), "_cluster_occupancy");
  } else {
    // fallback: strip a trailing .pod (if any) and append the occupancy suffix
    auto dot = occfilename.rfind(".pod");
    if (dot != std::string::npos) occfilename.erase(dot);
    occfilename += "_cluster_occupancy.pod";
  }

  int ncols = nClusters * nelements;

  SafeFilePtr fpocc;
  int missing = 0;
  if (comm->me == 0) {
    fpocc = utils::open_potential(occfilename, lmp, nullptr);
    if (fpocc == nullptr) missing = 1;
  }
  MPI_Bcast(&missing, 1, MPI_INT, 0, world);
  if (missing) {
    if (comm->me == 0)
      error->warning(FLERR, "Cluster occupancy file {} not found; active-learning density "
                            "metric will be unweighted", occfilename);
    return;
  }

  char line[MAXLINE], *ptr;
  int eof = 0;
  int nwords = 0;

  // header line: "cluster_occupancy: nClusters nelements"
  while (nwords == 0) {
    if (comm->me == 0) {
      ptr = fgets(line, MAXLINE, fpocc);
      if (ptr == nullptr) eof = 1;
    }
    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) break;
    MPI_Bcast(line, MAXLINE, MPI_CHAR, 0, world);
    nwords = utils::count_words(utils::trim_comment(line));
  }
  if (nwords != 3) error->all(FLERR, "Incorrect format in cluster occupancy file");

  int nc_file, nel_file;
  try {
    ValueTokenizer words(utils::trim_comment(line), "\"' \t\n\r\f");
    words.next_string();    // "cluster_occupancy:"
    nc_file = words.next_int();
    nel_file = words.next_int();
  } catch (TokenizerException &e) {
    error->all(FLERR, "Incorrect format in cluster occupancy file: {}", e.what());
    return;
  }
  if ((nc_file != nClusters) || (nel_file != nelements))
    error->all(FLERR, "cluster occupancy file does not match the model (clusters/elements)");

  memory->create(clusterOccupancy, ncols, "pod:clusterOccupancy");
  for (int i = 0; i < ncols; i++) {
    if (comm->me == 0) {
      ptr = fgets(line, MAXLINE, fpocc);
      if (ptr == nullptr) eof = 1;
    }
    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) error->all(FLERR, "Incorrect format in cluster occupancy file");
    MPI_Bcast(line, MAXLINE, MPI_CHAR, 0, world);
    try {
      ValueTokenizer occ(utils::trim_comment(line));
      if (occ.count() != 1) error->all(FLERR, "Incorrect format in cluster occupancy file");
      clusterOccupancy[i] = occ.next_int();
    } catch (TokenizerException &e) {
      error->all(FLERR, "Incorrect format in cluster occupancy file: {}", e.what());
    }
  }

  if (comm->me == 0)
    utils::logmesg(lmp, "loaded cluster occupancy for active learning: {} values\n", ncols);
}

void EAPOD::peratombase_descriptors(double *bd1, double *bdd1, double *rij, double *temp,
        int *ti, int *tj, int Nj)
{
  for (int i=0; i<Mdesc; i++) bd1[i] = 0.0;
  for (int i=0; i<3*Nj*Mdesc; i++) bdd1[i] = 0.0;

  if (Nj == 0) return;

  double *d2 =  &bd1[0]; // nl2
  double *d3 =  &bd1[nl2]; // nl3
  double *d4 =  &bd1[nl2 + nl3]; // nl4
  double *d33 =  &bd1[nl2 + nl3 + nl4]; // nl33
  double *d34 =  &bd1[nl2 + nl3 + nl4 + nl33]; // nl34
  double *d44 =  &bd1[nl2 + nl3 + nl4 + nl33 + nl34]; // nl44

  double *dd2 = &bdd1[0]; // 3*Nj*nl2
  double *dd3 = &bdd1[3*Nj*nl2]; // 3*Nj*nl3
  double *dd4 = &bdd1[3*Nj*(nl2+nl3)]; // 3*Nj*nl4
  double *dd33 = &bdd1[3*Nj*(nl2+nl3+nl4)]; // 3*Nj*nl33
  double *dd34 = &bdd1[3*Nj*(nl2+nl3+nl4+nl33)]; // 3*Nj*nl34
  double *dd44 = &bdd1[3*Nj*(nl2+nl3+nl4+nl33+nl34)]; // 3*Nj*nl44

  int n1 = Nj*K3*nrbf3;
  int n2 = Nj*nrbf2;
  int n3 = Nj*ns;
  int n4 = Nj*K3;
  int n5 = K3*nrbf3*nelements;

  double *Ux = &temp[0]; // Nj*K3*nrbf3
  double *Uy = &temp[n1]; // Nj*K3*nrbf3
  double *Uz = &temp[2*n1]; // Nj*K3*nrbf3
  double *sumU = &temp[3*n1]; // K3*nrbf3*nelements

  double *rbf = &temp[3*n1 + n5]; // Nj*nrbf2
  double *drbf = &temp[3*n1 + n5 + n2]; // Nj*nrbf2

  if (use_spline) {
    radialbasis_spline(rbf, drbf, rij, ti, tj, Nj);
  } else {
    double *rbft = &temp[3*n1 + n5 + 2*n2]; // Nj*ns
    double *drbft = &temp[3*n1 + n5 + 2*n2 + n3]; // Nj*ns
    radialbasis(rbft, drbft, rij, rin, invrdiff, ti, tj, besseldegree, inversedegree, nbesselpars, Nj);
    radialPhi(rbf, drbf, rbft, drbft, ti, tj, Nj);
  }

  twobodydescderiv(d2, dd2, rbf, drbf, rij, tj, Nj);

  if ((nl3 > 0) && (Nj>1)) {
    double *abf = &temp[3*n1 + n5 + 2*n2]; // Nj*K3
    double *abfx = &temp[3*n1 + n5 + 2*n2 + n4]; // Nj*K3
    double *abfy = &temp[3*n1 + n5 + 2*n2 + 2*n4]; // Nj*K3
    double *abfz = &temp[3*n1 + n5 + 2*n2 + 3*n4]; // Nj*K3
    double *tm = &temp[3*n1 + n5 + 2*n2 + 4*n4]; // 4*K3

    angularbasis(abf, abfx, abfy, abfz, rij, tm, pq3, Nj, K3);

    radialangularbasis(sumU, Ux, Uy, Uz, rbf, drbf, rij,
            abf, abfx, abfy, abfz, tm, tj, Nj, K3, nrbf3, nelements);

    threebodydesc(d3, sumU, nelements);
    threebodydescderiv(dd3, sumU, Ux, Uy, Uz, tj, Nj);

    if ((nl33>0) && (Nj>3)) {
      crossdesc(d33, d3, d3, ind33l, ind33r, nl33);
      crossdescderiv(dd33, d3, d3, dd3, dd3, ind33l, ind33r, nl33, 3*Nj);
    }

    if ((nl4 > 0) && (Nj>2)) {
      if (K4 < K3) {
        for (int m=0; m<nrbf4; m++)
          for (int k=0; k<K4; k++)
            for (int i=0; i<nelements; i++)
              sumU[i + nelements*k + nelements*K4*m] = sumU[i + nelements*k + nelements*K3*m];

        for (int m=0; m<nrbf4; m++)
          for (int k=0; k<K4; k++)
            for (int i=0; i<Nj; i++) {
              int ii = i + Nj*k + Nj*K4*m;
              int jj = i + Nj*k + Nj*K3*m;
              Ux[ii] = Ux[jj];
              Uy[ii] = Uy[jj];
              Uz[ii] = Uz[jj];
            }
      }
      fourbodydescderiv(d4, dd4, sumU, Ux, Uy, Uz, tj, Nj);

      if ((nl34>0) && (Nj>4)) {
        crossdesc(d34, d3, d4, ind34l, ind34r, nl34);
        crossdescderiv(dd34, d3, d4, dd3, dd4, ind34l, ind34r, nl34, 3*Nj);
      }

      if ((nl44>0) && (Nj>5)) {
        crossdesc(d44, d4, d4, ind44l, ind44r, nl44);
        crossdescderiv(dd44, d4, d4, dd4, dd4, ind44l, ind44r, nl44, 3*Nj);
      }
    }
  }
}

double EAPOD::peratombase_coefficients(double *cb, double *bd, int *ti)
{
  int nc = nCoeffPerElement*ti[0];

  double ei = coeff[0 + nc];
  for (int m=0; m<Mdesc; m++) {
    ei += coeff[1 + m + nc]*bd[m];
    cb[m] = coeff[1 + m + nc];
  }

  return ei;
}

inline void EAPOD::cluster_cutoff_hat(double pc, double inv_rcut2, double &fcut, double &dfcut)
{
  constexpr int    p  = 2;
  constexpr int    q  = 4;
  constexpr double PQ = -p * q;

  double y2     = pc * pc * inv_rcut2;
  double y2pm1  = powint(y2, p - 1);
  double y2p    = y2pm1 * y2;
  double omy    = 1.0 - y2p;
  double omq1   = powint(omy, q - 1);

  fcut  = omq1 * omy;
  dfcut = PQ * inv_rcut2 * y2pm1 * omq1;
}

inline void EAPOD::cluster_cutoff_poly_sq(double pc, double inv_rcut2, double &fcut, double &dfcut)
{
  double u = pc * pc * inv_rcut2;

  double u2 = u * u;
  double omu = 1.0 - u;

  // fcut(u) = 1 - 10 * u3 + 15 * u4 - 6 * u5;
  fcut = 1.0 - u2 * u * (10.0 - u * (15.0 - 6.0 * u));
  dfcut = -30.0 * u2 * omu * omu * inv_rcut2;
}

// Main Routine to find active clusters.
// Binary searches O(Log(nClusters) + Log(nMaxActiveClusters))
// Active range in ledges[k] <= pca < redges[k]
inline void EAPOD::find_active_clusters(double pca, double* ledges, double* redges,
                                        const int nClusters, const int nMaxActiveClusters,
                                        int& ks, int& ke)
{
  ks = static_cast<int>(std::lower_bound(redges, redges + nClusters, pca) - redges);
  int kend = std::min(nClusters, ks + nMaxActiveClusters);
  ke = static_cast<int>(std::upper_bound(ledges + ks + 1, ledges + kend, pca) - ledges);
}

double EAPOD::peratom_local_environment_descriptors(double *cb, double *bd, double *tm, int *ti)
{
  int itype = ti[0];
  int nc    = nCoeffPerElement*itype;
  int ncct  = nClusters*itype;

  double *proj = &Proj[Mdesc*itype];
  
  double pca = 0.0;
  for (int m = 0; m < Mdesc; m++)
    pca += proj[m] * bd[m];

  double *ledges = &leftClusterEdges[ncct];
  double *redges = &rightClusterEdges[ncct];

  int ks = 0;
  int kn = nClusters;
  find_active_clusters(pca, ledges, redges, nClusters, nMaxActiveClusters, ks, kn);
  kn -= ks;
  ncct += ks;

  double *cent     = &Centroids[ncct];
  double *invlcut2 = &invLeftClusterRcut2[ncct];
  double *invrcut2 = &invRightClusterRcut2[ncct];

  double *D       = &tm[0];
  double *dD_dpca = &tm[nMaxActiveClusters];

  double fcut, dfcut;   // fcut(pc), dfcut/dpc
  double sumfDi = 0.0;
  double S      = 0.0;
  for (int k = 0; k < kn; k++) {
    double pc        = pca - cent[k];
    double inv_rcut2 = (pc >= 0.0) ? invrcut2[k] : invlcut2[k];

    cluster_cutoff_hat(pc, inv_rcut2, fcut, dfcut);

    double invDk = 1.0 / (pc * pc + 1e-20);
    double fDk   = fcut * invDk;
    double dfDkdpc = 2.0 * pc * invDk * (dfcut - fDk);

    D[k]       = fDk;
    dD_dpca[k] = dfDkdpc;
    sumfDi    += fDk;
    S         += dfDkdpc;
  }
  sumfDi = 1.0 / sumfDi;
  
  double ei = coeff[nc];
  double *ceffs = &coeff[1 + ks*Mdesc + nc];
  double T  = 0.0;
  double A  = 0.0;
  for (int k = 0; k < kn; k++) {
    double sumE = 0.0;
    for (int m = 0; m < Mdesc; m++)
      sumE += ceffs[m + k*Mdesc] * bd[m];
    double Pk = D[k] * sumfDi;
    double cpk = sumE * sumfDi;
    D[k]  = Pk;
    ei    += sumE * Pk;
    T     += cpk * Pk;
    A     += cpk * dD_dpca[k];
  }

  const double U = A - S * T;
  for (int m = 0; m < Mdesc; m++) {
    double sum = U * proj[m];
    for (int k = 0; k < kn; k++)
      sum += ceffs[m + k*Mdesc] * D[k];
    cb[m] = sum;
  }

  return ei;
}

double EAPOD::peratom_environment_descriptors(double *cb, double *bd, double *tm, int *ti)
{
  double *P    = &tm[0];    // nClusters
  double *cp   = &tm[(nClusters)];  // nClusters
  double *D    = &tm[(2*nClusters)];   // nClusters
  double *pca  = &tm[(3*nClusters)]; // nComponents

  double *proj = &Proj[0];
  double *cent = &Centroids[0];
  int typei = ti[0];

  for (int k=0; k<nComponents; k++) {
    double sum = 0.0;
    for (int m = 0; m < Mdesc; m++) {
      sum += proj[k + nComponents*m + nComponents*Mdesc*typei] * bd[m];
    }
    pca[k] = sum;
  }

  for (int j=0; j<nClusters; j++) {
    double sum = 1e-20;
    for (int k = 0; k < nComponents; k++) {
      double c = cent[k + j * nComponents + nClusters*nComponents*typei];
      double p = pca[k];
      sum += (p - c) * (p - c);
    }
    D[j] = 1.0 / sum;
  }

  double sum = 0.0;
  for (int j = 0; j < nClusters; j++) sum += D[j];
  double sumD = sum;
  if (sum != 0.0) {
    for (int j = 0; j < nClusters; j++) P[j] = D[j]/sum;
  } else {
    for (int j = 0; j < nClusters; j++) P[j] = 0.0;
  }

  int nc = nCoeffPerElement*ti[0];
  double ei = coeff[0 + nc];
  for (int k = 0; k<nClusters; k++)
    for (int m=0; m<Mdesc; m++)
      ei += coeff[1 + m + Mdesc*k + nc]*bd[m]*P[k];

  for (int k=0; k<nClusters; k++) {
    double sum = 0;
    for (int m = 0; m<Mdesc; m++)
      sum += coeff[1 + m + k*Mdesc + nc]*bd[m];
    cp[k] = sum;
  }

  for (int m = 0; m<Mdesc; m++) {
    double sum = 0.0;
    for (int k = 0; k<nClusters; k++)
      sum += coeff[1 + m + k*Mdesc + nc]*P[k];
    cb[m] = sum;
  }

  for (int m = 0; m<Mdesc; m++) {
    double S1 = 1.0/sumD;
    double S2 = S1*S1;
    double sum = 0.0;
    for (int j=0; j<nClusters; j++) {
      double dP_dB = 0.0;
      for (int k = 0; k < nClusters; k++) {
        double dP_dD = -D[j] * S2;
        if (k==j) dP_dD += S1;
        double dD_dB = 0.0;
        double D2 = 2 * D[k] * D[k];
        for (int n = 0; n < nComponents; n++) {
          double dD_dpca = D2 * (cent[n + k * nComponents + nClusters*nComponents*typei] - pca[n]);
          dD_dB += dD_dpca * proj[n + m * nComponents + nComponents*Mdesc*typei];
        }
        dP_dB += dP_dD * dD_dB;
      }
      sum += cp[j]*dP_dB;
    }
    cb[m] += sum;
  }

  return ei;
}

void EAPOD::twobody_forces(double *fij, double *cb2, double *drbf, double *rij, int *tj, int Nj)
{
  for (int n = 0; n < Nj; ++n) {
    const double *c = &cb2[nrbf2*tj[n]];
    double fr = 0.0;
    for (int m = 0; m < nrbf2; ++m)
      fr += c[m]*drbf[n + Nj*m];
    const int i1 = 3*n;
    fij[i1] += fr * rij[0 + i1];
    fij[i1+1] += fr * rij[1 + i1];
    fij[i1+2] += fr * rij[2 + i1];
  }
}

void EAPOD::threebody_forcecoeff(double *fb3, double *cb3, double *sumU)
{
  if (nelements == 1) {
    for (int m = 0; m < nrbf3; ++m) {
      for (int a = 0; a < nabf3_active; a++) {
        int p  = p3_active[a];
        double c3 = 2.0 * cb3[a + nabf3_active * m];

        int n1 = pn3[p];
        int n2 = pn3[p + 1];
        int nn = n2 - n1;
        int idxU = K3 * m;

        for (int q = 0; q < nn; q++) {
          int k = n1 + q;
          fb3[k + idxU] += c3 * pc3[k] * sumU[k + idxU];
        }
      }
    }
  } else {
    int N3 = nabf3_active * nrbf3;
    for (int m = 0; m < nrbf3; ++m) {
      for (int a = 0; a < nabf3_active; ++a) {
        const int p   = p3_active[a];
        const int n1  = pn3[p];
        const int n2  = pn3[p + 1];
        const int jmp = a + nabf3_active * m;
      
        for (int k = n1; k < n2; ++k) {
          const double pk = pc3[k];
          const int idxU  = nelements * (k + K3 * m);
        
          int em = 0;
          for (int i1 = 0; i1 < nelements; ++i1) {
            const double u1 = sumU[idxU + i1];
            for (int i2 = i1; i2 < nelements; ++i2, ++em) {
              const double w  = pk * cb3[jmp + N3 * em];
              const double u2 = sumU[idxU + i2];
            
              fb3[idxU + i2] += w * u1;
              fb3[idxU + i1] += w * u2;
            }
          }
        }
      }
    }
  }
}

void EAPOD::fourbody_forcecoeff(double *fb4, double *cb4, double *sumU)
{
  if (nelements == 1) {
    for (int m = 0; m < nrbf4; ++m) {
      int idxU = K3 * m;
      for (int a = 0; a < nabf4_active; a++) {
        int p  = p4_active[a];
        int n1 = pa4[p];
        int n2 = pa4[p + 1];
        int nn = n2 - n1;
        double c4 = cb4[a + nabf4_active * m];

        for (int q = 0; q < nn; q++) {
          int iq = n1 + q;
          int c  = pc4[iq];
          int j1 = idxU + pb4[iq];
          int j2 = idxU + pb4[iq + Q4];
          int j3 = idxU + pb4[iq + 2 * Q4];

          double c1 = sumU[j1];
          double c2 = sumU[j2];
          double c3 = sumU[j3];

          fb4[j3] += c4 * c * c1 * c2;
          fb4[j2] += c4 * c * c1 * c3;
          fb4[j1] += c4 * c * c2 * c3;
        }
      }
    }
  } else {
    int N3 = nabf4_active * nrbf4;
    for (int m = 0; m < nrbf4; ++m) {
      for (int a = 0; a < nabf4_active; a++) {
        int p  = p4_active[a];
        int n1 = pa4[p];
        int n2 = pa4[p + 1];
        int nn = n2 - n1;
        int jpm = a + nabf4_active * m;

        for (int q = 0; q < nn; q++) {
          int iq = n1 + q;
          int c  = pc4[iq];
          int j1 = pb4[iq];
          int j2 = pb4[iq + Q4];
          int j3 = pb4[iq + 2 * Q4];

          int idx1 = nelements * j1 + nelements * K3 * m;
          int idx2 = nelements * j2 + nelements * K3 * m;
          int idx3 = nelements * j3 + nelements * K3 * m;

          int k = 0;
          for (int i1 = 0; i1 < nelements; i1++) {
            double c1 = sumU[idx1 + i1];
            for (int i2 = i1; i2 < nelements; i2++) {
              double c2 = sumU[idx2 + i2];
              for (int i3 = i2; i3 < nelements; i3++) {
                double c3 = sumU[idx3 + i3];
                double c4 = c * cb4[jpm + N3 * k];
                fb4[idx3 + i3] += c4 * (c1 * c2);
                fb4[idx2 + i2] += c4 * (c1 * c3);
                fb4[idx1 + i1] += c4 * (c2 * c3);
                k += 1;
              }
            }
          }
        }
      }
    }
  }
}

void EAPOD::allbody_forces(double *fij, double *forcecoeff, double *rbf, double *drbf, double *rij,
                           double *abf, double *abfx, double *abfy, double *abfz, int *tj, int Nj)
{
  const int Ne = nelements, K = K3;

  for (int n = 0; n < Nj; ++n) {
    const int e = tj[n];
    double fdRA  = 0.0;
    double fRdAx = 0.0;
    double fRdAy = 0.0;
    double fRdAz = 0.0;
    for (int k = 0; k < K; ++k) {
      double fdR = 0.0;
      double fR = 0.0;
      for (int m = 0; m < nrbf3; ++m) {
        const double fc = forcecoeff[e + Ne * (k + K * m)];
        const int id = n + Nj * m;
        fdR += fc * drbf[id];
        fR  += fc * rbf [id];
      }
      const int ia = n + Nj * k;
      fdRA += fdR * abf [ia];
      fRdAx += fR * abfx[ia];
      fRdAy += fR * abfy[ia];
      fRdAz += fR * abfz[ia];
    }
    const int i1 = 3 * n;
    fij[i1]     += fRdAx + fdRA * rij[i1];
    fij[i1 + 1] += fRdAy + fdRA * rij[i1 + 1];
    fij[i1 + 2] += fRdAz + fdRA * rij[i1 + 2];
  }
}

double EAPOD::peratomenergyforce2(double *fij, double *rij, double *temp,
        int *ti, int *tj, int Nj)
{
  //double *coeff1 = &coeff[nCoeffPerElement*ti[0]];
  if (Nj==0) return coeff[nCoeffPerElement*ti[0]];

  int N = 3*Nj;
  for (int n=0; n<N; n++) fij[n] = 0.0;

  double e = 0.0;
  for (int i=0; i<Mdesc; i++) bd[i] = 0.0;

  double *d2  = &bd[0]; // nl2
  double *d3  = &bd[nl2]; // nl3
  double *d4  = &bd[nl2 + nl3]; // nl4
  double *d33 = &bd[nl2 + nl3 + nl4]; // nl33
  double *d34 = &bd[nl2 + nl3 + nl4 + nl33]; // nl34
  double *d44 = &bd[nl2 + nl3 + nl4 + nl33 + nl34]; // nl44

  int n2 = Nj*nrbf2;
  int n3 = Nj*ns;
  int n4 = Nj*K3;
  int n5 = K3*nrbf3*nelements;

  double *sumU = &temp[0]; // K3*nrbf3*nelements

  double *rbf = &temp[n5]; // Nj*nrbf2
  double *drbf = &temp[n5 + n2]; // Nj*nrbf2

  if (use_spline) {
    radialbasis_spline(rbf, drbf, rij, ti, tj, Nj);
  } else {
    double *rbft = &temp[n5 + 2*n2]; // Nj*ns
    double *drbft = &temp[n5 + 2*n2 + n3]; // Nj*ns
    radialbasis(rbft, drbft, rij, rin, invrdiff, ti, tj, besseldegree, inversedegree, nbesselpars, Nj);
    radialPhi(rbf, drbf, rbft, drbft, ti, tj, Nj);
  }

  twobodydesc(d2, rbf, tj, Nj, nelements);

  double *abf = &temp[n5 + 2*n2]; // Nj*K3
  double *abfx = &temp[n5 + 2*n2 + n4]; // Nj*K3
  double *abfy = &temp[n5 + 2*n2 + 2*n4]; // Nj*K3
  double *abfz = &temp[n5 + 2*n2 + 3*n4]; // Nj*K3
  double *tm = &temp[n5 + 2*n2 + 4*n4]; // 4*K3

  if ((nl3 > 0) && (Nj>1)) {
    angularbasis(abf, abfx, abfy, abfz, rij, tm, pq3, Nj, K3);

    radialangularsum(sumU, rbf, abf, tj, Nj, K3, nrbf3, nelements);

    threebodydesc(d3, sumU, nelements);

    if ((nl33>0) && (Nj>3)) crossdesc(d33, d3, d3, ind33l, ind33r, nl33);

    if ((nl4 > 0) && (Nj>2)) {
      fourbodydesc(d4, sumU);

      if ((nl34>0) && (Nj>4)) crossdesc(d34, d3, d4, ind34l, ind34r, nl34);
      if ((nl44>0) && (Nj>5)) crossdesc(d44, d4, d4, ind44l, ind44r, nl44);
    }
  }

  double *cb = &bdd[0];
  if (localeapod) e += peratom_local_environment_descriptors(cb, bd, tm, ti);
  else if (eapod) e += peratom_environment_descriptors(cb, bd, tm, ti);
  else            e += peratombase_coefficients(cb, bd, ti);

  double *cb2  = &cb[0]; // nl3
  double *cb3  = &cb[nl2]; // nl3
  double *cb4  = &cb[nl2 + nl3]; // nl4
  double *cb33 = &cb[nl2 + nl3 + nl4]; // nl33
  double *cb34 = &cb[nl2 + nl3 + nl4 + nl33]; // nl34
  double *cb44 = &cb[nl2 + nl3 + nl4 + nl33 + nl34]; // nl44

  if ((nl33>0) && (Nj>3)) crossdesc_reduction(cb3, cb3, cb33, d3, d3, ind33l, ind33r, nl33);
  if ((nl34>0) && (Nj>4)) crossdesc_reduction(cb3, cb4, cb34, d3, d4, ind34l, ind34r, nl34);
  if ((nl44>0) && (Nj>5)) crossdesc_reduction(cb4, cb4, cb44, d4, d4, ind44l, ind44r, nl44);

  twobody_forces(fij, cb2, drbf, rij, tj, Nj);

  // Initialize forcecoeff to zero
  double *forcecoeff = &cb[nl2 + nl3 + nl4]; // nelements*K3*nrbf3
  memset(forcecoeff, 0, nelements * K3 * nrbf3 * sizeof(*forcecoeff));
  if ((nl3 > 0) && (Nj>1)) threebody_forcecoeff(forcecoeff, cb3, sumU);
  if ((nl4 > 0) && (Nj>2)) fourbody_forcecoeff(forcecoeff, cb4, sumU);
  if ((nl3 > 0) && (Nj>1)) allbody_forces(fij, forcecoeff, rbf, drbf, rij, abf, abfx, abfy, abfz, tj, Nj);

  return e;
}

double EAPOD::peratomenergyforce(double *fij, double *rij, double *temp,
        int *ti, int *tj, int Nj)
{
  double *coeff1 = &coeff[nCoeffPerElement*ti[0]];
  double e = coeff1[0];
  if (Nj==0) return e;

  int N = 3*Nj;
  for (int n=0; n<N; n++) fij[n] = 0.0;

  // calculate base descriptors and their derivatives with respect to atom coordinates
  peratombase_descriptors(bd, bdd, rij, temp, ti, tj, Nj);

  if (eapod) { // multi-environment descriptors
    if (localeapod) peratomlocalenvironment_descriptors(pd, pdd, bd, bdd, temp, ti[0], Nj);
    else            peratomenvironment_descriptors(pd, pdd, bd, bdd, temp, ti[0], Nj);

    for (int j = 0; j<nClusters; j++)
      for (int m=0; m<Mdesc; m++)
        e += coeff1[1 + m + j*Mdesc]*bd[m]*pd[j];

    double *cb = &temp[0];
    double *cp = &temp[Mdesc];
    for (int m = 0; m<Mdesc; m++) cb[m] = 0.0;
    for (int j = 0; j<nClusters; j++) cp[j] = 0.0;
    for (int j = 0; j<nClusters; j++)
      for (int m = 0; m<Mdesc; m++)  {
        cb[m] += coeff1[1 + m + j*Mdesc]*pd[j];
        cp[j] += coeff1[1 + m + j*Mdesc]*bd[m];
      }
    char chn = 'N';
    double alpha = 1.0, beta = 0.0;
    int inc1 = 1;
    DGEMV(&chn, &N, &Mdesc, &alpha, bdd, &N, cb, &inc1, &beta, fij, &inc1);
    DGEMV(&chn, &N, &nClusters, &alpha, pdd, &N, cp, &inc1, &alpha, fij, &inc1);
  }
  else { // single-environment descriptors
    for (int m=0; m<Mdesc; m++)
      e += coeff1[1+m]*bd[m];

    char chn = 'N';
    double alpha = 1.0, beta = 0.0;
    int inc1 = 1;
    DGEMV(&chn, &N, &Mdesc, &alpha, bdd, &N, &coeff1[1], &inc1, &beta, fij, &inc1);
  }

  return e;
}

double EAPOD::energyforce(double *force, double *x, int *atomtype, int *alist,
          int *jlist, int *pairnumsum, int natom)
{
  double etot = 0.0;
  for (int i=0; i<3*natom; i++) force[i] = 0.0;

  for (int i=0; i<natom; i++) {
    int Nj = pairnumsum[i+1] - pairnumsum[i]; // # neighbors around atom i

    if (Nj==0) {
      etot += coeff[nCoeffPerElement*(atomtype[i]-1)];
    }
    else
    {
      // reallocate temporary memory
      if (Nj>Njmax) {
        Njmax = Nj;
        free_temp_memory();
        allocate_temp_memory(Njmax);
      }

      double *rij = &tmpmem[0];    // 3*Nj
      double *fij = &tmpmem[3*Nj]; // 3*Nj
      int *ai = &tmpint[0];        // Nj
      int *aj = &tmpint[Nj];       // Nj
      int *ti = &tmpint[2*Nj];     // Nj
      int *tj = &tmpint[3*Nj];     // Nj

      myneighbors(rij, x, ai, aj, ti, tj, jlist, pairnumsum, atomtype, alist, i);

      etot += peratomenergyforce(fij, rij, &tmpmem[6*Nj], ti, tj, Nj);

      tallyforce(force, fij, ai, aj, Nj);
    }
  }

  return etot;
}

void EAPOD::base_descriptors(double *basedesc, double *x,
        int *atomtype, int *alist, int *jlist, int *pairnumsum, int natom)
{
  for (int i=0; i<natom*Mdesc; i++) basedesc[i] = 0.0;

  for (int i=0; i<natom; i++) {
    int Nj = pairnumsum[i+1] - pairnumsum[i]; // # neighbors around atom i

    if (Nj>0) {
      // reallocate temporary memory
      if (Nj>Njmax) {
        Njmax = Nj;
        free_temp_memory();
        allocate_temp_memory(Njmax);
        if (comm->me == 0) utils::logmesg(lmp, "reallocate temporary memory with Njmax = {:d} ...\n", Njmax);
      }

      double *rij = &tmpmem[0]; // 3*Nj
      int *ai = &tmpint[0];     // Nj
      int *aj = &tmpint[Nj];   // Nj
      int *ti = &tmpint[2*Nj]; // Nj
      int *tj = &tmpint[3*Nj]; // Nj

      myneighbors(rij, x, ai, aj, ti, tj, jlist, pairnumsum, atomtype, alist, i);

      // many-body descriptors
      peratombase_descriptors(bd, bdd, rij, &tmpmem[3*Nj], ti, tj, Nj);

      for (int m=0; m<Mdesc; m++) {
        basedesc[i + natom*m] = bd[m];
      }

    }
  }
}

void EAPOD::descriptors(double *gd, double *gdd, double *basedesc, double *x,
        int *atomtype, int *alist, int *jlist, int *pairnumsum, int natom)
{
  for (int i=0; i<nd; i++) gd[i] = 0.0;
  for (int i=0; i<3*natom*nd; i++) gdd[i] = 0.0;
  for (int i=0; i<natom*Mdesc; i++) basedesc[i] = 0.0;

  for (int i=0; i<natom; i++) {
    int Nj = pairnumsum[i+1] - pairnumsum[i]; // # neighbors around atom i

    // one-body descriptor
    if (nd1>0) {
      gd[nCoeffPerElement*(atomtype[i]-1)] += 1.0;
    }

    if (Nj>0) {
      // reallocate temporary memory
      if (Nj>Njmax) {
        Njmax = Nj;
        free_temp_memory();
        allocate_temp_memory(Njmax);
        if (comm->me == 0) utils::logmesg(lmp, "reallocate temporary memory with Njmax = {:d} ...\n", Njmax);
      }

      double *rij = &tmpmem[0]; // 3*Nj
      int *ai = &tmpint[0];     // Nj
      int *aj = &tmpint[Nj];   // Nj
      int *ti = &tmpint[2*Nj]; // Nj
      int *tj = &tmpint[3*Nj]; // Nj

      myneighbors(rij, x, ai, aj, ti, tj, jlist, pairnumsum, atomtype, alist, i);

      // many-body descriptors
      peratombase_descriptors(bd, bdd, rij, &tmpmem[3*Nj], ti, tj, Nj);

      for (int m=0; m<Mdesc; m++) {
        basedesc[i + natom*m] = bd[m];
        int k = nCoeffPerElement*ti[0] + nl1 + m; // increment by nl1 because of the one-body descriptor
        gd[k] += bd[m];
        for (int n=0; n<Nj; n++) {
          int im = 3*ai[n] + 3*natom*k;
          int jm = 3*aj[n] + 3*natom*k;
          int nm = 3*n + 3*Nj*m;
          gdd[0 + im] += bdd[0 + nm];
          gdd[1 + im] += bdd[1 + nm];
          gdd[2 + im] += bdd[2 + nm];
          gdd[0 + jm] -= bdd[0 + nm];
          gdd[1 + jm] -= bdd[1 + nm];
          gdd[2 + jm] -= bdd[2 + nm];
        }
      }

    }
  }
}

void EAPOD::descriptors(double *gd, double *gdd, double *basedesc, double *probdesc, double *x,
        int *atomtype, int *alist, int *jlist, int *pairnumsum, int natom)
{
  for (int i=0; i<nCoeffAll; i++) gd[i] = 0.0;
  for (int i=0; i<3*natom*nCoeffAll; i++) gdd[i] = 0.0;
  for (int i=0; i<natom*Mdesc; i++) basedesc[i] = 0.0;
  for (int i=0; i<natom*nClusters; i++) probdesc[i] = 0.0;

  for (int i=0; i<natom; i++) {
    int Nj = pairnumsum[i+1] - pairnumsum[i]; // # neighbors around atom i

    // one-body descriptor
    if (nd1>0) {
      gd[nCoeffPerElement*(atomtype[i]-1)] += 1.0;
    }

    if (Nj>0) {
      // reallocate temporary memory
      if (Nj>Njmax) {
        Njmax = Nj;
        free_temp_memory();
        allocate_temp_memory(Njmax);
        if (comm->me == 0) utils::logmesg(lmp, "reallocate temporary memory with Njmax = {:d} ...\n", Njmax);
      }

      double *rij = &tmpmem[0]; // 3*Nj
      int *ai = &tmpint[0];     // Nj
      int *aj = &tmpint[Nj];   // Nj
      int *ti = &tmpint[2*Nj]; // Nj
      int *tj = &tmpint[3*Nj]; // Nj

      myneighbors(rij, x, ai, aj, ti, tj, jlist, pairnumsum, atomtype, alist, i);

      // many-body descriptors
      peratombase_descriptors(bd, bdd, rij, &tmpmem[3*Nj], ti, tj, Nj);

      // calculate multi-environment descriptors and their derivatives with respect to atom coordinates
      if (localeapod) peratomlocalenvironment_descriptors(pd, pdd, bd, bdd, tmpmem, ti[0], Nj);
      else            peratomenvironment_descriptors(pd, pdd, bd, bdd, tmpmem, ti[0], Nj);

      for (int j = 0; j < nClusters; j++) {
        probdesc[i + natom*j] = pd[j];
        for (int m=0; m<Mdesc; m++) {
          basedesc[i + natom*m] = bd[m];
          int k = nCoeffPerElement*ti[0] + nl1 + m + j*Mdesc; // increment by nl1 because of the one-body descriptor
          gd[k] += pd[j]*bd[m];
          for (int n=0; n<Nj; n++) {
            int im = 3*ai[n] + 3*natom*k;
            int jm = 3*aj[n] + 3*natom*k;
            int nm = 3*n + 3*Nj*m;
            int nj = 3*n + 3*Nj*j;
            gdd[0 + im] += bdd[0 + nm]*pd[j] + bd[m]*pdd[0 + nj];
            gdd[1 + im] += bdd[1 + nm]*pd[j] + bd[m]*pdd[1 + nj];
            gdd[2 + im] += bdd[2 + nm]*pd[j] + bd[m]*pdd[2 + nj];
            gdd[0 + jm] -= bdd[0 + nm]*pd[j] + bd[m]*pdd[0 + nj];
            gdd[1 + jm] -= bdd[1 + nm]*pd[j] + bd[m]*pdd[1 + nj];
            gdd[2 + jm] -= bdd[2 + nm]*pd[j] + bd[m]*pdd[2 + nj];
          }
        }
      }

    }
  }
}

void EAPOD::crossdesc(double *d12, double *d1, double *d2, int *ind1, int *ind2, int n12)
{
  for (int i = 0; i<n12; i++)
    d12[i] = d1[ind1[i]]*d2[ind2[i]];
}

void EAPOD::crossdescderiv(double *dd12, double *d1, double *d2, double *dd1, double *dd2,
        int *ind1, int *ind2, int n12, int N)
{
  for (int i = 0; i<n12; i++)
    for (int n=0; n<N; n++)
      dd12[n + N*i] = d1[ind1[i]]*dd2[n + N*ind2[i]] + dd1[n + N*ind1[i]]*d2[ind2[i]];
}

void EAPOD::crossdesc_reduction(double *cb1, double *cb2, double *c12, double *d1,
        double *d2, int *ind1, int *ind2, int n12)
{
  for (int m = 0; m < n12; m++) {
    int k1 = ind1[m]; // dd1
    int k2 = ind2[m]; // dd2
    double c = c12[m];
    cb1[k1] += c * d2[k2];
    cb2[k2] += c * d1[k1];
  }
}

void EAPOD::myneighbors(double *rij, double *x, int *ai, int *aj, int *ti, int *tj,
        int *jlist, int *pairnumsum, int *atomtype, int *alist, int i)
{
  int itype = atomtype[i] - 1;
  int start = pairnumsum[i];
  int m = pairnumsum[i+1] - start; // number of neighbors around i
  for (int l=0; l<m ; l++) {   // loop over each atom around atom i
    int j = jlist[l + start];  // atom j
    ai[l]        = i;
    aj[l]        = alist[j];
    ti[l]        = itype;
    tj[l]        = atomtype[alist[j]] - 1;
    rij[0 + 3*l]   = x[0 + 3*j] -  x[0 + 3*i];
    rij[1 + 3*l]   = x[1 + 3*j] -  x[1 + 3*i];
    rij[2 + 3*l]   = x[2 + 3*j] -  x[2 + 3*i];
  }
}

void EAPOD::fourbodydesc(double *d4, double *sumU)
{
  const int Me = nelements * (nelements + 1) * (nelements + 2) / 6;
  memset(d4, 0, nabf4_active * nrbf4 * Me * sizeof(*d4));

  for (int m = 0; m < nrbf4; m++) {
    int idxU = nelements * K3 * m;
    for (int a = 0; a < nabf4_active; a++) {
      int p  = p4_active[a];
      int n1 = pa4[p];
      int n2 = pa4[p + 1];
      int nn = n2 - n1;

      for (int q = 0; q < nn; q++) {
        int iq = n1 + q;
        int c  = pc4[iq];
        int j1 = pb4[iq];
        int j2 = pb4[iq + Q4];
        int j3 = pb4[iq + 2 * Q4];

        int k = 0;
        for (int i1 = 0; i1 < nelements; i1++) {
          double c1 = sumU[idxU + i1 + nelements * j1];
          for (int i2 = i1; i2 < nelements; i2++) {
            double c2  = sumU[idxU + i2 + nelements * j2];
            double t12 = c * c1 * c2;
            for (int i3 = i2; i3 < nelements; i3++) {
              double c3 = sumU[idxU + i3 + nelements * j3];
              int kk = a + nabf4_active * m + nabf4_active * nrbf4 * k;
              d4[kk] += t12 * c3;
              k += 1;
            }
          }
        }
      }
    }
  }
}

void EAPOD::fourbodydescderiv(double *d4, double *dd4, double *sumU,
                              double *Ux, double *Uy, double *Uz, int *tj, int N)
{
  const int Me = nelements * (nelements + 1) * (nelements + 2) / 6;
  memset(d4, 0, nabf4_active * nrbf4 * Me * sizeof(*d4));
  memset(dd4, 0, 3 * N * nabf4_active * nrbf4 * Me * sizeof(*dd4));

  if (nelements == 1) {
    for (int m = 0; m < nrbf4; m++) {
      for (int a = 0; a < nabf4_active; a++) {
        int p  = p4_active[a];
        int n1 = pa4[p];
        int n2 = pa4[p + 1];
        int nn = n2 - n1;

        for (int q = 0; q < nn; q++) {
          int iq = n1 + q;
          int c  = pc4[iq];
          int j1 = pb4[iq];
          int j2 = pb4[iq + Q4];
          int j3 = pb4[iq + 2 * Q4];

          double c1  = c * sumU[j1 + K4 * m];
          double c2  = c * sumU[j2 + K4 * m];
          double t12 = c1 * sumU[j2 + K4 * m];
          double c3  = sumU[j3 + K4 * m];
          double t13 = c1 * c3;
          double t23 = c2 * c3;

          int kk = a + nabf4_active * m;
          int ii = 3 * N * (a + nabf4_active * m);
          d4[kk] += t12 * c3;

          for (int j = 0; j < N; j++) {
            int jj = j + N * j3 + N * K4 * m;
            dd4[0 + 3 * j + ii] += t12 * Ux[jj];
            dd4[1 + 3 * j + ii] += t12 * Uy[jj];
            dd4[2 + 3 * j + ii] += t12 * Uz[jj];

            jj = j + N * j2 + N * K4 * m;
            dd4[0 + 3 * j + ii] += t13 * Ux[jj];
            dd4[1 + 3 * j + ii] += t13 * Uy[jj];
            dd4[2 + 3 * j + ii] += t13 * Uz[jj];

            jj = j + N * j1 + N * K4 * m;
            dd4[0 + 3 * j + ii] += t23 * Ux[jj];
            dd4[1 + 3 * j + ii] += t23 * Uy[jj];
            dd4[2 + 3 * j + ii] += t23 * Uz[jj];
          }
        }
      }
    }
  } else {
    for (int m = 0; m < nrbf4; m++) {
      for (int a = 0; a < nabf4_active; a++) {
        int p  = p4_active[a];
        int n1 = pa4[p];
        int n2 = pa4[p + 1];
        int nn = n2 - n1;

        for (int q = 0; q < nn; q++) {
          int iq = n1 + q;
          int c  = pc4[iq];
          int j1 = pb4[iq];
          int j2 = pb4[iq + Q4];
          int j3 = pb4[iq + 2 * Q4];

          int k = 0;
          for (int i1 = 0; i1 < nelements; i1++) {
            double c1 = c * sumU[i1 + nelements * j1 + nelements * K4 * m];
            for (int i2 = i1; i2 < nelements; i2++) {
              double c2  = c * sumU[i2 + nelements * j2 + nelements * K4 * m];
              double t12 = c1 * sumU[i2 + nelements * j2 + nelements * K4 * m];
              for (int i3 = i2; i3 < nelements; i3++) {
                double c3  = sumU[i3 + nelements * j3 + nelements * K4 * m];
                double t13 = c1 * c3;
                double t23 = c2 * c3;

                int kk = a + nabf4_active * m + nabf4_active * nrbf4 * k;
                int ii = 3 * N * (a + nabf4_active * m + nabf4_active * nrbf4 * k);
                d4[kk] += t12 * c3;

                for (int j = 0; j < N; j++) {
                  int jtype = tj[j];
                  if (jtype == i3) {
                    int jj = j + N * j3 + N * K4 * m;
                    dd4[0 + 3 * j + ii] += t12 * Ux[jj];
                    dd4[1 + 3 * j + ii] += t12 * Uy[jj];
                    dd4[2 + 3 * j + ii] += t12 * Uz[jj];
                  }
                  if (jtype == i2) {
                    int jj = j + N * j2 + N * K4 * m;
                    dd4[0 + 3 * j + ii] += t13 * Ux[jj];
                    dd4[1 + 3 * j + ii] += t13 * Uy[jj];
                    dd4[2 + 3 * j + ii] += t13 * Uz[jj];
                  }
                  if (jtype == i1) {
                    int jj = j + N * j1 + N * K4 * m;
                    dd4[0 + 3 * j + ii] += t23 * Ux[jj];
                    dd4[1 + 3 * j + ii] += t23 * Uy[jj];
                    dd4[2 + 3 * j + ii] += t23 * Uz[jj];
                  }
                }
                k += 1;
              }
            }
          }
        }
      }
    }
  }
}

void EAPOD::threebodydesc(double *d3, double *sumU, int Ne)
{
  const int nAB  = nabf3_active * nrbf3;
  const int NeK3 = Ne * K3;

  if (Ne == 1) {
    for (int m = 0, mK3 = 0, mAB = 0; m < nrbf3; ++m, mK3 += K3, mAB += nabf3_active) {
      for (int a = 0; a < nabf3_active; ++a) {
        int p  = p3_active[a];
        int n1 = pn3[p];
        int n2 = pn3[p + 1];

        double acc = 0.0;
        int suIdx = mK3 + n1;
        for (int q = n1; q < n2; ++q, ++suIdx) {
          double u = sumU[suIdx];
          acc += pc3[q] * u * u;
        }
        d3[mAB + a] = acc;
      }
    }
  } else {
    for (int m = 0, mSu = 0, mAB = 0; m < nrbf3; ++m, mSu += NeK3, mAB += nabf3_active) {
      for (int a = 0; a < nabf3_active; ++a) {
        int p  = p3_active[a];
        int n1 = pn3[p];
        int n2 = pn3[p + 1];
        int d3base = mAB + a;

        int ep = 0;
        for (int i1 = 0; i1 < Ne; ++i1) {
          for (int i2 = i1; i2 < Ne; ++i2, ++ep) {
            double acc = 0.0;
            int suBase = mSu + Ne * n1;
            for (int q = n1; q < n2; ++q, suBase += Ne) {
              acc += pc3[q] * sumU[suBase + i1] * sumU[suBase + i2];
            }
            d3[d3base + nAB * ep] = acc;
          }
        }
      }
    }
  }
}

void EAPOD::threebodydescderiv(double *dd3, double *sumU, double *Ux, double *Uy, double *Uz,
                               int *tj, int N)
{
  int Me = nelements * (nelements + 1) / 2;
  memset(dd3, 0, 3 * N * nabf3_active * nrbf3 * Me * sizeof(*dd3));

  if (nelements == 1) {
    for (int m = 0; m < nrbf3; m++) {
      for (int a = 0; a < nabf3_active; a++) {
        int p  = p3_active[a];
        int n1 = pn3[p];
        int n2 = pn3[p + 1];
        int nn = n2 - n1;

        for (int q = 0; q < nn; q++) {
          int qq = n1 + q;
          double t1 = pc3[qq] * sumU[qq + K3 * m];
          for (int j = 0; j < N; j++) {
            double f  = 2.0 * t1;
            int ii = 3 * j + 3 * N * (a + nabf3_active * m);
            int jj = j + N * qq + N * K3 * m;
            dd3[0 + ii] += f * Ux[jj];
            dd3[1 + ii] += f * Uy[jj];
            dd3[2 + ii] += f * Uz[jj];
          }
        }
      }
    }
  } else {
    for (int m = 0; m < nrbf3; m++) {
      for (int a = 0; a < nabf3_active; a++) {
        int p  = p3_active[a];
        int n1 = pn3[p];
        int n2 = pn3[p + 1];
        int nn = n2 - n1;

        for (int q = 0; q < nn; q++) {
          int qq = n1 + q;
          for (int i1 = 0; i1 < nelements; i1++) {
            double t1 = pc3[qq] * sumU[i1 + nelements * qq + nelements * K3 * m];
            for (int j = 0; j < N; j++) {
              int i2 = tj[j];
              int k  = elemindex[i1][i2];
              double f = (i1 == i2) ? (2.0 * t1) : t1;
              int ii = 3 * j + 3 * N * (a + nabf3_active * m + nabf3_active * nrbf3 * k);
              int jj = j + N * qq + N * K3 * m;
              dd3[0 + ii] += f * Ux[jj];
              dd3[1 + ii] += f * Uy[jj];
              dd3[2 + ii] += f * Uz[jj];
            }
          }
        }
      }
    }
  }
}

void EAPOD::twobodydesc(double *d2, double *rbf, int *tj, int N, int Ne)
{
  if (Ne == 1) {
    for (int m = 0, mN = 0; m < nrbf2; ++m, mN += N) {
      double sum = 0.0;
      for (int n = 0, i2 = mN; n < N; ++n, ++i2)
        sum += rbf[i2];
      d2[m] = sum;
    }
  } else {
    for (int m=0; m<nl2; m++) d2[m] = 0.0;
    for (int n = 0; n < N; ++n) {
      int d2idx = nrbf2 * tj[n];
      int i2 = n;
      for (int m = 0; m < nrbf2; ++m, ++d2idx, i2 += N)
        d2[d2idx] += rbf[i2];
    }
  }
}

void EAPOD::twobodydescderiv(double *d2, double *dd2, double *rbf, double *drbf, double *rij, int *tj, int N)
{
  for (int m=0; m<nl2; m++) d2[m] = 0.0;
  for (int m=0; m<3*N*nl2; m++) dd2[m] = 0.0;

  // Calculate the two-body descriptors and their derivatives
  for (int m=0; m<nrbf2; m++) {
    for (int n=0; n<N; n++) {
      int i2 = n + N*m;
      int i1 = n + N*m + N*nrbf2*tj[n];
      double drbfi2 = drbf[i2];
      d2[m + nrbf2*tj[n]] += rbf[i2];
      dd2[0 + 3*i1] += drbfi2 * rij[0 + 3*n];
      dd2[1 + 3*i1] += drbfi2 * rij[1 + 3*n];
      dd2[2 + 3*i1] += drbfi2 * rij[2 + 3*n];
    }
  }
}

void EAPOD::init_bessel_const()
{
  bessel_const_storage.resize(nelements * nelements * nbesselpars);
  for (int it = 0; it < nelements; ++it) {
    for (int jt = 0; jt < nelements; ++jt) {
      double invrmax = invrdiff[it][jt];
      BesselConst *bc = &bessel_const_storage[(it*nelements + jt) * nbesselpars];
      for (int j = 0; j < nbesselpars; ++j) {
        double alpha = besselparams[j];
        double inv_t1 = MY_PI / expm1(-alpha);
        bc[j].neg_alpha = -alpha;
        bc[j].pi_inv_t1 = inv_t1;
        bc[j].dx_factor = -alpha * invrmax * inv_t1;
      }
    }
  }
}

inline void EAPOD::cutoff_exp(double r, double invrmax, double e_v,
                                  double &fcut, double &dfcut)
{
  double y  = r * invrmax;
  double y2 = y * y;
  double y3 = y * y2 - 1.0;
  double y4 = y3 * y3 + 1e-6;
  double y5 = sqrt(y4);
  fcut  = e_v * exp(-1.0 / y5);
  dfcut = fcut * 3.0 * invrmax * y2 * y3 / (y4 * y5);
}

inline void EAPOD::cutoff_poly_sq(double r, double invrmax,
                                  double &fcut, double &dfcut)
{
  double y = r * invrmax;
  if (y >= 1.0) { fcut = 0.0; dfcut = 0.0; return; }
  double dudy = 2.0 * y;

  double u   = y * y;
  double u2  = u * u;
  double u3  = u2 * u;
  double u4  = u2 * u2;
  double u5  = u4 * u;
  double omu = 1.0 - u;

  fcut  = 1.0 - 10.0 * u3 + 15.0 * u4 - 6.0 * u5;
  dfcut = -30.0 * invrmax * dudy * u2 * omu * omu;
}

inline void EAPOD::cutoff_hat(double r, double invrmax,
                                  double &fcut, double &dfcut)
{
  constexpr int p = 2;
  constexpr int q = 4;
  constexpr double TWO_PQ = -2.0 * p * q;

  double y = r * invrmax;

  double y2      = y * y;
  double y2_pm1  = powint(y2, p - 1);
  double u       = y2_pm1 * y2;
  double omu     = 1.0 - u;
  double omu_qm1 = powint(omu, q - 1);

  fcut  = omu_qm1 * omu;
  dfcut = TWO_PQ * invrmax * y * y2_pm1 * omu_qm1;
}

/**
 * @brief Calculates the radial basis functions and their derivatives.
 *
 * @param rbf           Pointer to the array of radial basis functions.
 * @param drbf          Pointer to the array of derivatives of radial basis functions with respect to r.
 * @param rij           Pointer to the relative positions of neighboring atoms and atom i.
 * @param rin           Minimum distance for radial basis functions.
 * @param rmax          Maximum distance for radial basis functions.
 * @param besseldegree  Degree of Bessel functions.
 * @param inversedegree Degree of inverse distance functions.
 * @param nbesselpars   Number of Bessel function parameters.
 * @param N             Number of neighboring atoms.
 */
void EAPOD::radialbasis(double *rbf, double *drbf, double *rij, double **rin, double **invrdiff,
                        int *ti, int *tj,
                        int besseldegree, int inversedegree, int nbesselpars, int N)
{
  const int itype = ti[0];

  for (int n = 0; n < N; n++) {
    int jtype = tj[n];
    const BesselConst *bc = bessel_const(itype, jtype);

    double xij = rij[3*n    ];
    double yij = rij[3*n + 1];
    double zij = rij[3*n + 2];
    double dij    = sqrt(xij*xij + yij*yij + zij*zij);
    double invdij = 1.0 / dij;

    double invrmax = invrdiff[itype][jtype];
    double r       = dij - rin[itype][jtype];
    double invr    = 1.0 / r;
    double y       = r * invrmax;

    // cutoff choice
    double fcut, dfcut;
    cutoff_exp(r, invrmax, e_v, fcut, dfcut);
    //cutoff_poly_sq(r, invrmax, fcut, dfcut);
    //cutoff_hat(r, invrmax, fcut, dfcut);

    // shared Bessel atom-constants
    double f1   = fcut * invr;
    double g1   = (dfcut - f1) * invr;
    double bfac = sqrt(2.0 * invrmax);
    double bf1  = bfac * f1;
    double bg1  = bfac * g1;

    int nij = n;

    // Bessel rbf
    for (int j = 0; j < nbesselpars; ++j) {
      double mt2   = expm1(bc[j].neg_alpha * y);
      double xpi   = mt2 * bc[j].pi_inv_t1;
      double ix    = xpi;
      double pi_dx = bc[j].dx_factor * (1.0 + mt2);
      double Kf1dx = bf1 * pi_dx;

      for (int i = 1; i <= besseldegree; ++i) {
        double cosax = cos(ix);
        double inv_i = 1.0 / (double)i;
        double isinax = inv_i * sin(ix);

        double rbfv   = bf1 * isinax;
        double drbfdr = bg1 * isinax + Kf1dx * cosax;

        rbf [nij] = rbfv;
        drbf[nij] = drbfdr * invdij;
        ix += xpi;
        nij += N;
      }
    }

    // Inverse-poly rbf
    double fcut_invd = fcut * invdij;
    double dterm = dfcut;

    double inva  = 1.0;
    for (int i = 1; i <= inversedegree; ++i) {
      dterm -= fcut_invd;
      inva  *= invdij;

      double rbfv   = fcut * inva;
      double drbfdr = dterm * inva;

      rbf [nij] = rbfv;
      drbf[nij] = drbfdr * invdij;
      nij += N;
    }
  }
}

// Orthogonalize rbf with Phi (eigenvectors)
// Apply Phi transformation for each atom pair
// rbf = Phi * rbft
void EAPOD::radialPhi(double *rbf, double *drbf,
                      double *rbft, double *drbft,
                      int *ti, int *tj, int N)
{
  const int ns2 = ns*ns;
  const int itypene = ti[0]*nelements;
  for (int n=0; n<N; n++) {
    int jtype = tj[n];
    int nsij = (jtype + itypene)*ns2;
    for (int k=0; k<nrbf2; k++) {
      double sum_rbf = 0.0;
      double sum_drbf = 0.0;

      for (int j=0; j<ns; j++) {
        double Phis = Phi[j + k*ns + nsij];
        sum_rbf  += Phis * rbft[n + N*j];
        sum_drbf += Phis * drbft[n + N*j];
      }

      int nij = n + N * k;
      rbf[nij] = sum_rbf;
      drbf[nij] = sum_drbf;
    }
  }
}

/* ----------------------------------------------------------------------
   Precompute C2 cubic-spline tables of the (Phi-orthogonalized)
   radial basis f_k(r) for every element pair.

   This builds a *clamped cubic spline* per (pair,k):
     - interpolates f_k at all nodes
     - enforces endpoint slopes from analytic df/dr (drbf)
     - is C2 across all interior knots
------------------------------------------------------------------------- */
void EAPOD::init_spline_radialbasis()
{
  if (nspline_grid < 4) nspline_grid = 4;

  const int ne = nelements;
  const int Ng = nspline_grid;   // nodes
  const int nb = Ng - 1;         // bins
  nspline_bins = nb;

  memory->create(spline_r0,    ne*ne,                  "spline_r0");
  memory->create(spline_invdr, ne*ne,                  "spline_invdr");
  memory->create(rbf_spline_coeffs, ne*ne*nb*nrbf2*4, "rbf_spline_coeffs");

  // scratch buffers for basis sampling
  double *rij, *rbft, *drbft;
  double *rbf, *drbf;
  int *tit, *tjt;
  memory->create(rij,   3*Ng,       "spl:rij");
  memory->create(rbft,  Ng*ns,      "spl:rbft");
  memory->create(drbft, Ng*ns,      "spl:drbft");
  memory->create(rbf,   Ng*nrbf2, "spl:rbf");
  memory->create(drbf,  Ng*nrbf2, "spl:drbf");
  memory->create(tit,   1,          "spl:tit");
  memory->create(tjt,   Ng,         "spl:tjt");

  // scratch for tridiagonal solve (size Ng), reused for each k
  double *y, *M, *dl, *dd, *du, *rhs;
  memory->create(y,   Ng, "spl:y");
  memory->create(M,   Ng, "spl:M");
  memory->create(dl,  Ng, "spl:dl");
  memory->create(dd,  Ng, "spl:dd");
  memory->create(du,  Ng, "spl:du");
  memory->create(rhs, Ng, "spl:rhs");

  for (int it = 0; it < ne; ++it) {
    for (int jt = 0; jt < ne; ++jt) {
      const int pair = it*ne + jt;

      // grid spans [rin+eps, rcut]
      const double r0    = rin[it][jt] + 1e-6;
      const double r1    = rcut[it][jt];
      const double dr    = (r1 - r0) / nb;
      const double invdr = 1.0 / dr;
      const double dr2   = dr * dr;

      spline_r0[pair]    = r0;
      spline_invdr[pair] = invdr;

      tit[0] = it;
      for (int i = 0; i < Ng; ++i) {
        const double r = r0 + i*dr;
        rij[3*i+0] = r;
        rij[3*i+1] = 0.0;
        rij[3*i+2] = 0.0;
        tjt[i] = jt;
      }

      // sample analytical rbf and df/dr at nodes
      radialbasis(rbft, drbft, rij, rin, invrdiff,
                  tit, tjt, besseldegree, inversedegree, nbesselpars, Ng);
      radialPhi(rbf, drbf, rbft, drbft, tit, tjt, Ng);

      // Build one clamped C2 cubic spline per basis channel k
      for (int k = 0; k < nrbf2; ++k) {
        // y[i] = f(r_i)
        for (int i = 0; i < Ng; ++i) y[i] = rbf[i + Ng*k];

        // endpoint slopes m0,mN are df/dr from drbf
        const double m0 = drbf[0  + Ng*k];
        const double mN = drbf[nb + Ng*k];

        // Tridiagonal system for node second derivatives M[i] = f''(r_i)
        // Uniform-grid clamped cubic spline:
        dl[0] = 0.0; dd[0] = 2.0; du[0] = 1.0;
        rhs[0] = (6.0*invdr) * ((y[1] - y[0])*invdr - m0);

        for (int i = 1; i < nb; ++i) {
          dl[i] = 1.0; dd[i] = 4.0; du[i] = 1.0;
          rhs[i] = (6.0/dr2) * (y[i+1] - 2.0*y[i] + y[i-1]);
        }

        dl[nb] = 1.0; dd[nb] = 2.0; du[nb] = 0.0;
        rhs[nb] = (6.0*invdr) * (mN - (y[nb] - y[nb - 1])*invdr);

        // Thomas algorithm
        for (int i = 1; i < Ng; ++i) {
          const double w = dl[i] / dd[i - 1];
          dd[i]  -= w * du[i - 1];
          rhs[i] -= w * rhs[i - 1];
        }
        M[nb] = rhs[nb] / dd[nb];
        for (int i = nb - 1; i >= 0; --i) {
          M[i] = (rhs[i] - du[i] * M[i + 1]) / dd[i];
        }

        // Convert to local-t cubic coefficients per bin:
        // f(t)=c0 + c1 t + c2 t^2 + c3 t^3, t=(r-r_i)/dr
        for (int bin = 0; bin < nb; ++bin) {
          const double f0 = y[bin];
          const double f1 = y[bin + 1];
          const double Mi = M[bin];
          const double Mj = M[bin + 1];

          double *c = &rbf_spline_coeffs[((pair*nb + bin)*nrbf2 + k)*4];
          c[0] = f0;
          c[1] = (f1 - f0) - (dr2/6.0) * (2.0*Mi + Mj);
          c[2] = 0.5 * dr2 * Mi;
          c[3] = (dr2/6.0) * (Mj - Mi);
        }
      }
    }
  }

  memory->destroy(rij);
  memory->destroy(rbft);  memory->destroy(drbft);
  memory->destroy(rbf);   memory->destroy(drbf);
  memory->destroy(tit);   memory->destroy(tjt);

  memory->destroy(y);   memory->destroy(M);
  memory->destroy(dl);  memory->destroy(dd);
  memory->destroy(du);  memory->destroy(rhs);
}

void EAPOD::radialbasis_spline(double *rbf, double *drbf, double *rij, int *ti, int *tj, int N)
{
  const int ne = nelements;
  const int nb = nspline_bins;
  const int ncs = 4*nrbf2;
  const int itypene = ti[0]*ne;

  for (int n = 0; n < N; ++n) {
    int pair = itypene + tj[n];

    double x = rij[3*n+0];
    double y = rij[3*n+1];
    double z = rij[3*n+2];
    double dij = sqrt(x*x + y*y + z*z);
    //double invdij = 1.0/dij;

    double r0    = spline_r0[pair];
    double invdr = spline_invdr[pair];
    double invdrinvdij = invdr/dij;

    double tg = (dij - r0)*invdr;     // global "bin coordinate"
    int b = (int) tg;
    if (b < 0) b = 0;
    else if (b > nb-1) b = nb-1;
    double t = tg - (double) b;       // local coordinate in [0,1] (clamped ends extrapolate)

    const double *cbase = &rbf_spline_coeffs[(pair*nb + b)*ncs];

    for (int k = 0; k < nrbf2; ++k) {
      const double *c = cbase + 4*k;
      double c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];

      double f    = c0 + t*(c1 + t*(c2 + t*c3));
      double dfdt = c1 + t*(2.0*c2 + 3.0*c3*t);
      double h    = dfdt*invdrinvdij;   // = f'(r)/r

      int idx = n + N*k;
      rbf [idx] = f;
      drbf[idx] = h;
    }
  }
}

/**
 * @brief Calculates the angular basis functions and their derivatives.
 *
 * @param abf   Pointer to the angular basis functions.
 * @param abfx  Pointer to the derivative of the angular basis functions w.r.t. x.
 * @param abfy  Pointer to the derivative of the angular basis functions w.r.t. y.
 * @param abfz  Pointer to the derivative of the angular basis functions w.r.t. z.
 * @param rij   Pointer to the relative positions of neighboring atoms and atom i.
 * @param tm    Pointer to temporary array.
 * @param pq    Pointer to array of indices for angular basis functions.
 * @param N     Number of neighboring atoms.
 * @param K     Number of angular basis functions.
 */
void EAPOD::angularbasis(double *abf,double *abfx, double *abfy,double *abfz, double *rij, double *tm, int *pq, int N, int K)
{
  // Scratch: value + x/y/z derivatives for current j
  double *tmx = &tm[K];
  double *tmy = &tm[2*K];
  double *tmz = &tm[3*K];

  tm[0] = 1.0;  tmx[0] = 0.0;  tmy[0] = 0.0;  tmz[0] = 0.0;

  for (int j = 0; j < N; ++j) {
    const double x = rij[3*j + 0];
    const double y = rij[3*j + 1];
    const double z = rij[3*j + 2];

    const double xx = x*x, yy = y*y, zz = z*z;
    const double xy = x*y, xz = x*z, yz = y*z;

    const double invdij  = 1.0 / sqrt(xx + yy + zz);
    const double u = x * invdij;
    const double v = y * invdij;
    const double w = z * invdij;

    const double invdij3 = invdij * invdij * invdij;

    const double dudx = (yy + zz) * invdij3;
    const double dvdy = (xx + zz) * invdij3;
    const double dwdz = (xx + yy) * invdij3;

    const double dudy = -xy * invdij3;
    const double dudz = -xz * invdij3;
    const double dvdz = -yz * invdij3;
    
    const double dvdx = dudy;
    const double dwdx = dudz;
    const double dwdy = dvdz;

    // n = 0
    int out = j;
    abf[out]  = 1.0;
    abfx[out] = 0.0;
    abfy[out] = 0.0;
    abfz[out] = 0.0;

    // n = 1..K-1
    double c, dcx, dcy, dcz;
    for (int n = 1; n < K; ++n) {
      out += N;
      const int m = pq[n];
      const int d = pq[n + K];

      const double t0  = tm[m];
      const double tx0 = tmx[m];
      const double ty0 = tmy[m];
      const double tz0 = tmz[m];

      if (d == 1) {
        c = u; dcx = dudx; dcy = dudy; dcz = dudz;
      } else if (d == 2) {
        c = v; dcx = dvdx; dcy = dvdy; dcz = dvdz;
      } else { // d == 3
        c = w; dcx = dwdx; dcy = dwdy; dcz = dwdz;
      }

      const double t  = t0 * c;
      const double tx = tx0 * c + t0 * dcx;
      const double ty = ty0 * c + t0 * dcy;
      const double tz = tz0 * c + t0 * dcz;

      tm[n]  = t;
      tmx[n] = tx;
      tmy[n] = ty;
      tmz[n] = tz;

      abf[out]  = t;
      abfx[out] = tx;
      abfy[out] = ty;
      abfz[out] = tz;
    }
  }
}

void EAPOD::radialangularsum(double *sumU, double *rbf, double *abf, int *tj,
                             int Nj, int K, int M, int Ne)
{
  if (Ne == 1) {
    for (int m = 0; m < M; ++m) {
      const double *r = &rbf[Nj*m];
      double *su = &sumU[K*m];
      for (int k = 0; k < K; ++k) {
        const double *a = &abf[Nj*k];
        double sum = 0.0;
        for (int n = 0; n < Nj; ++n)
          sum += r[n] * a[n];
        su[k] = sum;
      }
    }
  } else {
    const int NeK = Ne*K;
    memset(sumU, 0,  NeK*M * sizeof(*sumU));
    for (int n = 0; n < Nj; ++n) {
      const int e = tj[n];
      const double *a = &abf[n];
      for (int m = 0; m < M; ++m) {
        const double c = rbf[n + Nj*m];
        double *su = &sumU[e + NeK*m];
        for (int k = 0; k < K; ++k) su[Ne*k] += c * a[Nj*k];
      }
    }
  }
}

/**
 * @brief Calculates the radial-angular basis functions and their derivatives.
 *
 * @param sumU  Pointer to the array to store the sum of the basis functions.
 * @param Ux    Pointer to the array to store the derivative of U with respect to x.
 * @param Uy    Pointer to the array to store the derivative of U with respect to y.
 * @param Uz    Pointer to the array to store the derivative of U with respect to z.
 * @param rbf   Pointer to the radial basis function array.
 * @param drbf  Pointer to the derivative of rbf with respect to x.
 * @param abf   Pointer to the angular basis function array.
 * @param abfx  Pointer to the derivative of abf with respect to x.
 * @param abfy  Pointer to the derivative of abf with respect to y.
 * @param abfz  Pointer to the derivative of abf with respect to z.
 * @param tm    Pointer to the temporary memory array .
 * @param atomtype  Pointer to the array of atom types.
 * @param N     Number of neighboring atoms.
 * @param K     Number of angular basis functions.
 * @param M     Number of radial basis functions.
 * @param Ne    Number of elements.
 */
void EAPOD::radialangularbasis(double *sumU, double *Ux, double *Uy, double *Uz,
                               double *rbf, double *drbf, double *rij,
                               double *abf, double *abfx, double *abfy, double *abfz,
                               double *tm, int *tj, int N, int K, int M, int Ne)
{
  const int NK  = N * K;
  const int NeK = Ne * K;

  if (Ne == 1) {
    for (int m = 0, mN = 0, mNK = 0, sumM = 0; m < M; ++m, mN += N, mNK += NK, sumM += K) {
      for (int k = 0, kN = 0, sumIdx = sumM; k < K; ++k, kN += N, ++sumIdx) {
        int ia = kN;
        int ib = mN;
        int ii = mNK + kN;
        double sum = 0.0;
        for (int n = 0; n < N; ++n, ++ia, ++ib, ++ii) {
          double c1 = rbf[ib];
          double c2 = abf[ia];
          double c3 = drbf[ib];
          Ux[ii] = abfx[ia] * c1 + c2 * c3 * rij[3*n];
          Uy[ii] = abfy[ia] * c1 + c2 * c3 * rij[3*n+1];
          Uz[ii] = abfz[ia] * c1 + c2 * c3 * rij[3*n+2];
          sum += c1 * c2;
        }
        sumU[sumIdx] = sum;
      }
    }
  } else {
    for (int m = 0, mN = 0, mNK = 0, sumM = 0; m < M; ++m, mN += N, mNK += NK, sumM += NeK) {
      for (int k = 0, kN = 0, sumBase = sumM; k < K; ++k, kN += N, sumBase += Ne) {
        // zero per-element accumulator
        for (int e = 0; e < Ne; ++e) tm[e] = 0.0;
        int ia = kN;
        int ib = mN;
        int ii = mNK + kN;
        for (int n = 0; n < N; ++n, ++ia, ++ib, ++ii) {
          double c1 = rbf[ib];
          double c2 = abf[ia];
          double c3 = drbf[ib];
          Ux[ii] = abfx[ia] * c1 + c2 * c3 * rij[3*n];
          Uy[ii] = abfy[ia] * c1 + c2 * c3 * rij[3*n+1];
          Uz[ii] = abfz[ia] * c1 + c2 * c3 * rij[3*n+2];
          tm[tj[n]] += c1 * c2;  // accumulate per element type
        }
        // contiguous writeback
        for (int e = 0; e < Ne; ++e) sumU[sumBase + e] = tm[e];
      }
    }
  }
}

/**
 * @brief Tally the force on each atom i and its neighboring atoms.
 *
 * @param force Pointer to the output array for the global force
 * @param fij Pointer to the array of forces between each pair of neighboring atoms.
 * @param ai Pointer to the array of atom indices for each pair of neighboring atoms.
 * @param aj Pointer to the array of neighboring atom indices for each pair of neighboring atoms.
 * @param N Number of neighboring atom pairs.
 */
void EAPOD::tallyforce(double *force, double *fij,  int *ai, int *aj, int N)
{
  // Loop over all neighboring atoms
  for (int n=0; n<N; n++) {
    int im =  3*ai[n];
    int jm =  3*aj[n];
    int nm = 3*n;
    force[0 + im] += fij[0 + nm];
    force[1 + im] += fij[1 + nm];
    force[2 + im] += fij[2 + nm];
    force[0 + jm] -= fij[0 + nm];
    force[1 + jm] -= fij[1 + nm];
    force[2 + jm] -= fij[2 + nm];
  }
}

/**
 * @brief Create new coefficients for the local descriptors.
 *
 * @param c Pointer to the input array of original coefficients for the global descriptors.
 */
void EAPOD::mknewcoeff(double *c, int nc)
{
  // Allocate memory for the new coefficients
  memory->create(coeff, nc, "coeff");

  // Copy the  coefficients
  for (int n=0; n<nc; n++)
    coeff[n] = c[n];
}

/**
 * @brief Initialize the two-body coefficients for all element pairs.
 *
 * @param None
 */
void EAPOD::init2body()
{
  // Allocate memory for the eigenvectors and eigenvalues for all element pairs
  memory->create(Phi, ns * ns * nelements * nelements, "Phi");
  memory->create(Lambda, ns * nelements * nelements, "Lambda");

  // Perform eigenvalue decomposition for each element pair
  int NsnElms = 2000;
  for (int i = 0; i < nelements; i++) {
    for (int j = 0; j < nelements; j++) {
      int ijpair = j + i * nelements;
      double rinij = rin[i][j];
      double rdiffij = rcut[i][j] - rinij;
      eigenvaluedecomposition(&Phi[ijpair*ns*ns], &Lambda[ijpair*ns], rinij, rdiffij, NsnElms);
    }
  }

  memory->destroy(Lambda);
}

/**
 * @brief Perform eigenvalue decomposition of the snapshots matrix S and return the eigenvectors and eigenvalues.
 *
 * @param Phi Pointer to the output array for the eigenvectors.
 * @param Lambda Pointer to the output array for the eigenvalues.
 * @param N Number of points in the interval [rin, rcut].
 */
void EAPOD::eigenvaluedecomposition(double *Phi, double *Lambda, double rinij, double rdiffij, int N)
{
  double *xij;
  double *S;
  double *Q;
  double *A;
  double *work;
  double *b;

  memory->create(xij, N, "eapod:xij");
  memory->create(S, N*ns, "eapod:S");
  memory->create(Q, N*ns, "eapod:Q");
  memory->create(A, ns*ns, "eapod:A");
  memory->create(work, ns*ns, "eapod:work");
  memory->create(b, ns, "eapod:ns");

  // Generate the xij array
  for (int i=0; i<N; i++)
    xij[i] = (rinij+1e-6) + (rdiffij-1e-6)*(i*1.0/(N-1));
  
  // Compute the snapshots matrix S
  snapshots(S, xij, rinij, rdiffij, N);

  // Compute the matrix A = S^T * S
  char chn = 'N';
  char cht = 'T';
  double alpha = 1.0, beta = 0.0;
  DGEMM(&cht, &chn, &ns, &ns, &N, &alpha, S, &N, S, &N, &beta, A, &ns);

  // Normalize the matrix A by dividing by N
  for (int i=0; i<ns*ns; i++)
    A[i] = A[i]*(1.0/N);

  // Compute the eigenvectors and eigenvalues of A
  int lwork = ns * ns;  // the length of the array work, lwork >= max(1,3*N-1)
  int info = 1;     // = 0:  successful exit
  //double work[ns*ns];
  char chv = 'V';
  char chu = 'U';
  DSYEV(&chv, &chu, &ns, A, &ns, b, work, &lwork, &info);

  // Order eigenvalues and eigenvectors from largest to smallest
  for (int j=0; j<ns; j++)
    for (int i=0; i<ns; i++)
      Phi[i + ns*(ns-j-1)] = A[i + ns*j];

  for (int i=0; i<ns; i++)
    Lambda[(ns-i-1)] = b[i];

  // Compute the matrix Q = S * Phi
  DGEMM(&chn, &chn, &N, &ns, &ns, &alpha, S, &N, Phi, &ns, &beta, Q, &N);

  // Compute the area of each snapshot and normalize the eigenvectors
  for (int i=0; i<(N-1); i++)
    xij[i] = xij[i+1] - xij[i];
  double area;
  for (int m=0; m<ns; m++) {
    area = 0.0;
    for (int i=0; i<(N-1); i++)
      area += 0.5*xij[i]*(Q[i + N*m]*Q[i + N*m] + Q[i+1 + N*m]*Q[i+1 + N*m]);
    for (int i=0; i<ns; i++)
      Phi[i + ns*m] = Phi[i + ns*m]/sqrt(area);
  }

  // Enforce consistent signs for the eigenvectors
  for (int m=0; m<ns; m++) {
    if (Phi[m + ns*m] < 0.0) {
      for (int i=0; i<ns; i++)
        Phi[i + ns*m] = -Phi[i + ns*m];
    }
  }

  // Free temporary arrays
  memory->destroy(xij);
  memory->destroy(S);
  memory->destroy(A);
  memory->destroy(work);
  memory->destroy(b);
  memory->destroy(Q);
}

/**
 * @brief Compute the radial basis function (RBF) for each atom.
 *
 * @param rbf Pointer to the output array for the RBF.
 * @param xij Pointer to the array of distances between each pair of atoms.
 * @param N Number of points in the interval [rin, rcut].
 */
void EAPOD::snapshots(double *rbf, double *xij, double rinij, double rdiffij, int N)
{
  const double invrmax = 1.0 / rdiffij;
  const double bfac = sqrt(2.0 * invrmax);

  for (int n = 0; n < N; ++n) {
    double dij    = xij[n];
    double invdij = 1.0 / dij;
    double r      = dij - rinij;
    double invr   = 1.0 / r;
    double y      = r * invrmax;

    // cutoff choice
    double fcut, dfcut;
    cutoff_exp(r, invrmax, e_v, fcut, dfcut);
    //cutoff_poly_sq(r, invrmax, fcut, dfcut);
    //cutoff_hat(r, invrmax, fcut, dfcut);

    double bf1 = bfac * fcut * invr;

    int nij = n;

    // Bessel rbf
    for (int j = 0; j < nbesselpars; ++j) {
      double alpha   = besselparams[j];
      double mt2_t1  = expm1(-alpha * y) / expm1(-alpha);
      double xpi     = MY_PI * mt2_t1;
      double ix      = xpi;

      for (int i = 1; i <= besseldegree; ++i) {
        double bf1i = bf1 / i;
        rbf[nij] = bf1i * sin(ix);
        ix += xpi;
        nij += N;
      }
    }

    // Inverse-poly rbf
    double inva = fcut;
    for (int i = 1; i <= inversedegree; ++i) {
      inva *= invdij;
      rbf[nij] = inva;
      nij += N;
    }
  }
}

/* ----------------------------------------------------------------------
   Per-atom active-learning uncertainty quantities
   out[0] = full atomic energy (one-body + blended many-body)  [eV]    (diagnostic)
   out[1] = P-weighted committee std of E_k                    [eV]    (Method B, primary)
   out[2] = uniform committee std over active clusters         [eV]    (Method B, variant)
   out[3] = P-weighted committee std of F_k                    [eV/A]  (Method B, primary)
   out[4] = uniform committee std over active clusters         [eV/A]  (Method B, variant)
   out[5] = S = sum_k D_k                                      (Method A: U = 1/(S+eps))
   out[6] = population-weighted density uncertainty 1/(sum_k D_k*n_k)   (Method C)
   out[7] = pca (first principal component of the base descriptor)     (diagnostic)
   out[8] = number of active clusters (double)                         (diagnostic)
   'tm' scratch must hold >= 2*nClusters + nComponents doubles.
------------------------------------------------------------------------- */
void EAPOD::peratom_uncertainty(double *out, double *bd, double *bdd, int Nj, double *tm, int *ti)
{
  for (int i = 0; i < 9; i++) out[i] = 0.0;

  int typei = ti[0];
  int nc   = nCoeffPerElement*typei;
  int nct  = nComponents*typei;
  int ncct = nClusters*nct;
  double *proj = &Proj[Mdesc*nct];
  double *invspan = &invPcaSpan[nct];

  // reduce bdd (N x Mdesc, column-major, N=3*Nj) to net force on atom i:
  // G[d + 3*m] = sum_p bdd[(3*p+d) + N*m]   (sum over neighbor pairs p)
  int N = 3 * Nj;
  double *G  = &tm[2 * nClusters + nComponents];        // 3*Mdesc
  double *Fk = &tm[2 * nClusters + nComponents + 3 * Mdesc]; // 3*nClusters
  for (int m = 0; m < Mdesc; m++) {
    double gx = 0.0, gy = 0.0, gz = 0.0;
    double *col = &bdd[N * m];
    for (int p = 0; p < Nj; p++) { gx += col[3*p]; gy += col[3*p+1]; gz += col[3*p+2]; }
    G[0 + 3*m] = gx; G[1 + 3*m] = gy; G[2 + 3*m] = gz;
  }

  double *D  = &tm[0];           // nClusters
  double *Ek = &tm[nClusters];   // nClusters
  double *pca = &tm[2*nClusters];   // nComponents

  double S_density = 0.0;
  double S_pop = 0.0;    // occupancy-weighted density: sum_k D_k * n_k (training population)
  double U = 0.0;
  double U_pop = 0.0;
  double Ebar = 0.0;

  int ks = 0;
  int ke = nClusters;
  int kn = ke - ks;

  for (int c = 0; c < nComponents; c++) {
    double s = 0.0;
    for (int m = 0; m < Mdesc; m++) s += proj[c + nComponents * m] * bd[m];
    pca[c] = s;
  }
  double pca0 = pca[0];

  if (localeapod) { // local EA-POD
    double *ledges = &leftClusterEdges[ncct];
    double *redges = &rightClusterEdges[ncct];

    find_active_clusters(pca0, ledges, redges, nClusters, nMaxActiveClusters, ks, ke);
    kn = ke - ks;
    ncct += ks;

    double *cent     = &Centroids[ncct];
    double *invlcut2 = &invLeftClusterRcut2[ncct];
    double *invrcut2 = &invRightClusterRcut2[ncct];
    double invspan0 = invspan[0];
    int *occ = (clusterOccupancy != nullptr) ? &clusterOccupancy[ncct] : nullptr;

    double fcut, dfcut;
    for (int k = 0; k < kn; k++) {
      double pc = pca0 - cent[k];
      double inv_rcut2 = (pc >= 0.0) ? invrcut2[k] : invlcut2[k];
      cluster_cutoff_hat(pc, inv_rcut2, fcut, dfcut);
      double pcn = pc * invspan0;
      double Dk = fcut / (pcn * pcn + 1e-20);
      D[k] = Dk;
      S_density += Dk;
      if (occ != nullptr) S_pop += Dk * occ[k];
    }
  } else { // EA-POD
    double *cent = &Centroids[ncct];
    int *occ = (clusterOccupancy != nullptr) ? &clusterOccupancy[nClusters * typei] : nullptr;
    for (int k = 0; k < kn; k++) {
      double s = 0.0;
      for (int c = 0; c < nComponents; c++) {
        double d = (pca[c] - cent[c + k * nComponents]) * invspan[c];
        s += d * d;
      }
      double Dk = 1.0 / (s + 1e-20);
      D[k] = Dk;
      S_density += Dk;
      if (occ != nullptr) S_pop += Dk * occ[k];
    }
  }
  U = (S_density > 0.0) ? 1.0 / S_density : 0.0;
  U_pop = (S_pop > 0.0) ? 1.0 / S_pop : U;

  double *ceffs = &coeff[1 + ks * Mdesc + nc];

  for (int k = 0; k < kn; k++) {
    double e = 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int m = 0; m < Mdesc; m++) {
      double c = ceffs[m + k * Mdesc];
      e  += c * bd[m];
      fx += c * G[0 + 3*m];
      fy += c * G[1 + 3*m];
      fz += c * G[2 + 3*m];
    }
    Ek[k] = e;
    Fk[3*k] = fx; Fk[3*k+1] = fy; Fk[3*k+2] = fz;
    Ebar += (D[k] * U) * e;
  }

  // committee variances (weighted and uniform over the active window)
  // Energy:
  double varP = 0.0, meanU = 0.0;
  for (int k = 0; k < kn; k++) {
    double Pk = D[k] * U;
    double Ekk = Ek[k];
    meanU += Ekk;
    varP  += Pk * (Ekk - Ebar) * (Ekk - Ebar);
  }
  meanU /= kn;
  double varU = 0.0;
  for (int k = 0; k < kn; k++) varU += (Ek[k] - meanU) * (Ek[k] - meanU);
  varU /= kn;

  // Force (RMS vector deviation across clusters)
  double Fbar[3] = {0.0, 0.0, 0.0};
  for (int k = 0; k < kn; k++) {
    double Pk = D[k] * U;
    Fbar[0] += Pk * Fk[3*k];
    Fbar[1] += Pk * Fk[3*k+1];
    Fbar[2] += Pk * Fk[3*k+2];
  }
  double FmeanU[3] = {0.0, 0.0, 0.0};
  for (int k = 0; k < kn; k++) {
    FmeanU[0] += Fk[3*k];
    FmeanU[1] += Fk[3*k+1];
    FmeanU[2] += Fk[3*k+2];
  }
  FmeanU[0] /= kn;
  FmeanU[1] /= kn;
  FmeanU[2] /= kn;

  double varFP = 0.0, varFU = 0.0;
  for (int k = 0; k < kn; k++) {
    double Pk = D[k] * U;
    double dx = Fk[3*k]-Fbar[0], dy = Fk[3*k+1]-Fbar[1], dz = Fk[3*k+2]-Fbar[2];
    varFP += Pk * (dx*dx + dy*dy + dz*dz);
    double ux = Fk[3*k]-FmeanU[0], uy = Fk[3*k+1]-FmeanU[1], uz = Fk[3*k+2]-FmeanU[2];
    varFU += (ux*ux + uy*uy + uz*uz);
  }
  varFU /= kn;

  out[0] = coeff[nc] + Ebar;
  out[1] = sqrt(varP);
  out[2] = sqrt(varU);
  out[3] = sqrt(varFP);
  out[4] = sqrt(varFU);
  out[5] = U;
  out[6] = U_pop;
  out[7] = pca0;
  out[8] = (double) kn;
}

/**
 * @brief Estimate the amount of memory needed for the computation.
 *
 * @param Nj Number of neighboring atoms.
 * @return int The estimated amount of memory needed.
 */
int EAPOD::estimate_temp_memory(int Nj)
{
  // Maximum number of >=5-body cross descriptors: d33, d34, d44
  int nld = MAX(MAX(nl33, nl34), nl44);

  // d2, d3, d4, nld
  int nmaxnl = nl2 + nl3 + nl4 + nld;

  // dd2, dd3, dd4 (training only)
  int nmaxnd = 3*Nj*nmaxnl;

  // rij, fij, d2, d3, d4, dd2, dd3, dd4
  int nmax1 = 6*Nj + nmaxnl + nmaxnd;

  // Ux, Uy, Uz (training only) K3>=K4, nrbf3>=nrbf4
  int nmax2 = 3*Nj*K3*nrbf3;

  // sumU and cU
  int nmax3 = 2*nelements*K3*nrbf3;

  // rbf, drbf
  int nmax4 = 2*Nj*nrbf2;

  // rbft, drbft
  int nmax5 = 2*Nj*ns;

  // abf, abfx, abfy, abfz
  int nmax6 = 4*(Nj+1)*K3;

  // Determine the total amount of memory needed for all double memory
  ndblmem = nmax1 + nmax2 + nmax3 + nmax4 + MAX(nmax5, nmax6);

  int eatmpmem = nComponents;
  if (localeapod) eatmpmem += nMaxActiveClusters*(4*nComponents + nMaxActiveClusters + 2*Mdesc);
  else if (eapod) eatmpmem += nClusters*(1 + nComponents + nClusters + 2*Mdesc);

  int nmax9 = 6*Nj + eatmpmem;
  if (ndblmem < nmax9) ndblmem = nmax9;

  // Determine the total amount of memory needed for all integer memory
  nintmem = 4*Nj;

  // Return the estimated amount of memory needed
  return ndblmem;
}

void EAPOD::allocate_temp_memory(int Nj)
{
  // guarantee all buffers exist even for atoms without neighbors
  if (Nj < 1) Nj = 1;
  estimate_temp_memory(Nj);
  // in peratomenergyforce2() the bdd buffer stores the coefficients cb and the
  // force coefficients, which require (nl2 + nl3 + nl4) + nelements*K3*nrbf3 entries.
  // this size is set by the potential and does not depend on the
  // number of neighbors, so it can exceed 3*Nj*Mdesc when Nj is small.
  int nbdd = 3*Nj*Mdesc;
  int ncb = (nl2 + nl3 + nl4) + nelements*K3*nrbf3;
  if (nbdd < ncb) nbdd = ncb;
  memory->create(tmpmem, ndblmem, "tmpmem");
  memory->create(tmpint, nintmem, "tmpint");
  memory->create(bd, Mdesc, "bd");
  memory->create(bdd, nbdd, "bdd");
  memory->create(pd, nClusters, "pd");
  memory->create(pdd, 3*Nj*nClusters, "pdd");
}

void EAPOD::free_temp_memory()
{
  memory->destroy(tmpmem);
  memory->destroy(tmpint);
  memory->destroy(bd);
  memory->destroy(bdd);
  memory->destroy(pd);
  memory->destroy(pdd);
}

/**
 * @brief Calculate the number of cross descriptors between two sets of descriptors.
 *
 * @return int The number of cross descriptors between two sets of descriptors.
 */
int EAPOD::crossindices(int *dabf1, int nabf1, int nrbf1, int nebf1,
         int *dabf2, int nabf2, int nrbf2, int nebf2, int dabf12, int nrbf12)
{
  int n = 0;

  // Loop over the first set of descriptors
  for (int i1=0; i1<nebf1; i1++)
    for (int j1=0; j1<nrbf1; j1++)
      for (int k1=0; k1<nabf1; k1++) {
        int m1 = k1 + j1*nabf1;
        int a1 = dabf1[k1];
        // Loop over the second set of descriptors
        for (int i2=0; i2<nebf2; i2++)
          for (int j2=0; j2<nrbf2; j2++)
            for (int k2=0; k2<nabf2; k2++) {
              int m2 = k2 + j2*nabf2;
              int a2 = dabf2[k2];
              // Check if the sum of the angular degrees is less than or equal to dabf12,
              // the number of radial basis functions is less than nrbf12, and the indices are in the correct order
              if ((m2 >= m1) && (i2 >= i1) && (a1 + a2 <= dabf12) && (j1+j2 < nrbf12)) {
                n += 1;
              }
            }
      }

  return n;
}

/**
 * @brief Calculate the number of cross descriptors between two sets of descriptors and store the indices in two arrays.
 *
 * @return int The number of cross descriptors between two sets of descriptors.
 */
int EAPOD::crossindices(int *ind1, int *ind2, int *dabf1, int nabf1, int nrbf1, int nebf1,
         int *dabf2, int nabf2, int nrbf2, int nebf2, int dabf12, int nrbf12)
{
  int n = 0;

  // Loop over the first set of descriptors
  for (int i1=0; i1<nebf1; i1++)
    for (int j1=0; j1<nrbf1; j1++)
      for (int k1=0; k1<nabf1; k1++) {
        int m1 = k1 + j1*nabf1;
        int n1 = m1 + i1*nabf1*nrbf1;
        int a1 = dabf1[k1];
        // Loop over the second set of descriptors
        for (int i2=0; i2<nebf2; i2++)
          for (int j2=0; j2<nrbf2; j2++)
            for (int k2=0; k2<nabf2; k2++) {
              int m2 = k2 + j2*nabf2;
              int n2 = m2 + i2*nabf2*nrbf2;
              int a2 = dabf2[k2];
              // Check if the sum of the angular degrees is less than or equal to dabf12,
              // the number of radial basis functions is less than nrbf12, and the indices are in the correct order
              if ((m2 >= m1) && (i2 >= i1) && (a1 + a2 <= dabf12) && (j1+j2 < nrbf12)) {
                ind1[n] = n1;
                ind2[n] = n2;
                n += 1;
              }
            }
      }

  return n;
}

void EAPOD::calculateClusterEdges(int nClusters, double nActiveClusters, int nComponents, int nelements) 
{  
  double lrange = 0.5 * nActiveClusters;
  int l = lrange; // number of clusters to consider on each side
  double fac = lrange - l;

  for (int elem = 0; elem < nelements; elem++) {
    double *centroids = &Centroids[nComponents*nClusters*elem];
    double *ledges = &leftClusterEdges[nComponents*nClusters*elem];
    double *redges = &rightClusterEdges[nComponents*nClusters*elem];
    double *invlc2 = &invLeftClusterRcut2[nComponents*nClusters*elem];
    double *invrc2 = &invRightClusterRcut2[nComponents*nClusters*elem];

    std::vector<double> dist(nClusters-1);
    for (int k = 0; k < nClusters-1; k++) {
        dist[k] = centroids[k+1] - centroids[k];
    }
    // Calculate left edge positions of clusters
    for (int k = 1; k < nClusters; k++) {
      double lsum = 0.0;
      for (int j = 0; j < l && k - j > 0; j++) {
        lsum += dist[k - j - 1];
      }
      double lrcut = lsum + fac * dist[(k - l - 1) > 0 ? (k - l - 1) : 0];
      ledges[k] = centroids[k] - lrcut;
      invlc2[k] = 1.0 / (lrcut * lrcut);
    }
    // Calculate right edge positions of clusters
    for (int k = 0; k < nClusters-1; k++) {
      double rsum = 0.0;
      for (int j = 0; j < l && k + j < nClusters-1; j++) {
        rsum += dist[k + j];
      }
      double rrcut = rsum + fac * dist[(k + l < nClusters-1) ? (k + l) : (nClusters-2)];
      redges[k] = centroids[k] + rrcut;
      invrc2[k] = 1.0 / (rrcut * rrcut);
    }

    double infty = std::numeric_limits<double>::infinity();
    ledges[0] = -infty;
    redges[nClusters-1] = infty;
    invlc2[0] = 0.0;
    invrc2[nClusters-1] = 0.0;
  }
}

// precomputes: 1 / span (max centroid[j,Z] - min centroid[j,Z])
// per-element, per-component normalization to [0,1]
void EAPOD::calculatePcaSpan()
{
  for (int elem = 0; elem < nelements; elem++) {
    double *centroids = &Centroids[nComponents*nClusters*elem];
    double *invspan   = &invPcaSpan[nComponents*elem];
    for (int c = 0; c < nComponents; c++) {
      double cmin = centroids[c];
      double cmax = centroids[c];
      for (int j = 1; j < nClusters; j++) {
        double v = centroids[c + j*nComponents];
        if (v < cmin) cmin = v;
        if (v > cmax) cmax = v;
      }
      double span = cmax - cmin;
      invspan[c] = (span > 0.0) ? 1.0 / span : 1.0;
    }
  }
}

inline void EAPOD::cluster_cutoff_hat_train(double u, double &fcut, double &dfcut_du)
{
  constexpr int p = 2;
  constexpr int q = 4;
  constexpr int pq = -p * q;

  double up_m1 = powint(u, p - 1);
  double up    = up_m1 * u;
  double v     = 1.0 - up;
  double v_qm1 = powint(v, q - 1);

  fcut     = v_qm1 * v;
  dfcut_du = pq * up_m1 * v_qm1;
}

inline void EAPOD::cluster_cutoff_poly_sq_train(double u, double &fcut, double &dfcut)
{
  double u2 = u * u;
  double omu = 1.0 - u;

  // fcut(u) = 1 - 10 * u3 + 15 * u4 - 6 * u5;
  fcut = 1.0 - u2 * u * (10.0 - u * (15.0 - 6.0 * u));
  dfcut = -30.0 * u2 * omu * omu;
}

void EAPOD::peratomlocalenvironment_descriptors(double *P, double *dP_dR, double *B, double *dB_dR, double *tmp, int elem, int nNeighbors)
{
  memset(P, 0, nClusters * sizeof(*P));
  memset(dP_dR, 0, 3*nNeighbors*nClusters * sizeof(*dP_dR));

  const int ncde = nComponents*Mdesc*elem;
  const int ncce = nComponents*nClusters*elem;

  double *ProjMat = &Proj[ncde];
  double *centroids = &Centroids[ncce];

  double *invlcut2 = &invLeftClusterRcut2[ncce];
  double *invrcut2 = &invRightClusterRcut2[ncce];
  double *ledges = &leftClusterEdges[ncce];
  double *redges = &rightClusterEdges[ncce];
  
  // tmp memory treated for the nComponent dimensional PCA (general case)
  // Local EA-POD currently only supports nComponents==1 (enforced at init)
  const int nActClsCp = nMaxActiveClusters*nComponents;
  const int ntmpmem = nComponents + nMaxActiveClusters*(4*nComponents + nMaxActiveClusters + 2*Mdesc);
  memset(tmp, 0, ntmpmem * sizeof(*tmp));

  double *pca = &tmp[0];
  double *D = &tmp[nComponents];
  double *clusterFcut = &tmp[nComponents + nActClsCp];
  double *clusterDFcut = &tmp[nComponents + 2*nActClsCp];
  double *dD_dpca = &tmp[nComponents + 3*nActClsCp];
  double *dP_dD = &tmp[nComponents + 4*nActClsCp];
  double *dD_dB = &tmp[nComponents + 4*nActClsCp + nMaxActiveClusters*nMaxActiveClusters];
  double *dP_dB = &tmp[nComponents + 4*nActClsCp + nMaxActiveClusters*nMaxActiveClusters + nMaxActiveClusters*Mdesc];

  // calculate pca descriptors
  for (int k = 0; k < nComponents; k++) {
    double sum = 0.0;
    for (int m = 0; m < Mdesc; m++) {
      sum += ProjMat[k + nComponents*m] * B[m];
    }
    pca[k] = sum;
  }

  int ks = 0;
  int ke = nClusters;
  find_active_clusters(pca[0], ledges, redges, nClusters, nMaxActiveClusters, ks, ke);
  int kn = ke - ks;

  // Cluster soft membership and dD/dpca
  double fcut, dfcut_du;  // fcut and dfcut/dpca[n]
  for (int j = 0; j < kn; j++) {
    double inv_rcut2 = 0.0;
    if      (pca[0] >= centroids[0 + (j+ks)*nComponents]) inv_rcut2 = invrcut2[0 + (j+ks)*nComponents];
    else if (pca[0] < centroids[0 + (j+ks)*nComponents]) inv_rcut2 = invlcut2[0 + (j+ks)*nComponents];

    double s2 = 0.0;
    for (int n = 0; n < nComponents; n++) {
      double pc = pca[n] - centroids[n + (j+ks)*nComponents];
      s2 += pc * pc;
    }
    double u = s2 * inv_rcut2;

    // cluster cutoff choice
    cluster_cutoff_hat_train(u, fcut, dfcut_du);
    //cluster_cutoff_poly_sq_train(u, fcut, dfcut_du);

    // guards for training
    if (abs(u - 1.0) <= 1e-4) {
      fcut = 0.0;
      dfcut_du = 0.0;
    }

    if (u <= 1e-4) {
      fcut = 1.0;
      dfcut_du = 0.0;
    }
    
    // dfcut/dpca[n] = (df/du) * (du/dpca[n])
    clusterFcut[j] = fcut;
    for (int n = 0; n < nComponents; n++) {
      double pc = pca[n] - centroids[n + (j+ks)*nComponents];
      double du_dpc = 2.0 * pc * inv_rcut2;
      clusterDFcut[j + n*kn] = dfcut_du * du_dpc;
    }
    D[j] = s2 + 1e-20; // fix for zero distances
  }

  for (int j = 0; j < kn; j++) {
    D[j] = 1.0 / D[j];
  }

  // dD/dpca[n] = dfcut/dpca[n] * D_inv  +  fcut * dD_inv/dpca[n]
  for (int j = 0; j < kn; j++) {
    for (int n = 0; n < nComponents; n++) {
      double pc         = pca[n] - centroids[n + (j+ks)*nComponents];
      double dDinv_dpca = -2.0 * pc * D[j] * D[j];
      double dfcut_dpca = clusterDFcut[j + n*kn];

      dD_dpca[j + n*kn] = dfcut_dpca * D[j] + clusterFcut[j] * dDinv_dpca;
    }
  }

  double sumD = 0.0;
  for (int j = 0; j < kn; j++) {
    D[j] *= clusterFcut[j];
    sumD += D[j];
  }

  double S1 = 1.0 / sumD;
  for (int j = 0; j < kn; j++) {
    P[j+ks] = D[j] * S1;
  }

  // calculate dP_dD
  double S2 = S1 * S1;
  for (int j = 0; j < kn; j++) {
    for (int k = 0; k < kn; k++) {
      dP_dD[k + j * kn] = -D[k] * S2;
    }
    dP_dD[j + j * kn] += S1;
  }

  // calculate dD_dB
  char chn = 'N';
  char cht = 'T';
  double alpha = 1.0, beta = 0.0;
  DGEMM(&chn, &chn, &kn, &Mdesc, &nComponents, &alpha, dD_dpca, &kn, ProjMat, &nComponents, &beta, dD_dB, &kn);

  // calculate dP_dB = dP_dD * dD_dB, which are derivatives of probabilities with respect to local descriptors
  DGEMM(&chn, &chn, &kn, &Mdesc, &kn, &alpha, dP_dD, &kn, dD_dB, &kn, &beta, dP_dB, &kn);

  // calculate dP_dR = dB_dR * dP_dB , which are derivatives of probabilities with respect to atomic coordinates
  int N = 3*nNeighbors;
  DGEMM(&chn, &cht, &N, &kn, &Mdesc, &alpha, dB_dR, &N, dP_dB, &kn, &beta, &dP_dR[N * ks], &N);
  
}

void EAPOD::peratomenvironment_descriptors(double *P, double *dP_dR, double *B, double *dB_dR, double *tmp, int elem, int nNeighbors)
{
  double *ProjMat = &Proj[nComponents*Mdesc*elem];
  double *centroids = &Centroids[nComponents*nClusters*elem];
  double *pca = &tmp[0];
  double *D = &tmp[nComponents];
  double *dD_dpca = &tmp[nComponents + nClusters];
  double *dD_dB = &tmp[nComponents + nClusters + nClusters*nComponents];
  double *dP_dD = &tmp[nComponents + nClusters + nClusters*nComponents + nClusters*Mdesc];
  double *dP_dB = &tmp[nComponents + nClusters + nClusters*nComponents + nClusters*Mdesc + nClusters*nClusters];

  // calculate principal components
  for (int k = 0; k < nComponents; k++) {
    pca[k] = 0.0;
    for (int m = 0; m < Mdesc; m++) {
      pca[k] += ProjMat[k + nComponents*m] * B[m];
    }
  }

  // calculate inverse square distances
  double sumD = 0.0;
  for (int j = 0; j < nClusters; j++) {
    D[j] = 1e-20; // fix for zero distances
    for (int k = 0; k < nComponents; k++) {
      D[j] += (pca[k] - centroids[k + j * nComponents]) * (pca[k] - centroids[k + j * nComponents]);
    }
    D[j] = 1.0 / D[j];
    sumD += D[j];
  }

  // calculate probabilities
  for (int j = 0; j < nClusters; j++) {
    P[j] = D[j] / sumD;
  }

  // calculate dD_dpca
  for (int n = 0; n < nComponents; n++) {
    for (int k = 0; k < nClusters; k++) {
      dD_dpca[k + n * nClusters] = 2 * D[k] * D[k] * (centroids[n + k * nComponents] - pca[n]);
    }
  }

  // calculate dD_dB
  char chn = 'N';
  char cht = 'T';
  double alpha = 1.0, beta = 0.0;
  DGEMM(&chn, &chn, &nClusters, &Mdesc, &nComponents, &alpha, dD_dpca, &nClusters, ProjMat, &nComponents, &beta, dD_dB, &nClusters);

  // calculate dP_dD
  double S1 = 1.0 / sumD;
  double S2 = S1 * S1;
  for (int k = 0; k < nClusters; k++) {
    for (int j = 0; j < nClusters; j++) {
      dP_dD[j + k * nClusters] = -D[j] * S2;
    }
  }
  for (int j = 0; j < nClusters; j++) {
    dP_dD[j + j * nClusters] += S1;
  }

  // calculate dP_dB = dP_dD * dD_dB, which are derivatives of probabilities with respect to local descriptors
  DGEMM(&chn, &chn, &nClusters, &Mdesc, &nClusters, &alpha, dP_dD, &nClusters, dD_dB, &nClusters, &beta, dP_dB, &nClusters);

  // calculate dP_dR = dB_dR * dP_dB , which are derivatives of probabilities with respect to atomic coordinates
  int N = 3*nNeighbors;
  DGEMM(&chn, &cht, &N, &nClusters, &Mdesc, &alpha, dB_dR, &N, dP_dB, &nClusters, &beta, dP_dR, &N);
}

inline int EAPOD::binom(int n, int k) {
  if (k == 0 || k == n) return 1;
  if (k > n - k) k = n - k;   // symmetry C(n,k)=C(n,n-k)
  int facni = 1;
  int faci = 1;
  for (int i = 0; i < k; ++i) {
    facni *= (n - i);
    faci *= (i + 1);
  }
  return facni / faci;
}

// number of monomials of total degree < d
inline int EAPOD::npa(int d) {
  return d * (d + 1) * (d + 2) / 6;
}

// number of monomials of total degree, d
inline int EAPOD::nmono(int d) {
  return (d + 1) * (d + 2) / 2;
}

inline int EAPOD::trinom(int p, int q, int r) {
  return binom(p + q + r, p) * binom(q + r, q);
}

inline int EAPOD::monomial_idx(int p, int q, int r) {
  int d = p + q + r;
  return npa(d) + (d + 1) * r - r * (r - 1) / 2 + q;
}

inline int EAPOD::fourbody_channels(int Pa, int *pa, int *deg)
{
  int n = 0;
  int total = 0;
  for (int d = 0; d <= Pa; ++d) {
    for (int a = d; a >= 0; --a) {
      for (int c = MIN(a, d - a); c >= 0; --c) {
        const int b = d - a - c;
        if (b > c) break;           // keep a >= c >= b
        if (pa) pa[n] = total;
        if (deg) deg[n] = d;
        total += nmono(a) * nmono(b) * nmono(c);
        ++n;
      }
    }
  }
  if (pa) pa[n] = total;
  return n;
}

void EAPOD::init_active_angular_ranges()
{
  // cleanup in case this is called again
  memory->destroy(p3_active);
  memory->destroy(p4_active);
  memory->destroy(dabf3_active);
  memory->destroy(dabf4_active);

  // 3-body: for a full channel p, its degree equals p
  nabf3_active = 0;
  for (int p = 0; p < nabf3; ++p)
    if (p >= L3min && p <= L3max) ++nabf3_active;

  memory->create(p3_active,    nabf3_active, "p3_active");
  memory->create(dabf3_active, nabf3_active, "dabf3_active");

  for (int p = 0, a = 0; p < nabf3; ++p) {
    if (p >= L3min && p <= L3max) {
      p3_active[a]    = p;
      dabf3_active[a] = p;
      ++a;
    }
  }

  // 4-body: use cached degrees
  nabf4_active = 0;
  for (int p = 0; p < nabf4; ++p) {
    const int d = deg4_full[p];
    if (d >= L4min && d <= L4max) ++nabf4_active;
  }

  memory->create(p4_active,    nabf4_active, "p4_active");
  memory->create(dabf4_active, nabf4_active, "dabf4_active");

  for (int p = 0, a = 0; p < nabf4; ++p) {
    const int d = deg4_full[p];
    if (d >= L4min && d <= L4max) {
      p4_active[a]    = p;
      dabf4_active[a] = d;
      ++a;
    }
  }
}

/**
 * @brief Initialize arrays for the three-body descriptors.
 *
 * @param Pa3 The degree of the angular basis functions of the three-body descriptors.
 */
void EAPOD::init3body(int Pa3)
{
  nabf3 = Pa3 + 1;    // Number of angular basis functions
  K3 = npa(nabf3);  // number of monomials

  // Allocate memory for the coefficients, the basis functions, and the cutoff function
  memory->create(pn3, nabf3+1, "pn3"); // array stores the number of monomials for each degree
  memory->create(pq3, K3*2, "pq3"); // array needed for the recursive computation of the angular basis functions
  memory->create(pc3, K3, "pc3");   // array needed for the computation of the three-body descriptors

  // Initialize the arrays
  init3bodyarray(pn3, pq3, pc3, Pa3);
}

/**
 * @brief Initialize arrays for the four-body descriptors.
 *
 * @param Pa4 The degree of the angular basis functions of the four-body descriptors.
 */
void EAPOD::init4body(int Pa4)
{
  // Set the number of monomials for the angular basis functions of the four-body descriptors
  K4 = npa(Pa4+1);

  // Enumerate 4-body channels: get count, offsets, and per-channel degrees.
  nabf4 = fourbody_channels(Pa4, nullptr, nullptr);
  memory->create(pa4,       nabf4 + 1, "pa4");
  memory->create(deg4_full, nabf4,     "deg4_full");
  fourbody_channels(Pa4, pa4, deg4_full);
  Q4 = pa4[nabf4];

  memory->create(pb4, Q4*3, "pb4");
  memory->create(pc4, Q4, "pc4");

  init4bodyarray(pa4, pb4, pc4, Pa4);
}

void EAPOD::init3bodyarray(int *np, int *pq, int *pc, int Pa)
{
  const int maxD = Pa + 1;
  for (int d = 0; d <= maxD; ++d) np[d] = npa(d);
  const int K = np[maxD];

  // seed: monomial (0,0,0)
  pc[0] = 1;
  pq[0] = 0;    // parent monomial index (unused for d=0)
  pq[K] = 0;    // recursion direction   (unused for d=0)

  int idx = 1;
  for (int d = 1; d < maxD; ++d) {
    for (int r = 0; r <= d; ++r) {
      for (int q = 0; q <= d - r; ++q) {
        const int p = d - q - r;
        pc[idx] = trinom(p, q, r);

        int pp = p, qq = q, rr = r, dir;
        if      (r > 0) { rr = r - 1; dir = 3; }
        else if (q > 0) { qq = q - 1; dir = 2; }
        else            { pp = p - 1; dir = 1; }

        pq[idx]     = monomial_idx(pp, qq, rr);
        pq[idx + K] = dir;
        ++idx;
      }
    }
  }
}

void EAPOD::init4bodyarray(int *pa4, int *pb4, int *pc4, int Pa)
{
  int iq = 0;
  for (int d = 0; d <= Pa; ++d) {
    for (int a = d; a >= 0; --a) {
      for (int c = MIN(a, d - a); c >= 0; --c) {
        const int b = d - a - c;
        if (b > c) break;
        for (int s1 = 0; s1 <= a + c; ++s1) {
          for (int r1 = 0; r1 <= s1; ++r1) {
            const int p1 = a + c - s1, q1 = s1 - r1;
            for (int s2 = 0; s2 <= a + b; ++s2) {
              for (int r2 = 0; r2 <= s2; ++r2) {
                const int p2 = a + b - s2, q2 = s2 - r2;
                for (int s3 = 0; s3 <= b + c; ++s3) {
                  for (int r3 = 0; r3 <= s3; ++r3) {
                    const int p3 = b + c - s3, q3 = s3 - r3;
                    const int tp = p1 + p2 - p3, tq = q1 + q2 - q3, tr = r1 + r2 - r3;
                    if ((tp < 0) || (tq < 0) || (tr < 0)) continue;
                    if (((tp | tq | tr) & 1) != 0)     continue;
                    const int up = tp / 2, uq = tq / 2, ur = tr / 2;
                    const int vp = p2 - up, vq = q2 - uq, vr = r2 - ur;
                    const int wp = p1 - up, wq = q1 - uq, wr = r1 - ur;
                    if ((vp < 0) || (vq < 0) || (vr < 0)) continue;
                    if ((wp < 0) || (wq < 0) || (wr < 0)) continue;

                    pb4[iq       ] = monomial_idx(p1, q1, r1);
                    pb4[iq +   Q4] = monomial_idx(p2, q2, r2);
                    pb4[iq + 2*Q4] = monomial_idx(p3, q3, r3);
                    pc4[iq]        = trinom(up, uq, ur) *
                                     trinom(vp, vq, vr) *
                                     trinom(wp, wq, wr);
                    ++iq;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
