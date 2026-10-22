// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   aE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Ngoc Cuong Nguyen (MIT) and Dionysios Sema (MIT)
------------------------------------------------------------------------- */

#include "pair_pod_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "math_const.h"
#include "memory_kokkos.h"
#include "neighbor_kokkos.h"
#include "neigh_request.h"
#include "safe_pointers.h"

#include <cstring>
#include <chrono>
#include <utility>
#include <type_traits>
#include <algorithm>
#include <vector>

#include "eapod.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

static constexpr int GPU_CONCURRENCY_DIV = 1;

static constexpr int POD_MAX_NRBF  = 5;   // max nrbf3 held in registers
static constexpr int POD_MAX_NELEM = 1;   // max nelements in the fused sum
static constexpr int POD_MAX_DEG   = 3;   // max monomial degree (P3)

// true : allbody_forces handles force calculations, tallying: forces, virial
// false: allbody_forces writes fij, tallyforce() does the atomics (original, for benchmarking)
static constexpr bool POD_FUSE_FORCE_TALLY = true;

// monomial direction table: 0 -> 1, 1 -> u, 2 -> v, 3 -> w
KOKKOS_FORCEINLINE_FUNCTION
KK_FLOAT pod_uvw_factor(const int c, const KK_FLOAT u, const KK_FLOAT v, const KK_FLOAT w)
{
  return (c == 1) ? u : ((c == 2) ? v : ((c == 3) ? w : (KK_FLOAT) 1.0));
}

enum{FS,FS_SHIFTEDSCALED};

KOKKOS_FORCEINLINE_FUNCTION constexpr int npa(int d)   { return d*(d+1)*(d+2)/6; }  // # monomials of degree < d
KOKKOS_FORCEINLINE_FUNCTION constexpr int nmono(int d) { return (d+1)*(d+2)/2;   }  // # monomials of degree == d

// Linear index of monomial u^p v^q w^r, d = p+q+r.
KOKKOS_FORCEINLINE_FUNCTION constexpr int pod_mono_index(int p, int q, int r) {
  const int d = p + q + r;
  return (d*(d+1)*(d+2))/6 + (d+1)*r - (r*(r-1))/2 + q;
}

/* ----------------------------------------------------------------------
   compile-time index usable on host and device
   (std::integral_constant::operator int is host-only under nvcc)
------------------------------------------------------------------------- */

template<int I>
struct pod_ic {
  static constexpr int value = I;
  KOKKOS_FORCEINLINE_FUNCTION constexpr operator int() const { return I; }
};

/* ----------------------------------------------------------------------
   compile-time loop: replaces #pragma unroll, index is a constant
------------------------------------------------------------------------- */

template<class F, int... I>
KOKKOS_FORCEINLINE_FUNCTION
void pod_static_for_impl(F &f, std::integer_sequence<int, I...>)
{
  (f(pod_ic<I>{}), ...);
}

template<int N, class F>
KOKKOS_FORCEINLINE_FUNCTION
void pod_static_for(F &&f)
{
  if constexpr (N > 0) pod_static_for_impl(f, std::make_integer_sequence<int, N>{});
}

/* ----------------------------------------------------------------------
   host-side runtime: compile-time integer dispatch, v in [Lo,Hi]
------------------------------------------------------------------------- */

template<int Lo, int Hi, class F>
inline void pod_dispatch_int(int v, F &&f)
{
  if constexpr (Lo <= Hi) {
    if (v == Lo) f(std::integral_constant<int, Lo>{});
    else pod_dispatch_int<Lo + 1, Hi>(v, f);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairPODKokkos<DeviceType>::PairPODKokkos(LAMMPS *lmp) : PairPOD(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
  host_flag = (execution_space == HostKK);

  ni = 0;
  nimax = 0;
  nij = 0;
  nijmax = 0;
  atomBlockSize = getStreamingProcessorCount();
  if (comm->me == 0) utils::logmesg(lmp, "Atom Block Size: {:d}\n", atomBlockSize);
  nAtomBlocks = 0;
  timing = 0;
  for (int i=0; i<30; i++) comptime[i] = 0;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

template<class DeviceType>
PairPODKokkos<DeviceType>::~PairPODKokkos()
{
  if (copymode) return;

  memoryKK->destroy_kokkos(k_eatom,eatom);
  memoryKK->destroy_kokkos(k_vatom,vatom);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::init_style()
{
  if (host_flag) {
    if (lmp->kokkos->nthreads > 1)
      error->all(FLERR,"Pair style pod/kk can currently only run on a single "
                         "CPU thread");

    PairPOD::init_style();
    return;
  }

  if (atom->tag_enable == 0) error->all(FLERR, "Pair style POD requires atom IDs");
  if (force->newton_pair == 0) error->all(FLERR, "Pair style POD requires newton pair on");

  neighflag = lmp->kokkos->neighflag;

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL)
    error->all(FLERR,"Must use half neighbor list style with pair pace/kk");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairPODKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairPOD::init_one(i,j);
  //double cutoneji = PairPOD::init_one(j,i);
  //k_cutsq.view_host()(i,j) = cutone*cutone;
  //k_cutsq.view_host()(j,i) = cutoneji*cutoneji;

  //k_cutsq.view_host()(i,j) = k_cutsq.view_host()(j,i) = cutone*cutone;
  //k_cutsq.modify_host();

  return cutone;
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::coeff(int narg, char **arg)
{
  if (narg < 5) utils::missing_cmd_args(FLERR, "pair_coeff", error);

  PairPOD::coeff(narg,arg); // create a PairPOD object

  copy_from_pod_class(PairPOD::fastpodptr); // copy parameters and arrays from pod class

  int n = atom->ntypes + 1;
  MemKK::realloc_kokkos(d_map, "pair_pod:map", n);

  MemKK::realloc_kokkos(k_cutsq, "pair_pod:cutsq", n, n);
  //d_cutsq = k_cutsq.template view<DeviceType>();

  MemKK::realloc_kokkos(k_scale, "pair_pod:scale", n, n);
  d_scale = k_scale.template view<DeviceType>();

  // Set up element lists

  auto h_map = Kokkos::create_mirror_view(d_map);

  for (int i = 1; i <= atom->ntypes; i++)
    h_map(i) = map[i];

  Kokkos::deep_copy(d_map,h_map);

  MemKK::realloc_kokkos(rin, "pair_pod:rin", nelements, nelements);
  auto h_rin = Kokkos::create_mirror_view(rin);
  for (int i = 0; i < nelements; i++)
    for (int j = 0; j < nelements; j++) {
      h_rin(i, j) = fastpodptr->rin[i][j];
    }
      
  Kokkos::deep_copy(rin, h_rin);

  MemKK::realloc_kokkos(rcut, "pair_pod:rcut", nelements, nelements);
  auto h_rcut = Kokkos::create_mirror_view(rcut);
  for (int i = 0; i < nelements; i++)
    for (int j = 0; j < nelements; j++) {
      h_rcut(i, j) = fastpodptr->rcut[i][j];
    }
      
  Kokkos::deep_copy(rcut, h_rcut);

  MemKK::realloc_kokkos(invrdiff, "pair_pod:invrdiff", nelements, nelements);
  auto h_invrdiff = Kokkos::create_mirror_view(invrdiff);
  for (int i = 0; i < nelements; i++)
    for (int j = 0; j < nelements; j++) {
      h_invrdiff(i, j) = fastpodptr->invrdiff[i][j];
    }
      
  Kokkos::deep_copy(invrdiff, h_invrdiff);

  MemKK::realloc_kokkos(rcutsq, "pair_pod:rcutsq", nelements, nelements);
  auto h_rcutsq = Kokkos::create_mirror_view(rcutsq);
  for (int i = 0; i < nelements; i++)
    for (int j = 0; j < nelements; j++) {
      KK_FLOAT rcutsq_ij = fastpodptr->rcutsq[i][j];
      k_cutsq.view_host()(i,j) = rcutsq_ij;
      k_cutsq.modify_host();
      h_rcutsq(i, j) =  rcutsq_ij;
    }
  d_cutsq = k_cutsq.template view<DeviceType>();
  Kokkos::deep_copy(rcutsq, h_rcutsq);

}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::allocate()
{
  PairPOD::allocate();
}

template<class DeviceType>
struct FindMaxNumNeighs {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  NeighListKokkos<DeviceType> k_list;

  FindMaxNumNeighs(NeighListKokkos<DeviceType>* nl): k_list(*nl) {}
  ~FindMaxNumNeighs() {k_list.copymode = 1;}

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator() (const int& ii, int& max_neighs) const {
    const int i = k_list.d_ilist[ii];
    const int num_neighs = k_list.d_numneigh[i];
    if (max_neighs<num_neighs) max_neighs = num_neighs;
  }
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary
  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  copymode = 1;
  int newton_pair = force->newton_pair;
  if (newton_pair == false)
    error->all(FLERR,"PairPODKokkos requires 'newton on'");

  atomKK->sync(execution_space,X_MASK|F_MASK|TYPE_MASK);
  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  maxneigh = 0;
  if (host_flag) {
    inum = list->inum;
    d_numneigh = typename AT::t_int_1d("pair_pod:numneigh",inum);
    for (int i=0; i<inum; i++) d_numneigh(i) = list->numneigh[i];
    d_ilist = typename AT::t_int_1d("pair_pod:ilist",inum);
    for (int i=0; i<inum; i++) d_ilist(i) = list->ilist[i];

    int maxn = 0;
    for (int i=0; i<inum; i++)
      if (maxn < list->numneigh[i]) maxn = list->numneigh[i];
    MemoryKokkos::realloc_kokkos(d_neighbors,"neighlist:neighbors",inum,maxn);
    for (int i=0; i<inum; i++) {
      int gi = list->ilist[i];
      int m = list->numneigh[gi];
      if (maxneigh<m) maxneigh = m;
      for (int l = 0; l < m; l++) {           // loop over each atom around atom i
        d_neighbors(gi, l) = list->firstneigh[gi][l];
      }
    }
  }
  else {
    NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
    d_numneigh = k_list->d_numneigh;
    d_neighbors = k_list->d_neighbors;
    d_ilist = k_list->d_ilist;
    inum = list->inum;
    int maxneighs;
    Kokkos::parallel_reduce("PairPODKokkos::find_max_neighs",inum, FindMaxNumNeighs<DeviceType>(k_list), Kokkos::Max<int>(maxneighs));
    maxneigh = maxneighs;
  }

  auto begin = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  // determine the number of atom blocks and divide atoms into blocks
  nAtomBlocks = calculateNumberOfIntervals(inum, atomBlockSize);
  if (nAtomBlocks > 100) nAtomBlocks = 100;
  divideInterval(atomBlocks, inum, nAtomBlocks);

  int nmax = 0;
  for (int block=0; block<nAtomBlocks; block++) {
    int n = atomBlocks[block+1] - atomBlocks[block];
    if (nmax < n) nmax = n;
  }
  grow_atoms(nmax);
  grow_pairs(nmax*maxneigh);

  for (int block=0; block<nAtomBlocks; block++) {
    int gi1 = atomBlocks[block]-1;
    int gi2 = atomBlocks[block+1]-1;
    ni = gi2 - gi1; // total number of atoms in the current atom block

    begin = std::chrono::high_resolution_clock::now();
    // calculate the total number of pairs (i,j) in the current atom block
    nij = NeighborCount(numij, gi1, ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[0] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

    begin = std::chrono::high_resolution_clock::now();
    // obtain the neighbors within rcut
    NeighborList(rr, rhat, numij, typeai, idxi, ai, aj, ti, tj, gi1, ni, nij);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[1] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

    // compute atomic energy and force for the current atom block
    begin = std::chrono::high_resolution_clock::now();
    blockatom_energyforce(ei, fij, ni, nij);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[2] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

    begin = std::chrono::high_resolution_clock::now();
    // tally atomic energy to global energy
    tallyenergy(ei, gi1, ni);

    // tally atomic force to global force (for benchmarking)
    if (!POD_FUSE_FORCE_TALLY) tallyforce(fij, ai, aj, nij);

    // tally atomic stress
    //if (vflag) tallystress(fij, rij, ai, aj, nij);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[3] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

    //savedatafordebugging();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.sync_host();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.sync_host();
  }

  atomKK->modified(execution_space,F_MASK);

  copymode = 0;
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::copy_from_pod_class(EAPOD *podptr)
{
  nelements = podptr->nelements; // number of elements
  onebody = podptr->onebody;   // one-body descriptors
  besseldegree = podptr->besseldegree; // degree of Bessel functions
  inversedegree = podptr->inversedegree; // degree of inverse functions
  nbesselpars = podptr->nbesselpars;  // number of Bessel parameters
  nCoeffPerElement = podptr->nCoeffPerElement; // number of coefficients per element = (nl1 + Mdesc*nClusters)
  nl1 = podptr->nl1;  // number of one-body descriptors
  nl2 = podptr->nl2;  // number of two-body descriptors
  nl3 = podptr->nl3;  // number of three-body descriptors
  nl4 = podptr->nl4;  // number of four-body descriptors
  nl33 = podptr->nl33; // number of three-body x three-body descriptors
  nl34 = podptr->nl34; // number of three-body x four-body descriptors
  nl44 = podptr->nl44; // number of four-body x four-body descriptors
  nl = podptr->nl;   // number of local descriptors
  ns = podptr->ns;
  nrbfmax = podptr->nrbfmax;
  nrbf2 = podptr->nrbf2;
  nrbf3 = podptr->nrbf3;
  nrbf4 = podptr->nrbf4;
  nabf3 = podptr->nabf3;     // number of three-body angular basis functions
  nabf4 = podptr->nabf4;     // number of four-body angular basis functions
  nabf3_active = podptr->nabf3_active;     // number of three-body angular basis functions
  nabf4_active = podptr->nabf4_active;     // number of four-body angular basis functions
  P3 = podptr->P3;           // polynomial degree of three-body angular basis [0,...,P3]
  K3 = podptr->K3;           // number of three-body monomials
  K4 = podptr->K4;           // number of four-body monomials
  Q4 = podptr->Q4;           // number of four-body monomial coefficients
  nClusters = podptr->nClusters; // number of environment clusters
  nComponents = podptr->nComponents; // number of principal components
  Mdesc = podptr->Mdesc; // number of base descriptors
  eapod = podptr->eapod; // boolean for ea-pod method
  localeapod = podptr->localeapod; // boolean for local ea-pod method
  nActiveClusters = podptr->nActiveClusters; // average number of active clusters
  nMaxActiveClusters = podptr->nMaxActiveClusters; // max number of active clusters
  use_hermite   = podptr->use_hermite;
  nhermite_bins = podptr->nhermite_bins;
  use_bspline   = podptr->use_bspline;
  bs_ileft = podptr->bs_ileft;
  bs_nb = podptr->bs_nb;
  //hat_p = podptr->hat_p;    // order of hat basis function
  //hat_q = podptr->hat_q;    // order of hat function
  //hat_q1 = podptr->hat_q1;    // order-1 of hat function
  //hat_pq = podptr->hat_pq;

  if (nrbf3 > POD_MAX_NRBF)    error->all(FLERR, "pod/kk: nrbf3 exceeds POD_MAX_NRBF");
  if (nelements > POD_MAX_NELEM) error->all(FLERR, "pod/kk: nelements exceeds POD_MAX_NELEM");

  if (P3 > POD_MAX_DEG) error->all(FLERR, "pod/kk: P3 exceeds POD_MAX_DEG");

  MemKK::realloc_kokkos(coefficients, "pair_pod:coefficients", nCoeffPerElement * nelements);
  auto h_coefficients = Kokkos::create_mirror_view(coefficients);
  for (int i=0; i<nCoeffPerElement * nelements; i++) h_coefficients[i] = podptr->coeff[i];
  Kokkos::deep_copy(coefficients, h_coefficients);

  if (use_bspline) {
    const int npair = nelements * nelements;

    MemKK::realloc_kokkos(bs_r0, "pair_pod:bs_r0", npair);
    MemKK::realloc_kokkos(bs_invh, "pair_pod:bs_invh", npair);

    auto h_bsr0   = Kokkos::create_mirror_view(bs_r0);
    auto h_bsinvh = Kokkos::create_mirror_view(bs_invh);

    for (int i = 0; i < npair; ++i) {
      h_bsr0(i)   = podptr->bs_r0[i];
      h_bsinvh(i) = podptr->bs_invh[i];
    }

    Kokkos::deep_copy(bs_r0, h_bsr0);
    Kokkos::deep_copy(bs_invh, h_bsinvh);
  }

  if (use_hermite) {
    const int npair = nelements * nelements;
    const size_t ncoef = (size_t)npair * (size_t)nhermite_bins * (size_t)nrbfmax * 4ull;

    MemKK::realloc_kokkos(hermite_r0, "pair_pod:hermite_r0", npair);
    MemKK::realloc_kokkos(hermite_invdr, "pair_pod:hermite_invdr", npair);
    MemKK::realloc_kokkos(rbf_hermite_coeffs, "pair_pod:rbf_hermite_coeffs", ncoef);

    auto h_r0    = Kokkos::create_mirror_view(hermite_r0);
    auto h_invdr = Kokkos::create_mirror_view(hermite_invdr);
    auto h_coef  = Kokkos::create_mirror_view(rbf_hermite_coeffs);

    for (int i = 0; i < npair; ++i) {
      h_r0(i)    = podptr->hermite_r0[i];
      h_invdr(i) = podptr->hermite_invdr[i];
    }
    for (size_t i = 0; i < ncoef; ++i) {
      h_coef(i) = podptr->rbf_hermite_coeffs[i];
    }

    Kokkos::deep_copy(hermite_r0, h_r0);
    Kokkos::deep_copy(hermite_invdr, h_invdr);
    Kokkos::deep_copy(rbf_hermite_coeffs, h_coef);
  } else {
    const int nbc = nelements * nelements * nbesselpars;
    MemKK::realloc_kokkos(bessel_neg_alpha, "pair_pod:bessel_neg_alpha", nbc);
    MemKK::realloc_kokkos(bessel_pi_inv_t1, "pair_pod:bessel_pi_inv_t1", nbc);
    MemKK::realloc_kokkos(bessel_dx_factor, "pair_pod:bessel_dx_factor", nbc);

    auto h_neg_alpha = Kokkos::create_mirror_view(bessel_neg_alpha);
    auto h_pi_inv_t1 = Kokkos::create_mirror_view(bessel_pi_inv_t1);
    auto h_dx_factor = Kokkos::create_mirror_view(bessel_dx_factor);

    for (int it = 0; it < nelements; ++it) {
      for (int jt = 0; jt < nelements; ++jt) {
        const EAPOD::BesselConst *bc = podptr->bessel_const(it, jt);
        for (int j = 0; j < nbesselpars; ++j) {
          const int idx = (it * nelements + jt) * nbesselpars + j;
          h_neg_alpha[idx] = bc[j].neg_alpha;
          h_pi_inv_t1[idx] = bc[j].pi_inv_t1;
          h_dx_factor[idx] = bc[j].dx_factor;
        }
      }
    }
    Kokkos::deep_copy(bessel_neg_alpha, h_neg_alpha);
    Kokkos::deep_copy(bessel_pi_inv_t1, h_pi_inv_t1);
    Kokkos::deep_copy(bessel_dx_factor, h_dx_factor);

    MemKK::realloc_kokkos(Phi, "pair_pod:Phi", ns*ns);
    auto h_Phi = Kokkos::create_mirror_view(Phi);
    for (int i=0; i<ns*ns; i++) h_Phi[i] = podptr->Phi[i];
    Kokkos::deep_copy(Phi, h_Phi);
  }

  if (eapod) {
    MemKK::realloc_kokkos(Proj, "pair_pod:Proj",  Mdesc * nComponents * nelements);
    auto h_Proj = Kokkos::create_mirror_view(Proj);
    for (int i=0; i<Mdesc * nComponents * nelements; i++) h_Proj[i] = podptr->Proj[i];
    Kokkos::deep_copy(Proj, h_Proj);

    int ncce = nClusters * nComponents * nelements;
    MemKK::realloc_kokkos(Centroids, "pair_pod:Centroids", ncce);
    auto h_Centroids = Kokkos::create_mirror_view(Centroids);
    for (int i=0; i<ncce; i++) h_Centroids[i] = podptr->Centroids[i];
    Kokkos::deep_copy(Centroids, h_Centroids);

    if (localeapod) {
      MemKK::realloc_kokkos(invLeftClusterRcut2, "pair_pod:invLeftClusterRcut2", ncce);
      MemKK::realloc_kokkos(invRightClusterRcut2, "pair_pod:invRightClusterRcut2", ncce);
      MemKK::realloc_kokkos(leftClusterEdges, "pair_pod:leftClusterEdges", ncce);
      MemKK::realloc_kokkos(rightClusterEdges, "pair_pod:rightClusterEdges", ncce);

      auto h_invLeftClusterRcut2 = Kokkos::create_mirror_view(invLeftClusterRcut2);
      for (int i=0; i<ncce; i++) h_invLeftClusterRcut2[i] = podptr->invLeftClusterRcut2[i];
      Kokkos::deep_copy(invLeftClusterRcut2, h_invLeftClusterRcut2);

      auto h_invRightClusterRcut2 = Kokkos::create_mirror_view(invRightClusterRcut2);
      for (int i=0; i<ncce; i++) h_invRightClusterRcut2[i] = podptr->invRightClusterRcut2[i];
      Kokkos::deep_copy(invRightClusterRcut2, h_invRightClusterRcut2);

      auto h_leftClusterEdges = Kokkos::create_mirror_view(leftClusterEdges);
      for (int i=0; i<ncce; i++) h_leftClusterEdges[i] = podptr->leftClusterEdges[i];
      Kokkos::deep_copy(leftClusterEdges, h_leftClusterEdges);

      auto h_rightClusterEdges = Kokkos::create_mirror_view(rightClusterEdges);
      for (int i=0; i<ncce; i++) h_rightClusterEdges[i] = podptr->rightClusterEdges[i];
      Kokkos::deep_copy(rightClusterEdges, h_rightClusterEdges);
    }
  }

  MemKK::realloc_kokkos(pn3, "pn3", nabf3+1);
  MemKK::realloc_kokkos(pq_m, "pq_m", K3);
  MemKK::realloc_kokkos(pq_d, "pq_d", K3);
  MemKK::realloc_kokkos(pc3, "pc3", K3);
  
  auto h_pn3 = Kokkos::create_mirror_view(pn3);
  for (int i=0; i<nabf3+1; i++) h_pn3[i] = podptr->pn3[i];
  Kokkos::deep_copy(pn3, h_pn3);

  auto h_pq_m = Kokkos::create_mirror_view(pq_m);
  auto h_pq_d = Kokkos::create_mirror_view(pq_d);
  for (int i = 0; i < K3; i++) {
    h_pq_m[i] = podptr->pq3_m[i];
    h_pq_d[i] = podptr->pq3_d[i];
  }
  Kokkos::deep_copy(pq_m, h_pq_m);
  Kokkos::deep_copy(pq_d, h_pq_d);

  auto h_pc3 = Kokkos::create_mirror_view(pc3);
  for (int i = 0; i < K3; i++) h_pc3[i] = podptr->pc3[i];
  Kokkos::deep_copy(pc3, h_pc3);

  MemKK::realloc_kokkos(p3_active, "p3_active", nabf3_active);
  auto h_p3_active = Kokkos::create_mirror_view(p3_active);
  for (int i=0; i<nabf3_active; i++) h_p3_active[i] = podptr->p3_active[i];
  Kokkos::deep_copy(p3_active, h_p3_active);

  build_monomial_factors(pq_m, pq_d, K3);

  if (nl4 > 0) {
    MemKK::realloc_kokkos(pa4, "pa4", nabf4+1);
    MemKK::realloc_kokkos(pb4, "pb4", Q4*3);
    MemKK::realloc_kokkos(pc4, "pc4", Q4);

    auto h_pa4 = Kokkos::create_mirror_view(pa4);
    for (int i = 0; i < nabf4+1; i++) h_pa4[i] = podptr->pa4[i];
    Kokkos::deep_copy(pa4, h_pa4);

    auto h_pb4 = Kokkos::create_mirror_view(pb4);
    for (int i = 0; i < Q4*3; i++) h_pb4[i] = podptr->pb4[i];
    Kokkos::deep_copy(pb4, h_pb4);

    auto h_pc4 = Kokkos::create_mirror_view(pc4);
    for (int i = 0; i < Q4; i++) h_pc4[i] = podptr->pc4[i];
    Kokkos::deep_copy(pc4, h_pc4);

    MemKK::realloc_kokkos(p4_active, "p4_active", nabf4_active);
    auto h_p4_active = Kokkos::create_mirror_view(p4_active);
    for (int i=0; i<nabf4_active; i++) h_p4_active[i] = podptr->p4_active[i];
    Kokkos::deep_copy(p4_active, h_p4_active);
  }

  if (nl33 > 0) {
    MemKK::realloc_kokkos(ind33l, "pair_pod:ind33l", nl33);
    MemKK::realloc_kokkos(ind33r, "pair_pod:ind33r", nl33);

    auto h_ind33l = Kokkos::create_mirror_view(ind33l);
    for (int i = 0; i < nl33; i++) h_ind33l[i] = podptr->ind33l[i];
    Kokkos::deep_copy(ind33l, h_ind33l);

    auto h_ind33r = Kokkos::create_mirror_view(ind33r);
    for (int i = 0; i < nl33; i++) h_ind33r[i] = podptr->ind33r[i];
    Kokkos::deep_copy(ind33r, h_ind33r);
  }

  if (nl34 > 0) {
    MemKK::realloc_kokkos(ind34l, "pair_pod:ind34l", nl34);
    MemKK::realloc_kokkos(ind34r, "pair_pod:ind34r", nl34);

    auto h_ind34l = Kokkos::create_mirror_view(ind34l);
    for (int i = 0; i < nl34; i++) h_ind34l[i] = podptr->ind34l[i];
    Kokkos::deep_copy(ind34l, h_ind34l);

    auto h_ind34r = Kokkos::create_mirror_view(ind34r);
    for (int i = 0; i < nl34; i++) h_ind34r[i] = podptr->ind34r[i];
    Kokkos::deep_copy(ind34r, h_ind34r);
  }

  if (nl44 > 0) {
    MemKK::realloc_kokkos(ind44l, "pair_pod:ind44l", nl44);
    MemKK::realloc_kokkos(ind44r, "pair_pod:ind44r", nl44);

    auto h_ind44l = Kokkos::create_mirror_view(ind44l);
    for (int i = 0; i < nl44; i++) h_ind44l[i] = podptr->ind44l[i];
    Kokkos::deep_copy(ind44l, h_ind44l);

    auto h_ind44r = Kokkos::create_mirror_view(ind44r);
    for (int i = 0; i < nl44; i++) h_ind44r[i] = podptr->ind44r[i];
    Kokkos::deep_copy(ind44r, h_ind44r);
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::build_monomial_factors(t_pod_1i d_pq_m, t_pod_1i d_pq_d, const int K3)
{
  auto h_m = Kokkos::create_mirror_view(d_pq_m);  Kokkos::deep_copy(h_m, d_pq_m);
  auto h_d = Kokkos::create_mirror_view(d_pq_d);  Kokkos::deep_copy(h_d, d_pq_d);

  std::vector<int> deg(K3, 0), ex(3*K3, 0);   // exponents (p,q,r) per monomial
  for (int k = 1; k < K3; ++k) {
    const int par = h_m(k);
    deg[k] = deg[par] + 1;
    for (int c = 0; c < 3; ++c) ex[3*k+c] = ex[3*par+c];
    ex[3*k + h_d(k) - 1]++;
  }
  const int maxdeg = std::max(1, *std::max_element(deg.begin(), deg.end()));
  if (maxdeg != P3) error->all(FLERR, "pod/kk: monomial degrees mismatch with eapod.cpp");

  // allbody_forces relies on the closed-form ordering: verify it once
  for (int k = 0; k < K3; ++k) {
    const int q = ex[3*k+1], r = ex[3*k+2], d = deg[k];
    const int kk = npa(d) + (d+1)*r - (r*(r-1))/2 + q;
    if (kk != k) error->all(FLERR, "pod/kk: monomial ordering mismatch with allbody_forces");
  }

  MemKK::realloc_kokkos(pq_f, "pod:pq_f", P3*K3);
  auto h_f = Kokkos::create_mirror_view(pq_f);
  for (int k = 0; k < K3; ++k) {
    int tmp[POD_MAX_DEG], n = 0;
    for (int kk = k; kk != 0; kk = h_m(kk)) tmp[n++] = h_d(kk);
    for (int t = 0; t < n; ++t)  h_f(P3*k + t) = tmp[n-1-t];
    for (int t = n; t < P3; ++t) h_f(P3*k + t) = 0;   // padding: slot 0 = 1.0
  }
  Kokkos::deep_copy(pq_f, h_f);
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::divideInterval(int *intervals, int N, int M)
{
  int intervalSize = N / M; // Basic size of each interval
  int remainder = N % M;    // Remainder to distribute
  intervals[0] = 1;         // Start of the first interval
  for (int i = 1; i <= M; i++) {
    intervals[i] = intervals[i - 1] + intervalSize + (remainder > 0 ? 1 : 0);
    if (remainder > 0) {
      remainder--;
    }
  }
}

template<class DeviceType>
int PairPODKokkos<DeviceType>::calculateNumberOfIntervals(int N, int intervalSize)
{
  int M = N / intervalSize;
  if (N % intervalSize != 0) {
    M++; // Add an additional interval to cover the remainder
  }

  return M;
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::grow_atoms(int Ni)
{
  if (Ni > nimax) {
    nimax = Ni;
    MemKK::realloc_kokkos(numij, "pair_pod:numij", nimax+1);
    MemKK::realloc_kokkos(ei, "pair_pod:ei", nimax);
    MemKK::realloc_kokkos(typeai, "pair_pod:typeai", nimax);
    int n = nimax * nelements * K3 * nrbf3;
    MemKK::realloc_kokkos(sumU, "pair_pod:sumU", n);
    MemKK::realloc_kokkos(forcecoeff, "pair_pod:forcecoeff", n);
    MemKK::realloc_kokkos(bd, "pair_pod:bd", nimax * Mdesc);
    MemKK::realloc_kokkos(cb, "pair_pod:cb", nimax * Mdesc);
    if (localeapod) {
      MemKK::realloc_kokkos(d_ks, "pair_pod:ks", nimax);
      MemKK::realloc_kokkos(d_kn, "pair_pod:kn", nimax);
      int envtmpmem = 1 + nMaxActiveClusters;
      MemKK::realloc_kokkos(pd, "pair_pod:pd", nimax * envtmpmem);
    }
    else if (eapod) {
      int envtmpmem = 1 + nComponents + 3*nClusters;
      MemKK::realloc_kokkos(pd, "pair_pod:pd", nimax * envtmpmem);
    }
    Kokkos::deep_copy(numij, 0);
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::grow_pairs(int Nij)
{
  if (Nij > nijmax) {
    nijmax = Nij;
    MemKK::realloc_kokkos(rr,   "pair_pod:rr",   nijmax);
    MemKK::realloc_kokkos(rhat, "pair_pod:rhat", 3 * nijmax);
    if (!POD_FUSE_FORCE_TALLY)
      MemKK::realloc_kokkos(fij, "pair_pod:f_ij", 3 * nijmax);

    MemKK::realloc_kokkos(idxi, "pair_pod:idxi", nijmax);
    MemKK::realloc_kokkos(ai, "pair_pod:ai", nijmax);
    MemKK::realloc_kokkos(aj, "pair_pod:aj", nijmax);
    MemKK::realloc_kokkos(ti, "pair_pod:ti", nijmax);
    MemKK::realloc_kokkos(tj, "pair_pod:tj", nijmax);
    
    // Radial basis (nrbfmax = nrbf2 >= nrbf3)
    MemKK::realloc_kokkos(rbf, "pair_pod:rbf", nijmax * nrbfmax);
    MemKK::realloc_kokkos(drbf, "pair_pod:drbf", nijmax * nrbfmax);

    // Options:
    // 1) B-spline two-body radial basis: 4 active splines (of nrbf2 total)
    //    and Bessel for many-body (>= 3-body) radial basis (nrbfmax = nrbf3)
    // 2) Bessel for all body-order (nrbfmax = nrbf2)
    if (use_bspline) {
      MemKK::realloc_kokkos(rbf2, "pair_pod:rbf2", 4 * nijmax);
      MemKK::realloc_kokkos(drbf2, "pair_pod:drbf2", 4 * nijmax);
      MemKK::realloc_kokkos(bs_base, "pair_pod:bs_base", nijmax);
    } else {
      rbf2 = rbf;
      drbf2 = drbf;
    }

    if (!use_hermite) {
      MemKK::realloc_kokkos(rbft,  "pair_pod:rbft",  nijmax * ns);
      MemKK::realloc_kokkos(drbft, "pair_pod:drbft", nijmax * ns);
    }
  }
}

template<class DeviceType>
int PairPODKokkos<DeviceType>::NeighborCount(t_pod_1i l_numij, int gi1, int Ni)
{
  // create local shadow views for KOKKOS_LAMBDA to pass them into parallel_for
  auto l_ilist = d_ilist;
  auto l_x = x;
  auto l_numneigh = d_numneigh;
  auto l_neighbors = d_neighbors;
  auto l_map = d_map;
  auto l_type = type;
  auto l_rcutsq = rcutsq;

  // compute number of pairs for each atom i
  Kokkos::parallel_for("NeighborCount", typename Kokkos::TeamPolicy<DeviceType>(Ni, Kokkos::AUTO), KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team) {
    int i = team.league_rank();
    int gi = l_ilist(gi1 + i);
    int itype = l_map(l_type(gi));
    KK_FLOAT xi0 = l_x(gi, 0);
    KK_FLOAT xi1 = l_x(gi, 1);
    KK_FLOAT xi2 = l_x(gi, 2);
    int jnum = l_numneigh(gi);
    int ncount = 0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,jnum),
        [&] (const int jj, int& count) {
      int gj = l_neighbors(gi,jj);
      gj &= NEIGHMASK;
      int jtype = l_map(l_type(gj));
      KK_FLOAT delx = xi0 - l_x(gj,0);
      KK_FLOAT dely = xi1 - l_x(gj,1);
      KK_FLOAT delz = xi2 - l_x(gj,2);
      KK_FLOAT rsq = delx*delx + dely*dely + delz*delz;
      if (rsq < l_rcutsq(itype,jtype)) count++;
    },ncount);

    l_numij(i+1) = ncount;
  });

  // accumalative sum
  Kokkos::parallel_scan("InclusivePrefixSum", Kokkos::RangePolicy<DeviceType>(0,Ni + 1), KOKKOS_LAMBDA(int i, int& update, const bool final) {
    if (i > 0) {
      update += l_numij(i);
      if (final) {
        l_numij(i) = update;
      }
    }
  });

  int total_neighbors = 0;
  Kokkos::deep_copy(Kokkos::View<int,Kokkos::HostSpace>(&total_neighbors), Kokkos::subview(l_numij, Ni));

  return total_neighbors;
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::NeighborList(t_pod_1d l_rr, t_pod_1d l_rhat, t_pod_1i l_numij,
  t_pod_1i l_typeai, t_pod_1i l_idxi, t_pod_1i l_ai, t_pod_1i l_aj, t_pod_1i l_ti, t_pod_1i l_tj,
  int gi1, int Ni, int Nij)
{
  auto l_ilist = d_ilist;
  auto l_x = x;
  auto l_numneigh = d_numneigh;
  auto l_neighbors = d_neighbors;
  auto l_map = d_map;
  auto l_type = type;
  auto l_rcutsq = rcutsq;

  Kokkos::parallel_for("NeighborList", typename Kokkos::TeamPolicy<DeviceType>(Ni, Kokkos::AUTO),
    KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team) {
    const int i = team.league_rank();
    const int gi = l_ilist(gi1 + i);
    const int itype = l_map(l_type(gi));
    l_typeai(i) = itype;
    const KK_FLOAT xi0 = l_x(gi, 0);
    const KK_FLOAT xi1 = l_x(gi, 1);
    const KK_FLOAT xi2 = l_x(gi, 2);
    const int jnum = l_numneigh(gi);
    const int nij0 = l_numij(i);
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(team, jnum),
        [&] (const int jj, int& offset, bool final) {
      int gj = l_neighbors(gi, jj);
      gj &= NEIGHMASK;
      const int jtype = l_map(l_type(gj));
      const KK_FLOAT delx = l_x(gj,0) - xi0;
      const KK_FLOAT dely = l_x(gj,1) - xi1;
      const KK_FLOAT delz = l_x(gj,2) - xi2;
      const KK_FLOAT rsq = delx*delx + dely*dely + delz*delz;
      if (rsq >= l_rcutsq(itype,jtype)) return;
      if (final) {
        const int n = nij0 + offset;
        const KK_FLOAT r = Kokkos::sqrt(rsq);
        const KK_FLOAT s = (KK_FLOAT) 1.0 / r;
        l_rr(n)           = r;
        l_rhat(n)         = delx * s;
        l_rhat(n +   Nij) = dely * s;
        l_rhat(n + 2*Nij) = delz * s;
        l_idxi(n) = i;
        l_ai(n) = gi;
        l_aj(n) = gj;
        l_ti(n) = itype;
        l_tj(n) = jtype;
      }
      offset++;
    });
  });
}

template <class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::cutoff_exp(const KK_FLOAT y, const KK_FLOAT invrmax, const KK_FLOAT e_v,
                            KK_FLOAT &fcut, KK_FLOAT &dfcut)
{
  // y  = r * invrmax
  // f  = e_v * exp(-1/sqrt((y^3-1)^2 + 1e-6))
  // df = f * 3*invrmax*y^2*(y^3-1) / (y4*y5)
  //const KK_FLOAT y  = r * invrmax;
  const KK_FLOAT y2 = y * y;
  const KK_FLOAT y3 = y * y2 - 1.0;
  const KK_FLOAT y4 = y3 * y3 + 1.0e-6;
  const KK_FLOAT y5 = sqrt(y4);

  fcut  = e_v * exp(-1.0 / y5);
  dfcut = fcut * (3.0 * invrmax * y2 * y3) / (y4 * y5);
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::radialbasis(
    t_pod_1d rbft, t_pod_1d drbft,
    t_pod_1d l_rr, t_pod_2d l_rin, t_pod_2d l_invrdiff,
    t_pod_1d l_neg_alpha, t_pod_1d l_pi_inv_t1, t_pod_1d l_dx_factor,
    t_pod_1i l_ti, t_pod_1i l_tj,
    int l_nelements, int l_besseldegree, int l_inversedegree, int l_nbesselpars, int Nij)
{
  const KK_FLOAT E_V = static_cast<KK_FLOAT>(std::exp(1.0));

  Kokkos::parallel_for("ComputeRadialBasis",
    Kokkos::RangePolicy<DeviceType>(0, Nij), KOKKOS_LAMBDA(const int n)
  {
    const int itype   = l_ti(n);
    const int jtype   = l_tj(n);
    const int bc_base = (itype * l_nelements + jtype) * l_nbesselpars;

    const KK_FLOAT dij    = l_rr(n);
    const KK_FLOAT invdij = 1.0 / dij;

    const KK_FLOAT invrmax = l_invrdiff(itype, jtype);
    const KK_FLOAT r       = dij - l_rin(itype, jtype);
    const KK_FLOAT invr    = 1.0 / r;
    const KK_FLOAT y       = r * invrmax;

    // cutoff function for radial basis
    KK_FLOAT fcut, dfcut;
    cutoff_exp(y, invrmax, E_V, fcut, dfcut);

    // Shared Bessel constants
    const KK_FLOAT f1   = fcut * invr;
    const KK_FLOAT g1   = (dfcut - f1) * invr;
    const KK_FLOAT bfac = sqrt(2.0 * invrmax);
    const KK_FLOAT bf1  = bfac * f1;
    const KK_FLOAT bg1  = bfac * g1;

    int nij = n;

    // Bessel radial basis
    for (int j = 0; j < l_nbesselpars; ++j) {
      const int bcj = bc_base + j;

      const KK_FLOAT mt2   = expm1(l_neg_alpha(bcj) * y);
      const KK_FLOAT xpi   = mt2 * l_pi_inv_t1(bcj);
      const KK_FLOAT pi_dx = l_dx_factor(bcj) * (1.0 + mt2);
      const KK_FLOAT Kf1dx = bf1 * pi_dx;

      KK_ACC_FLOAT ix = xpi;
      for (int i = 1; i <= l_besseldegree; ++i) {
        const KK_FLOAT cosax  = cos(ix);
        const KK_FLOAT inv_i  = 1.0 / KK_FLOAT(i);
        const KK_FLOAT isinax = inv_i * sin(ix);

        const KK_FLOAT rbfv    = bf1 * isinax;
        const KK_FLOAT drbftdr = bg1 * isinax + Kf1dx * cosax;

        rbft (nij) = rbfv;
        drbft(nij) = drbftdr;  // R'(r)

        ix  += xpi;
        nij += Nij;
      }
    }

    // Inverse-poly radial basis
    // rbf_i = fcut * invdij^i
    // drbf_i/dr = (dfcut - i*fcut*invdij) * invdij^i
    const KK_FLOAT fcut_invd = fcut * invdij;
    KK_FLOAT dterm = dfcut;
    KK_FLOAT inva  = 1.0;

    for (int i = 1; i <= l_inversedegree; ++i) {
      dterm -= fcut_invd;
      inva  *= invdij;
      rbft (nij) = fcut * inva;
      drbft(nij) = dterm * inva;  // R'(r)
      nij += Nij;
    }
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::radialPhi(t_pod_1d rbft, t_pod_1d drbft, t_pod_1d Phi,
                                          t_pod_1d rbf,t_pod_1d drbf,
                                          int Nij, int ns, int nrbfmax)
{
  Kokkos::parallel_for("radialPhi",
    Kokkos::RangePolicy<DeviceType>(0, Nij * nrbfmax),
    KOKKOS_LAMBDA(const int idx) {
        const int j = idx / Nij;  // col in output
        const int i = idx % Nij;  // row in output

        KK_ACC_FLOAT sum1 = 0.0;
        KK_ACC_FLOAT sum2 = 0.0;

        for (int k = 0; k < ns; ++k) {
            const KK_FLOAT phi_kj = Phi(k + ns * j);
            sum1 += rbft(i + Nij * k) * phi_kj;
            sum2 += drbft(i + Nij * k) * phi_kj;
        }

        rbf(i + Nij * j) = sum1;
        drbf(i + Nij * j) = sum2;
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::radialbasis_hermite(
    t_pod_1d l_rbf, t_pod_1d l_drbf, t_pod_1d l_rr, t_pod_1i ti, t_pod_1i tj, int N)
{
  const int ne = nelements;
  const int nb = nhermite_bins;
  const int nr = nrbfmax;

  auto s_r0   = hermite_r0;
  auto s_invh = hermite_invdr;
  auto s_coef = rbf_hermite_coeffs;

  Kokkos::parallel_for("radialbasis_hermite",
    Kokkos::RangePolicy<DeviceType>(0, N),
    KOKKOS_LAMBDA(const int n) {

      const int itype = ti(n);
      const int jtype = tj(n);
      const int pair  = itype * ne + jtype;

      const KK_FLOAT r = l_rr(n);

      const KK_FLOAT r0   = s_r0(pair);
      const KK_FLOAT invh = s_invh(pair);

      const KK_FLOAT tg = (r - r0) * invh;
      int b = (int) tg;
      if (b < 0) b = 0;
      else if (b > nb-1) b = nb-1;

      const KK_FLOAT t = tg - (KK_FLOAT)b;

      const size_t cbase = ((size_t)pair * (size_t)nb + (size_t)b) * (size_t)nr * 4ull;

      for (int k = 0; k < nr; ++k) {
        const size_t off = cbase + (size_t)4*k;
        const KK_FLOAT c0 = s_coef(off + 0);
        const KK_FLOAT c1 = s_coef(off + 1);
        const KK_FLOAT c2 = s_coef(off + 2);
        const KK_FLOAT c3 = s_coef(off + 3);

        const KK_FLOAT f    = c0 + t * (c1 + t * (c2 + t * c3));
        const KK_FLOAT dfdt = c1 + t * (2.0*c2 + 3.0*c3*t);

        const int idx = n + N*k;
        l_rbf (idx) = f;
        l_drbf(idx) = dfdt * invh;   // R'(r)
      }
    });
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::bspline4(const KK_FLOAT t, const KK_FLOAT dscale,
    KK_FLOAT &B0,  KK_FLOAT &B1,  KK_FLOAT &B2,  KK_FLOAT &B3,
    KK_FLOAT &dB0, KK_FLOAT &dB1, KK_FLOAT &dB2, KK_FLOAT &dB3)
{
  constexpr KK_FLOAT S = (KK_FLOAT) (1.0/6.0);
  const KK_FLOAT t2 = t*t;
  const KK_FLOAT a  = (KK_FLOAT)1.0 - t;
  const KK_FLOAT a2 = a*a;
  const KK_FLOAT t3 = (KK_FLOAT)3.0*t;
  const KK_FLOAT hd = (KK_FLOAT)0.5*dscale;

  B0 = S*a2*a;
  B1 = (KK_FLOAT)4.0*S + (KK_FLOAT)0.5*t2*(t - (KK_FLOAT)2.0);
  B2 = S + (KK_FLOAT)0.5*t*((KK_FLOAT)1.0 + t*a);
  B3 = S*t2*t;

  dB0 = -a2*hd;
  dB1 = t*(t3 - (KK_FLOAT)4.0)*hd;
  dB2 = ((KK_FLOAT)1.0 + t*((KK_FLOAT)2.0 - t3))*hd;
  dB3 = t2*hd;
}

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
KK_FLOAT PairPODKokkos<DeviceType>::sel4(const int i, const KK_FLOAT v0,
    const KK_FLOAT v1, const KK_FLOAT v2, const KK_FLOAT v3)
{
  KK_FLOAT r = (KK_FLOAT)0.0;
  r = (i == 0) ? v0 : r;
  r = (i == 1) ? v1 : r;
  r = (i == 2) ? v2 : r;
  r = (i == 3) ? v3 : r;
  return r;
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::radialbasis_bspline(
    t_pod_1d l_rbf2, t_pod_1d l_drbf2, t_pod_1d l_rr,
    t_pod_1i l_base, t_pod_1i ti, t_pod_1i tj, int N)
{
  const int ne   = nelements;
  const int nbm1 = bs_nb - 1;  // largest admissible bin index
  const int gmax = nrbf2 - 4;  // largest admissible base index
  const int ish  = bs_ileft;   // 0 if left open || 3 if clamped

  auto s_r0   = bs_r0;
  auto s_invh = bs_invh;

  Kokkos::parallel_for("radialbasis_bspline",
    Kokkos::RangePolicy<DeviceType>(0, N),
    KOKKOS_LAMBDA(const int n) {

      const int itype = ti(n);
      const int jtype = tj(n);
      const int pair  = itype * ne + jtype;

      const KK_FLOAT r    = l_rr(n);
      const KK_FLOAT invh = s_invh(pair);

      KK_FLOAT tg = (r - s_r0(pair)) * invh;

      // for r <= r_in the basis is extended as a constant, so forces vanish
      // fold that into the derivative scale instead of zeroing dB afterwards
      const bool     inside = (tg >= (KK_FLOAT)0.0);
      const KK_FLOAT dscale = inside ? invh : (KK_FLOAT)0.0;
      if (!inside) tg = (KK_FLOAT)0.0;

      int b = (int) tg;
      if (b > nbm1) b = nbm1;

      KK_FLOAT B0, B1, B2, B3, dB0, dB1, dB2, dB3;
      bspline4(tg - (KK_FLOAT)b, dscale, B0, B1, B2, B3, dB0, dB1, dB2, dB3);

      const int g0 = b - ish;
      int gb = g0;
      if (gb < 0) gb = 0;
      else if (gb > gmax) gb = gmax;
      l_base(n) = gb;

      const int sh = gb - g0;   // 0 in the interior, [-3,3] at the ends

      // slot s reads the shifted entry of the zero-padded window
      l_rbf2 (n      ) = sel4(sh    , B0, B1, B2, B3);
      l_rbf2 (n +   N) = sel4(sh + 1, B0, B1, B2, B3);
      l_rbf2 (n + 2*N) = sel4(sh + 2, B0, B1, B2, B3);
      l_rbf2 (n + 3*N) = sel4(sh + 3, B0, B1, B2, B3);

      l_drbf2(n      ) = sel4(sh    , dB0, dB1, dB2, dB3);   // R'(r)
      l_drbf2(n +   N) = sel4(sh + 1, dB0, dB1, dB2, dB3);
      l_drbf2(n + 2*N) = sel4(sh + 2, dB0, dB1, dB2, dB3);
      l_drbf2(n + 3*N) = sel4(sh + 3, dB0, dB1, dB2, dB3);
    });
}

template<class DeviceType>
template<int NE, int NR, int MD>
void PairPODKokkos<DeviceType>::radialangularsum_impl(
    t_pod_1d l_sumU, t_pod_1d l_rbf, t_pod_1d l_rhat, t_pod_1i l_tj, t_pod_1i l_numij,
    const int K3, int Ni, int Nij)
{
  using c_1d = Kokkos::View<const KK_FLOAT*, typename t_pod_1d::array_layout, DeviceType,
                            Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
  using c_1i = Kokkos::View<const int*, typename t_pod_1i::array_layout, DeviceType,
                            Kokkos::MemoryTraits<Kokkos::RandomAccess>>;
  const c_1d rbf = l_rbf, uh = l_rhat;
  const c_1i tj = l_tj, numij = l_numij, pqf = pq_f;   // pq_f stride = P3 = MD
  auto sumU = l_sumU;

  Kokkos::parallel_for("RadialAngularSum",
    Kokkos::RangePolicy<DeviceType>(0, Ni*K3), KOKKOS_LAMBDA(const int idx) {
      const int i = idx / K3;
      const int k = idx - i*K3;   // k fastest -> coalesced sumU writes
      int c[MD];
      pod_static_for<MD>([&](auto t) { c[t] = pqf(MD*k + t); });

      const int start = numij(i);
      const int nj    = numij(i+1) - start;

      KK_ACC_FLOAT s[NE][NR];
      pod_static_for<NE>([&](auto e) {
        pod_static_for<NR>([&](auto m) { s[e][m] = 0.0; });
      });

      for (int j = 0; j < nj; ++j) {
        const int n = start + j;
        const KK_FLOAT u = uh(n), v = uh(n + Nij), w = uh(n + 2*Nij);
        KK_ACC_FLOAT a = pod_uvw_factor(c[0], u, v, w);
        pod_static_for<MD-1>([&](auto t) { a *= pod_uvw_factor(c[t + 1], u, v, w); });

        if (NE == 1) {
          pod_static_for<NR>([&](auto m) { s[0][m] += a * rbf(n + Nij*m); });
        } else {
          const int te = tj(n);
          pod_static_for<NE>([&](auto e) {
            if (e == te) {
              pod_static_for<NR>([&](auto m) { s[e][m] += a * rbf(n + Nij*m); });
            }
          });
        }
      }

      pod_static_for<NR>([&](auto m) {
        pod_static_for<NE>([&](auto e) {
          sumU(e + NE*(k + K3*(m + NR*i))) = s[e][m];
        });
      });
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::radialangularsum(
    t_pod_1d l_sumU, t_pod_1d l_rbf, t_pod_1d l_rhat, t_pod_1i l_tj, t_pod_1i l_numij,
    const int l_nelements, const int nrbf3, const int K3, const int P3, int Ni, int Nij)
{
  pod_dispatch_int<1, POD_MAX_NELEM>(l_nelements, [&](auto NE) {
    pod_dispatch_int<1, POD_MAX_NRBF>(nrbf3, [&](auto NR) {
      pod_dispatch_int<1, POD_MAX_DEG>(P3, [&](auto MD) {
        constexpr int ne = decltype(NE)::value, nr = decltype(NR)::value, md = decltype(MD)::value;
        this->template radialangularsum_impl<ne, nr, md>(
            l_sumU, l_rbf, l_rhat, l_tj, l_numij, K3, Ni, Nij);
      });
    });
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobodydesc(t_pod_1d d2, t_pod_1d l_rbf2, t_pod_1i l_idxi, t_pod_1i l_tj,
        int l_nrbf2, const int Ni, const int Nij)
{
  int totalIterations = l_nrbf2 * Nij;
  Kokkos::parallel_for("twobodydesc", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int n = idx / l_nrbf2;
    int m = idx % l_nrbf2;
    Kokkos::atomic_add(&d2(l_idxi(n) + Ni * (m + l_nrbf2 * l_tj(n))), l_rbf2(n + Nij * m));
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobody_forces(
    t_pod_1d fij, t_pod_1d cb2, t_pod_1d l_drbf2, t_pod_1d l_rhat,
    t_pod_1i l_idxi, t_pod_1i l_tj, int l_nrbf2, const int Ni, const int Nij)
{
  Kokkos::parallel_for("twobody_forces",
    Kokkos::RangePolicy<DeviceType>(0, Nij),
    KOKKOS_LAMBDA(const int n) {
      const int ii = l_idxi(n) + Ni * l_nrbf2 * l_tj(n);

      KK_ACC_FLOAT cR = 0.0;
      for (int m = 0; m < l_nrbf2; ++m)
        cR += cb2(ii + Ni * m) * l_drbf2(n + Nij * m);

      const int i1 = 3 * n;
      fij(i1 + 0) = cR * l_rhat(i1 + 0);
      fij(i1 + 1) = cR * l_rhat(i1 + 1);
      fij(i1 + 2) = cR * l_rhat(i1 + 2);
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobodydesc_bspline(t_pod_1d d2, t_pod_1d l_rbf2, t_pod_1i l_base,
        t_pod_1i l_idxi, t_pod_1i l_tj, int l_nrbf2, const int Ni, const int Nij)
{
  Kokkos::parallel_for("twobodydesc_bspline",
    Kokkos::RangePolicy<DeviceType>(0, Nij), KOKKOS_LAMBDA(const int n) {
      const int i0 = l_idxi(n) + Ni * (l_base(n) + l_nrbf2 * l_tj(n));
      for (int s = 0; s < 4; ++s)
        Kokkos::atomic_add(&d2(i0 + Ni * s), l_rbf2(n + Nij * s));
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobody_forces_bspline(
    t_pod_1d fij, t_pod_1d cb2, t_pod_1d l_drbf2, t_pod_1d l_rhat,
    t_pod_1i l_base, t_pod_1i l_idxi, t_pod_1i l_tj, int l_nrbf2, const int Ni, const int Nij)
{
  Kokkos::parallel_for("twobody_forces_bspline",
    Kokkos::RangePolicy<DeviceType>(0, Nij), KOKKOS_LAMBDA(const int n) {
      const int ii = l_idxi(n) + Ni * (l_base(n) + l_nrbf2 * l_tj(n));

      KK_ACC_FLOAT cR = 0.0;
      for (int s = 0; s < 4; ++s)
        cR += cb2(ii + Ni * s) * l_drbf2(n + Nij * s);

      const int i1 = 3 * n;
      fij(i1 + 0) = cR * l_rhat(i1 + 0);
      fij(i1 + 1) = cR * l_rhat(i1 + 1);
      fij(i1 + 2) = cR * l_rhat(i1 + 2);
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::threebodydesc(t_pod_1d d3, t_pod_1d l_sumU, t_pod_1i l_pc3, t_pod_1i l_pn3, t_pod_1i l_p3_active,
        int l_nelements, int l_nrbf3, int l_nabf3_active, int l_K3, const int Ni)
{
  const int totalIterations = Ni * l_nrbf3 * l_nabf3_active;
  if (l_nelements==1) {
    Kokkos::parallel_for("ThreeBodyDesc", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
      const int a =  idx % l_nabf3_active;
      const int m = (idx / l_nabf3_active) % l_nrbf3;
      const int i =  idx / (l_nabf3_active * l_nrbf3);
      int nmi = l_K3 * (m + l_nrbf3 * i);
      int p = l_p3_active(a);
      int n1 = l_pn3(p);
      int n2 = l_pn3(p + 1);
      KK_ACC_FLOAT tmp = 0.0;
      for (int q = n1; q < n2; q++) {
        const KK_FLOAT su = l_sumU(q + nmi);
        tmp += l_pc3(q) * su * su;
      }
      d3(i + Ni * (a + l_nabf3_active * m)) = tmp;
    });
  }
  else {
    Kokkos::parallel_for("ThreeBodyDesc2", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
      const int a =  idx % l_nabf3_active;
      const int m = (idx / l_nabf3_active) % l_nrbf3;
      const int i =  idx / (l_nabf3_active * l_nrbf3);
      int nmi = l_nelements * l_K3 * (m + l_nrbf3 * i);
      int nRA = Ni * l_nrbf3 * l_nabf3_active;
      int k = i + Ni * (a + l_nabf3_active * m);
      int p = l_p3_active(a);
      int n1 = l_pn3(p);
      int n2 = l_pn3(p + 1);
      for (int i1 = 0; i1 < l_nelements; i1++) {
        for (int i2 = i1; i2 < l_nelements; i2++) {
          KK_ACC_FLOAT tmp = 0.0;
          for (int q = n1; q < n2; q++) {
            tmp += l_pc3(q) * l_sumU(i1 + l_nelements * q + nmi) * l_sumU(i2 + l_nelements * q + nmi);
          }
          d3(k) = tmp;
          k += nRA;
        }
      }
    });
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::threebody_forcecoeff(t_pod_1d fb3, t_pod_1d cb3,
    t_pod_1d l_sumU, t_pod_1i l_pc3, t_pod_1i l_pn3, t_pod_1i l_p3_active,
    int l_nelements, int l_nrbf3, int l_nabf3_active, int l_K3, int Ni)
{
  const int totalIterations = Ni * l_nrbf3 * l_nabf3_active;
  if (l_nelements==1) {
    Kokkos::parallel_for("threebody_forcecoeff1", Kokkos::RangePolicy<DeviceType>(0, totalIterations), KOKKOS_LAMBDA(int idx) {
      const int a =  idx % l_nabf3_active;
      const int m = (idx / l_nabf3_active) % l_nrbf3;
      const int i =  idx / (l_nabf3_active * l_nrbf3);
      const int idxU = l_K3 * (m + l_nrbf3*i);
      const KK_FLOAT c3 = (KK_FLOAT)2.0 * cb3(i + Ni*(a + l_nabf3_active*m));
      const int p = l_p3_active(a);
      const int n1 = l_pn3(p);
      const int n2 = l_pn3(p + 1);
      for (int q = n1; q < n2; q++) {
        fb3(q + idxU) += c3 * l_pc3(q) * l_sumU(q + idxU);
      }
    });
  }
  else {
    Kokkos::parallel_for("threebody_forcecoeff2", Kokkos::RangePolicy<DeviceType>(0, totalIterations), KOKKOS_LAMBDA(const int idx) {
      const int a =  idx % l_nabf3_active;
      const int m = (idx / l_nabf3_active) % l_nrbf3;
      const int i =  idx / (l_nabf3_active * l_nrbf3);
      const int N3 = Ni * l_nabf3_active * l_nrbf3;
      const int nim = i + Ni * (a + l_nabf3_active * m);
      const int baseKMI = l_nelements * l_K3 * (m + l_nrbf3 * i);
    
      const int p = l_p3_active(a);
      const int n1  = l_pn3(p);
      const int n2  = l_pn3(p + 1);
      for (int q = n1; q < n2; ++q) {
        const int pk = l_pc3[q];
        const int idxU = baseKMI + l_nelements * q;
        int em = nim;
        for (int i1 = 0; i1 < l_nelements; ++i1) {
          const KK_FLOAT u1 = l_sumU[idxU + i1];
          for (int i2 = i1; i2 < l_nelements; ++i2, em+=N3) {
            const KK_FLOAT w = pk * cb3[em];
            const KK_FLOAT u2 = l_sumU[idxU + i2];
          
            fb3[idxU + i2] += w * u1;
            fb3[idxU + i1] += w * u2;
          }
        }
      }
    });
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::fourbodydesc(t_pod_1d d4,  t_pod_1d l_sumU, t_pod_1i l_pa4, t_pod_1i l_pb4,
    t_pod_1i l_pc4, int l_nelements, int l_nrbf3, int l_nrbf4, int l_nabf4, int l_K3, int l_Q4, int Ni)
{
  const int totalIterations = l_nrbf4 * Ni;
  Kokkos::parallel_for("fourbodydesc", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    const int m = idx % l_nrbf4;
    const int i = idx / l_nrbf4;
    const int NiAR4 = Ni * l_nabf4 * l_nrbf4;
    const int idxU = l_nelements * l_K3 * (m + l_nrbf3 * i);
    const int iNiA4m = i + Ni * l_nabf4 * m;
    for (int p = 0; p < l_nabf4; p++) {
      int n1 = l_pa4(p);
      int n2 = l_pa4(p + 1);
      int k = iNiA4m + Ni * p;
      for (int i1 = 0; i1 < l_nelements; i1++) {
        for (int i2 = i1; i2 < l_nelements; i2++) {
          for (int i3 = i2; i3 < l_nelements; i3++) {
            KK_ACC_FLOAT tmp = 0.0;
            for (int q = n1; q < n2; q++) {
              int c = l_pc4(q);
              int j1 = l_pb4(q);
              int j2 = l_pb4(q + l_Q4);
              int j3 = l_pb4(q + 2 * l_Q4);
              tmp += c * l_sumU(idxU + i1 + l_nelements * j1) * l_sumU(idxU + i2 + l_nelements * j2) * l_sumU(idxU + i3 + l_nelements * j3);
            }
            d4(k) = tmp;
            k += NiAR4;
          }
        }
      }
    }
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::fourbody_forcecoeff(t_pod_1d fb4, t_pod_1d cb4,
    t_pod_1d l_sumU, t_pod_1i l_pa4, t_pod_1i l_pb4, t_pod_1i l_pc4, int l_nelements,
        int l_nrbf3, int l_nrbf4, int l_nabf4, int l_K3, int l_Q4, int Ni)
{
  const int totalIterations = l_nrbf4 * Ni;
  if (l_nelements==1) {
    Kokkos::parallel_for("fourbody_forcecoeff1", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
      int i = idx / l_nrbf4;
      int m = idx % l_nrbf4;
      int idxU = l_K3 * (m + l_nrbf3 * i);
      int iAm = i + Ni * l_nabf4 * m;
      for (int p = 0; p < l_nabf4; p++) {
        int n1 = l_pa4(p);
        int n2 = l_pa4(p + 1);
        KK_FLOAT c4 = cb4(iAm + Ni*p);
        for (int q = n1; q < n2; q++) {
          KK_FLOAT c = c4 * l_pc4(q);
          int j1 = idxU + l_pb4(q);
          int j2 = idxU + l_pb4(q + l_Q4);
          int j3 = idxU + l_pb4(q + 2 * l_Q4);
          KK_FLOAT c1 = l_sumU(j1);
          KK_FLOAT c2 = l_sumU(j2);
          KK_FLOAT c3 = l_sumU(j3);
          fb4[j3] += c * c1 * c2;
          fb4[j2] += c * c1 * c3;
          fb4[j1] += c * c2 * c3;
        }
      }
    });
  }
  else {
    const int N3 = Ni * l_nabf4 * l_nrbf4;
    Kokkos::parallel_for("fourbody_forcecoeff2", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
      int i = idx / l_nrbf4;
      int m = idx % l_nrbf4;
      int idxU = l_nelements * l_K3 * (m + l_nrbf3 * i);
      int iAm = i + Ni * l_nabf4 * m;
      for (int p = 0; p < l_nabf4; p++)  {
        int n1 = l_pa4(p);
        int n2 = l_pa4(p + 1);
        int jpm = iAm + Ni*p;
        for (int q = n1; q < n2; q++) {
          int idx1 = idxU + l_nelements * l_pb4(q);
          int idx2 = idxU + l_nelements * l_pb4(q + l_Q4);
          int idx3 = idxU + l_nelements * l_pb4(q + 2 * l_Q4);
          int c = l_pc4(q);
          int k = jpm;
          for (int i1 = 0; i1 < l_nelements; i1++) {
            KK_FLOAT c1 = l_sumU[idx1 + i1];
            for (int i2 = i1; i2 < l_nelements; i2++) {
              KK_FLOAT c2 = l_sumU[idx2 + i2];
              for (int i3 = i2; i3 < l_nelements; i3++) {
                KK_FLOAT c3 = l_sumU[idx3 + i3];
                KK_FLOAT c4 = c * cb4[k];
                fb4[idx3 + i3] += c4*(c1 * c2);
                fb4[idx2 + i2] += c4*(c1 * c3);
                fb4[idx1 + i1] += c4*(c2 * c3);
                k += N3;
              }
            }
          }
        }
      }
    });
  }
}

struct PODVirial {
  KK_ACC_FLOAT v[6];
};

template<class DeviceType, bool BSPLINE, bool FUSEF, bool VGLOBAL, int NR, int MD>
struct PODAllBodyForceFunctor {
  typedef DeviceType device_type;
  typedef PODVirial value_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef Kokkos::View<KK_FLOAT*, DeviceType> t_1d;
  typedef Kokkos::View<int*, DeviceType> t_1i;

  t_1d rr, rhat, rbf, drbf, drbf2, fc, cb2, fij;
  t_1i idxi, tj, base, ai, aj;
  typename AT::t_kkacc_1d_3 f;
  typename AT::t_kkacc_1d_6 vatom;
  int Ni, Nij, ne, nr2, nr3, P3, K3;
  int vflag_atom;

  KOKKOS_INLINE_FUNCTION
  void init(value_type &dst) const { for (int d = 0; d < 6; ++d) dst.v[d] = 0.0; }

  KOKKOS_INLINE_FUNCTION
  void join(value_type &dst, const value_type &src) const {
    for (int d = 0; d < 6; ++d) dst.v[d] += src.v[d];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int j) const { value_type dummy; compute_pair(j, dummy); }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int j, value_type &vir) const { compute_pair(j, vir); }

  KOKKOS_INLINE_FUNCTION
  void compute_pair(const int j, value_type &vir) const
  {
    const int ia = idxi(j);
    const int te = tj(j);

    const KK_FLOAT dij  = rr(j);
    const KK_FLOAT u    = rhat(j);
    const KK_FLOAT v    = rhat(j +   Nij);
    const KK_FLOAT w    = rhat(j + 2*Nij);
    const KK_FLOAT invr = (KK_FLOAT) 1.0 / dij;

    KK_FLOAT rb[NR], drb[NR];
    pod_static_for<NR>([&](auto m) {
      rb[m]  =  rbf(j + Nij*m);
      drb[m] = drbf(j + Nij*m);
    });

    // two-body (drbf = R'): A00 = 1 -> accumulate to fcAdR
    KK_ACC_FLOAT fcAdR = 0.0;
    if constexpr (BSPLINE) {
      const int i2 = ia + Ni*(base(j) + nr2*te);
      #pragma unroll
      for (int s = 0; s < 4; ++s)
        fcAdR += cb2(i2 + Ni*s) * drbf2(j + Nij*s);
    } else {
      const int i2 = ia + Ni*nr2*te;
      pod_static_for<NR>([&](auto m) {
        fcAdR += cb2(i2 + Ni*m) * drb[m];
      });
      for (int m = NR; m < nr2; ++m)
        fcAdR += cb2(i2 + Ni*m) * drbf2(j + Nij*m);
    }

    // many-body: fcRdA + fcAdR
    // fcAdR = sum dR     u^(p  ) v^(q  ) w^(r  )
    // Du    = sum  R  p  u^(p-1) v^(q  ) w^(r  )
    // Dv    = sum  R  q  u^(p  ) v^(q-1) w^(r  )
    // Dw    = sum  R  r  u^(p  ) v^(q  ) w^(r-1)
    // Nested Horner in decreasing polynomial degree (w -> v -> u)
    const int NeK3 = ne * K3;
    const int ii   = te + NeK3 * NR * ia;

    // w-dir: value(dR) / value(R) / d/du / d/dv / d/dw
    KK_ACC_FLOAT E3 = 0.0, F3 = 0.0, Du = 0.0, Dv = 0.0, Dw = 0.0;

    #pragma unroll
    for (int r = MD; r >= 0; --r) {
      const int QM = MD - r;
      // v-dir: value(dR) / value(R) / d/du / d/dv
      KK_ACC_FLOAT E2 = 0.0, F2 = 0.0, Gu2 = 0.0, Gv2 = 0.0;

      #pragma unroll
      for (int q = QM; q >= 0; --q) {
        const int PM = QM - q;
        // u-dir: value(dR) / value(R) / d/du
        KK_ACC_FLOAT E1 = 0.0, F1 = 0.0, Gu1 = 0.0;

        #pragma unroll
        for (int p = PM; p >= 0; --p) {
          const int b = ii + ne * pod_mono_index(p, q, r);
          KK_ACC_FLOAT fcR = 0.0, fcdR = 0.0;
          pod_static_for<NR>([&](auto m) {
            const KK_FLOAT c = fc(b + NeK3*m);
            fcR  += c * rb[m];
            fcdR += c * drb[m];
          });

          if (p == PM) {                 // leading term: no multiply
            E1 = fcdR;
            F1 = fcR;                    // Gu1 = 0
          } else {                       // derivative first (reuse F1)
            Gu1 = (p == PM-1) ? F1 : Gu1*u + F1;
            F1  = F1*u + fcR;
            E1  = E1*u + fcdR;
          }
        }

        // fold u- into v- polynomial
        if (q == QM) {                   // leading term (PM == 0 -> Gu1 = 0)
          E2 = E1;
          F2 = F1;                       // Gu2, Gv2 = 0
        } else {
          const bool second = (q == QM-1);
          Gv2 = second ? F2  : Gv2*v + F2;    // d/dv, reuse F2
          Gu2 = second ? Gu1 : Gu2*v + Gu1;
          F2  = F2*v + F1;
          E2  = E2*v + E1;
        }
      }

      // fold v- into w- polynomial
      if (r == MD) {                     // leading term (QM == 0 -> Gu2, Gv2 = 0)
        E3 = E2;
        F3 = F2;                         // Du, Dv, Dw = 0
      } else {
        const bool third = (r == MD-1);
        Dw = third ? F3  : Dw*w + F3;    // d/dw, reuse F3
        Du = third ? Gu2 : Du*w + Gu2;
        Dv = third ? Gv2 : Dv*w + Gv2;
        F3 = F3*w + F2;
        E3 = E3*w + E2;
      }
    }
    fcAdR += E3;

    // Jacobian: J = d(u,v,w) / d(x,y,z) = (I - rhat rhat^T) / r
    // f_ij = fc * d(R * A) = J * Dr_hat + fc * A * r_hat
    //      = Dr_hat/r + (fcAdR - (Dr_hat*r_hat)/r) r_hat
    const KK_FLOAT g  = Du*u + Dv*v + Dw*w;
    const KK_FLOAT sr = fcAdR - g*invr;
    const KK_FLOAT fx = Du*invr + sr*u;
    const KK_FLOAT fy = Dv*invr + sr*v;
    const KK_FLOAT fz = Dw*invr + sr*w;

    // force tally
    if constexpr (FUSEF) {
      const int gi = ai(j), gj = aj(j);
      Kokkos::atomic_add(&f(gi,0), fx);
      Kokkos::atomic_add(&f(gi,1), fy);
      Kokkos::atomic_add(&f(gi,2), fz);
      Kokkos::atomic_sub(&f(gj,0), fx);
      Kokkos::atomic_sub(&f(gj,1), fy);
      Kokkos::atomic_sub(&f(gj,2), fz);
    } else {
      fij(3*j+0) = fx;
      fij(3*j+1) = fy;
      fij(3*j+2) = fz;
    }

    // virial
    if (VGLOBAL || vflag_atom) {
      const KK_ACC_FLOAT dx = dij*u, dy = dij*v, dz = dij*w;
      if constexpr (VGLOBAL) {
        vir.v[0] += dx*fx;
        vir.v[1] += dy*fy;
        vir.v[2] += dz*fz;
        vir.v[3] += dx*fy;
        vir.v[4] += dx*fz;
        vir.v[5] += dy*fz;
      }
      if (vflag_atom) {
        const int gi = ai(j), gj = aj(j);
        const KK_ACC_FLOAT h = -0.5;
        const KK_ACC_FLOAT vl[6] = {h*dx*fx, h*dy*fy, h*dz*fz, h*dx*fy, h*dx*fz, h*dy*fz};
        for (int d = 0; d < 6; ++d) {
          Kokkos::atomic_add(&vatom(gi,d), vl[d]);
          Kokkos::atomic_add(&vatom(gj,d), vl[d]);
        }
      }
    }
  }
};

template<class DeviceType>
void PairPODKokkos<DeviceType>::allbody_forces(t_pod_1d l_fij, t_pod_1d l_cb2, int Ni, int Nij)
{
  if (Nij <= 0) return;
  const bool vg = (vflag_global != 0);
  pod_dispatch_int<1, POD_MAX_NRBF>(nrbf3, [&](auto NR) {
    pod_dispatch_int<0, POD_MAX_DEG>(P3, [&](auto MD) {
      constexpr int nr = decltype(NR)::value, md = decltype(MD)::value;
      if (use_bspline) {
        if (vg) this->template allbody_forces_impl<true,  true,  nr, md>(l_fij, l_cb2, Ni, Nij);
        else    this->template allbody_forces_impl<true,  false, nr, md>(l_fij, l_cb2, Ni, Nij);
      } else {
        if (vg) this->template allbody_forces_impl<false, true,  nr, md>(l_fij, l_cb2, Ni, Nij);
        else    this->template allbody_forces_impl<false, false, nr, md>(l_fij, l_cb2, Ni, Nij);
      }
    });
  });
}

template<class DeviceType>
template<bool BSPLINE, bool VGLOBAL, int NR, int MD>
void PairPODKokkos<DeviceType>::allbody_forces_impl(t_pod_1d l_fij, t_pod_1d l_cb2,
                                                    const int Ni, const int Nij)
{
  PODAllBodyForceFunctor<DeviceType, BSPLINE, POD_FUSE_FORCE_TALLY, VGLOBAL, NR, MD> fn;
  fn.rr = rr;       fn.rhat = rhat;
  fn.rbf = rbf;     fn.drbf = drbf;   fn.drbf2 = drbf2;
  fn.fc = forcecoeff;
  fn.cb2 = l_cb2;   fn.fij = l_fij;   // if tally-fused: fij unused (empty)
  fn.idxi = idxi;   fn.tj = tj;       fn.base = bs_base;
  fn.ai = ai;       fn.aj = aj;
  fn.f = f;         fn.vatom = d_vatom;
  fn.Ni = Ni;  fn.Nij = Nij;  fn.ne = nelements;
  fn.nr2 = nrbf2;  fn.nr3 = nrbf3;  fn.P3 = P3;  fn.K3 = K3;
  fn.vflag_atom = vflag_atom;

  if constexpr (VGLOBAL) {
    PODVirial sv;
    Kokkos::parallel_reduce("allbody_forces_virial",
                            Kokkos::RangePolicy<DeviceType>(0, Nij), fn, sv);
    for (int d = 0; d < 6; ++d) virial[d] -= sv.v[d];
  } else {
    Kokkos::parallel_for("allbody_forces", Kokkos::RangePolicy<DeviceType>(0, Nij), fn);
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::crossdesc(t_pod_1d d12, t_pod_1d d1, t_pod_1d d2, t_pod_1i ind1, t_pod_1i ind2, int n12, int Ni)
{
  int totalIterations = n12 * Ni;
  Kokkos::parallel_for("crossdesc", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int n = idx % Ni;
    int i = idx / Ni;

    d12(n + Ni * i) = d1(n + Ni * ind1(i)) * d2(n + Ni * ind2(i));
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::crossdesc_reduction(t_pod_1d cb1, t_pod_1d cb2, t_pod_1d c12, t_pod_1d d1,
        t_pod_1d d2, t_pod_1i ind1, t_pod_1i ind2, int n12, int Ni)
{
  int totalIterations = n12 * Ni;
  Kokkos::parallel_for("crossdesc_reduction", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int n = idx % Ni; // Ni
    int m = idx / Ni; // n12
    int k1 = ind1(m); // dd1
    int k2 = ind2(m); // dd2
    int m1 = n + Ni * k1; // d1
    int m2 = n + Ni * k2; // d2
    KK_FLOAT c = c12(n + Ni * m);
    Kokkos::atomic_add(&cb1(m1), c * d2(m2));
    Kokkos::atomic_add(&cb2(m2), c * d1(m1));
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::set_array_to_zero(t_pod_1d a, int N)
{
  Kokkos::parallel_for("initialize_array", Kokkos::RangePolicy<DeviceType>(0,N), KOKKOS_LAMBDA(int i) {
    a(i) = 0.0;
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::blockatom_base_descriptors(t_pod_1d bd, int Ni, int Nij)
{
  auto begin = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  auto d2 = Kokkos::subview(bd, std::make_pair(0, Ni * nl2));
  auto d3 = Kokkos::subview(bd, std::make_pair(Ni * nl2, Ni * (nl2 + nl3)));
  auto d4 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3), Ni * (nl2 + nl3 + nl4)));
  auto d33 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3 + nl4), Ni * (nl2 + nl3 + nl4 + nl33)));
  auto d34 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3 + nl4 + nl33), Ni * (nl2 + nl3 + nl4 + nl33 + nl34)));
  auto d44 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3 + nl4 + nl33 + nl34), Ni * (nl2 + nl3 + nl4 + nl33 + nl34 + nl44)));

  if (use_hermite) {
    begin = std::chrono::high_resolution_clock::now();
    // Directly fills final basis rbf/drbf (Phi-orthogonalized)
    radialbasis_hermite(rbf, drbf, rr, ti, tj, Nij);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[11] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  } else {
    begin = std::chrono::high_resolution_clock::now();
    // Original analytic path
    radialbasis(rbft, drbft, rr, rin, invrdiff,
                bessel_neg_alpha, bessel_pi_inv_t1, bessel_dx_factor, ti, tj,
                nelements, besseldegree, inversedegree, nbesselpars, Nij);

    radialPhi(rbft, drbft, Phi, rbf, drbf, Nij, ns, nrbfmax);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[11] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  begin = std::chrono::high_resolution_clock::now();
  if (use_bspline) radialbasis_bspline(rbf2, drbf2, rr, bs_base, ti, tj, Nij);
  Kokkos::fence();
  end = std::chrono::high_resolution_clock::now();
  comptime[12] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

  begin = std::chrono::high_resolution_clock::now();
  set_array_to_zero(d2, Ni*nl2);
  if (use_bspline) twobodydesc_bspline(d2, rbf2, bs_base, idxi, tj, nrbf2, Ni, Nij);
  else             twobodydesc(d2, rbf2, idxi, tj, nrbf2, Ni, Nij);
  Kokkos::fence();
  end = std::chrono::high_resolution_clock::now();
  comptime[13] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

  if ((nl3 > 0) && (Nij>1)) {
    begin = std::chrono::high_resolution_clock::now();
    radialangularsum(sumU, rbf, rhat, tj, numij, nelements, nrbf3, K3, P3, Ni, Nij);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[15] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

    begin = std::chrono::high_resolution_clock::now();
    threebodydesc(d3, sumU, pc3, pn3, p3_active, nelements, nrbf3, nabf3_active, K3, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[16] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  if ((nl4 > 0) && (Nij>2)) {
    begin = std::chrono::high_resolution_clock::now();
    fourbodydesc(d4, sumU, pa4, pb4, pc4, nelements, nrbf3, nrbf4, nabf4, K3, Q4, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[17] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  if ((nl33>0) && (Nij>3)) {
    begin = std::chrono::high_resolution_clock::now();
    crossdesc(d33, d3, d3, ind33l, ind33r, nl33, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[18] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  if ((nl34>0) && (Nij>4)) {
    begin = std::chrono::high_resolution_clock::now();
    crossdesc(d34, d3, d4, ind34l, ind34r, nl34, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[19] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  if ((nl44>0) && (Nij>5)) {
    begin = std::chrono::high_resolution_clock::now();
    crossdesc(d44, d4, d4, ind44l, ind44r, nl44, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[20] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::blockatom_base_coefficients(t_pod_1d ei, t_pod_1d cb, t_pod_1d B, int Ni)
{
  auto cefs = coefficients;
  auto tyai = typeai;
  int nDes = Mdesc;
  int nCoeff = nCoeffPerElement;

  Kokkos::parallel_for("atomic_energies", Kokkos::RangePolicy<DeviceType>(0,Ni), KOKKOS_LAMBDA(int n) {
    int nc = nCoeff*tyai[n];
    ei[n] = cefs[0 + nc];
    for (int m=0; m<nDes; m++)
      ei[n] += cefs[1 + m + nc]*B[n + Ni*m];
  });

  int totalIterations = Ni*nDes;
  Kokkos::parallel_for("base_coefficients", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int n = idx % Ni;
    int m = idx / Ni;
    int nc = nCoeff*tyai[n];
    cb[n + Ni*m] = cefs[1 + m + nc];
  });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::cluster_cutoff_poly_sq(const KK_FLOAT pc, const KK_FLOAT inv_rcut2, KK_FLOAT &fcut, KK_FLOAT &dfcut)
{
  const KK_FLOAT u   = (pc * pc) * inv_rcut2;
  const KK_FLOAT u2  = u * u;
  const KK_FLOAT omu = 1.0 - u;

  fcut  = 1.0 - u2 * u * (10.0 - u * (15.0 - 6.0 * u));

  // dfcut := (1/(2pc)) * d(fcut)/dpc
  dfcut = -30.0 * u2 * omu * omu * inv_rcut2;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::cluster_cutoff_hat(
    const KK_FLOAT pc, const KK_FLOAT inv_rcut2, KK_FLOAT &fcut, KK_FLOAT &dfcut)
{
  const KK_FLOAT y2   = (pc * pc) * inv_rcut2; // (pc/r)^2
  const KK_FLOAT omy  = 1.0 - y2 * y2;         // 1-(pc/r)^4
  const KK_FLOAT omy2 = omy * omy;

  fcut = omy2 * omy2; // (1-y^4)^4

  // dfcut := (1/(2pc)) * d(fcut)/dpc
  dfcut = -8.0 * inv_rcut2 * y2 * omy2 * omy;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairPODKokkos<DeviceType>::blockatom_local_environment_descriptors(
    t_pod_1d ei, t_pod_1d cb, t_pod_1d B, int Ni)
{
  const int nCls    = nClusters;
  const int nDes    = Mdesc;
  const int nCoeff  = nCoeffPerElement;
  const int nActMax = nMaxActiveClusters;

  int totIters = Ni * nActMax;
  auto U = Kokkos::subview(pd, std::make_pair(0, Ni));
  auto P = Kokkos::subview(pd, std::make_pair(Ni, Ni + totIters));

  auto proj = Proj;
  auto cent = Centroids;
  auto cefs = coefficients;
  auto tyai = typeai;

  auto ledges   = leftClusterEdges;
  auto redges   = rightClusterEdges;
  auto invlcut2 = invLeftClusterRcut2;
  auto invrcut2 = invRightClusterRcut2;

  auto ks = d_ks;
  auto kn = d_kn;

  // 1) PCA + active cluster search
  // 2) Pk, ei, cp, U
  Kokkos::parallel_for("MoE_clusters_D_dD_Pk_ei_cp_U",
    Kokkos::RangePolicy<DeviceType>(0, Ni), KOKKOS_LAMBDA(const int i) {
      const int typei = tyai[i];
      const int nc    = typei * nCoeff;
      const int ncdt  = typei * nDes;
      const int ncct  = typei * nCls;

      // PCA
      KK_ACC_FLOAT pcai = 0.0;
      for (int m = 0; m < nDes; ++m)
        pcai += proj[ncdt + m] * B[i + Ni * m];

      // Active cluster search
      // lower_bound on redges[ncct + k], ks in [0, nCls-1]
      int lo = 0, hi = nCls;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (redges[ncct + mid] < pcai) lo = mid + 1;
        else                           hi = mid;
      }
      const int kl = lo;

      // upper_bound on ledges[ncct + k], ke in [ks+1, min(ks+nActMax,nCls))
      hi = kl + nActMax;
      if (hi > nCls) hi = nCls;
      lo++;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (ledges[ncct + mid] <= pcai) lo = mid + 1;
        else                            hi = mid;
      }
      const int kni = lo - kl;

      ks[i] = kl;
      kn[i] = kni;   // ke - ks

      // 2) Pk, ei, cp, U

      const int kncct = kl + ncct;
      const int knc   = 1 + nc + kl * nDes;
      
      KK_ACC_FLOAT sumfDi = 0.0;
      KK_ACC_FLOAT S      = 0.0;

      KK_FLOAT D[3], dD_dpca[3];
      KK_FLOAT fcut, dfcut;
      for (int k = 0; k < kni; ++k) {
        const int kg = k + kncct;

        const KK_FLOAT pc        = pcai - cent[kg];
        const KK_FLOAT inv_rcut2 = (pc >= 0.0) ? invrcut2[kg] : invlcut2[kg];

        cluster_cutoff_hat(pc, inv_rcut2, fcut, dfcut);
        //cluster_cutoff_poly_sq(pc, inv_rcut2, fcut, dfcut);

        const KK_FLOAT invDk = 1.0 / (pc * pc + 1e-20);
        const KK_FLOAT fDk   = fcut * invDk;
        const KK_FLOAT dfDkdpc = 2.0 * pc * invDk * (dfcut - fDk);

        D[k] = fDk;
        dD_dpca[k] = dfDkdpc;

        sumfDi += fDk;
        S      += dfDkdpc;
      }
      sumfDi = 1.0 / sumfDi;

      KK_ACC_FLOAT eival = cefs[nc];
      KK_ACC_FLOAT T     = 0.0;
      KK_ACC_FLOAT A     = 0.0;
      for (int k = 0; k < kni; ++k) {
        const int idx = knc + k*nDes;
        KK_ACC_FLOAT sumE = 0.0;
        for (int m = 0; m < nDes; ++m)
          sumE += cefs[m + idx] * B[i + Ni*m];
        
        const KK_FLOAT Pk  = D[k] * sumfDi;
        const KK_FLOAT cpk = sumE * sumfDi;

        P[i + Ni*k] = Pk;

        eival += sumE * Pk;
        T     += cpk * Pk;
        A     += cpk * dD_dpca[k];
      }
      ei[i] = eival;
      U[i]  = A - S * T;
  });

  // 3) Environment force coefficients: O(Mdesc * nActiveClusters)
  totIters = Ni * nDes;
  Kokkos::parallel_for("base_env_coefficients",
    Kokkos::RangePolicy<DeviceType>(0, totIters), KOKKOS_LAMBDA(const int idx) {
      const int i     = idx % Ni;
      const int m     = idx / Ni;
      const int kl    = ks[i];
      const int kni   = kn[i];
      const int typei = tyai[i];
      const int knc   = 1 + m + kl * nDes + nCoeff * typei;

      KK_ACC_FLOAT sum = U[i] * proj[m + nDes*typei];
      for (int k = 0; k < kni; ++k)
        sum += cefs[knc + k*nDes] * P[i + Ni*k];

      cb[i + Ni*m] = sum;
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::blockatom_environment_descriptors(t_pod_1d ei, t_pod_1d cb, t_pod_1d B, int Ni)
{
  auto P = Kokkos::subview(pd, std::make_pair(0, Ni * nClusters));
  auto cp = Kokkos::subview(pd, std::make_pair(Ni * nClusters, 2 * Ni * nClusters));
  auto D = Kokkos::subview(pd, std::make_pair(2 * Ni * nClusters, 3 * Ni * nClusters));
  auto pca = Kokkos::subview(pd, std::make_pair(3 * Ni * nClusters, 3 * Ni * nClusters + Ni * nComponents));
  auto sumD = Kokkos::subview(pd, std::make_pair(3 * Ni * nClusters + Ni * nComponents, 3 * Ni * nClusters + Ni * nComponents + Ni));

  auto proj = Proj;
  auto cent = Centroids;
  auto cefs = coefficients;
  auto tyai = typeai;

  int nCom = nComponents;
  int nCls = nClusters;
  int nDes = Mdesc;
  int nCoeff = nCoeffPerElement;

  int totalIterations = Ni*nCom;
  Kokkos::parallel_for("pca", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int i = idx % Ni;
    int k = idx / Ni;
    KK_FLOAT sum = 0.0;
    int typei = tyai[i];
    for (int m = 0; m < nDes; m++) {
      sum += proj[k + nCom*m + nCom*nDes*typei] * B[i + Ni*m];
    }
    pca[i + Ni*k] = sum;
  });

  totalIterations = Ni*nCls;
  Kokkos::parallel_for("inverse_square_distances", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int i = idx % Ni;
    int j = idx / Ni;
    int typei = tyai[i];
    KK_FLOAT sum = 1e-20;
    for (int k = 0; k < nCom; k++) {
      KK_FLOAT c = cent[k + j * nCom + nCls*nCom*typei];
      KK_FLOAT p = pca[i + Ni*k];
      sum += (p - c) * (p - c);
    }
    D[i + Ni*j] = 1.0 / sum;
  });

  Kokkos::parallel_for("Probabilities", Kokkos::RangePolicy<DeviceType>(0,Ni), KOKKOS_LAMBDA(int i) {
    KK_FLOAT sum = 0;
    for (int j = 0; j < nCls; j++) sum += D[i + Ni*j];
    sumD[i] = sum;
    for (int j = 0; j < nCls; j++) P[i + Ni*j] = D[i + Ni*j]/sum;
  });

  Kokkos::parallel_for("atomic_energies", Kokkos::RangePolicy<DeviceType>(0,Ni), KOKKOS_LAMBDA(int n) {
    int nc = nCoeff*tyai[n];
    ei[n] = cefs[0 + nc];
    for (int k = 0; k<nCls; k++)
      for (int m=0; m<nDes; m++)
        ei[n] += cefs[1 + m + nDes*k + nc]*B[n + Ni*m]*P[n + Ni*k];
  });

  Kokkos::parallel_for("env_coefficients", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int n = idx % Ni;
    int k = idx / Ni;
    int nc = nCoeff*tyai[n];
    KK_FLOAT sum = 0;
    for (int m = 0; m<nDes; m++)
      sum += cefs[1 + m + k*nDes + nc]*B[n + Ni*m];
    cp[n + Ni*k] = sum;
  });

  totalIterations = Ni*nDes;
  Kokkos::parallel_for("base_coefficients", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int n = idx % Ni;
    int m = idx / Ni;
    int nc = nCoeff*tyai[n];
    KK_FLOAT sum = 0.0;
    for (int k = 0; k<nCls; k++)
      sum += cefs[1 + m + k*nDes + nc]*P[n + Ni*k];
    cb[n + Ni*m] = sum;
  });

  Kokkos::parallel_for("base_env_coefficients", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int i = idx % Ni;
    int m = idx / Ni;
    int typei = tyai[i];
    KK_FLOAT S1 = 1/sumD[i];
    KK_FLOAT S2 = sumD[i]*sumD[i];
    KK_FLOAT sum = 0.0;
    for (int j=0; j<nCls; j++) {
      KK_FLOAT dP_dB = 0.0;
      for (int k = 0; k < nCls; k++) {
        KK_FLOAT dP_dD = -D[i + Ni*j] / S2;
        if (k==j) dP_dD += S1;
        KK_FLOAT dD_dB = 0.0;
        KK_FLOAT D2 = 2 * D[i + Ni*k] * D[i + Ni*k];
        for (int n = 0; n < nCom; n++) {
          KK_FLOAT dD_dpca = D2 * (cent[n + k * nCom + nCls*nCom*typei] - pca[i + Ni*n]);
          dD_dB += dD_dpca * proj[n + m * nCom + nCom*nDes*typei];
        }
        dP_dB += dP_dD * dD_dB;
      }
      sum += cp[i + Ni*j]*dP_dB;
    }
    cb[i + Ni*m] += sum;
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::blockatom_energyforce(t_pod_1d l_ei, t_pod_1d l_fij, int Ni, int Nij)
{
  auto begin = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  // calculate base descriptors and their derivatives with respect to atom coordinates
  begin = std::chrono::high_resolution_clock::now();
  blockatom_base_descriptors(bd, Ni, Nij);
  Kokkos::fence();
  end = std::chrono::high_resolution_clock::now();
  comptime[4] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

  begin = std::chrono::high_resolution_clock::now();
  if (localeapod) blockatom_local_environment_descriptors(l_ei, cb, bd, Ni);
  else if (eapod) blockatom_environment_descriptors(l_ei, cb, bd, Ni);
  else            blockatom_base_coefficients(l_ei, cb, bd, Ni);
  Kokkos::fence();
  end = std::chrono::high_resolution_clock::now();
  comptime[5] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

  begin = std::chrono::high_resolution_clock::now();
  auto d3 = Kokkos::subview(bd, std::make_pair(Ni * nl2, Ni * (nl2 + nl3)));
  auto d4 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3), Ni * (nl2 + nl3 + nl4)));
  auto cb2 = Kokkos::subview(cb, std::make_pair(0, Ni * nl2));
  auto cb3 = Kokkos::subview(cb, std::make_pair(Ni * nl2, Ni * (nl2 + nl3)));
  auto cb4 = Kokkos::subview(cb, std::make_pair(Ni * (nl2 + nl3), Ni * (nl2 + nl3 + nl4)));
  auto cb33 = Kokkos::subview(cb, std::make_pair(Ni * (nl2 + nl3 + nl4), Ni * (nl2 + nl3 + nl4 + nl33)));
  auto cb34 = Kokkos::subview(cb, std::make_pair(Ni * (nl2 + nl3 + nl4 + nl33), Ni * (nl2 + nl3 + nl4 + nl33 + nl34)));
  auto cb44 = Kokkos::subview(cb, std::make_pair(Ni * (nl2 + nl3 + nl4 + nl33 + nl34), Ni * (nl2 + nl3 + nl4 + nl33 + nl34 + nl44)));

  if ((nl33>0) && (Nij>3)) crossdesc_reduction(cb3, cb3, cb33, d3, d3, ind33l, ind33r, nl33, Ni);
  if ((nl34>0) && (Nij>4)) crossdesc_reduction(cb3, cb4, cb34, d3, d4, ind34l, ind34r, nl34, Ni);
  if ((nl44>0) && (Nij>5)) crossdesc_reduction(cb4, cb4, cb44, d4, d4, ind44l, ind44r, nl44, Ni);
  Kokkos::fence();
  end = std::chrono::high_resolution_clock::now();
  comptime[6] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

  //// GPUs have large atomblocks. Nij<=1 makes little sense: Nij is the total number of neighbors of all atoms in a block
  //// Would need to treat nij=1 neighbors differently: With numij in the allbody_forces and early return exit
  //if (Nij=1) {
  //  begin = std::chrono::high_resolution_clock::now();
  //  if (use_bspline) twobody_forces_bspline(l_fij, cb2, drbf2, rhat, bs_base, idxi, tj, nrbf2, Ni, Nij);
  //  else             twobody_forces(l_fij, cb2, drbf2, rhat, idxi, tj, nrbf2, Ni, Nij);
  //  Kokkos::fence();
  //  end = std::chrono::high_resolution_clock::now();
  //  comptime[7] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  //}

  // Similarly conditionals Nij>1 and Nij>2 make little sense on GPUs
  if (Nij>1) {
    begin = std::chrono::high_resolution_clock::now();
    set_array_to_zero(forcecoeff, nelements * nrbf3 * K3 * Ni);
    threebody_forcecoeff(forcecoeff, cb3, sumU, pc3, pn3, p3_active, nelements, nrbf3, nabf3_active, K3, Ni);
    if ((nl4 > 0) && (Nij>2)) fourbody_forcecoeff(forcecoeff, cb4, sumU, pa4, pb4, pc4, nelements, nrbf3, nrbf4, nabf4, K3, Q4, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[8] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  begin = std::chrono::high_resolution_clock::now();
  allbody_forces(l_fij, cb2, Ni, Nij);
  Kokkos::fence();
  end = std::chrono::high_resolution_clock::now();
  comptime[10] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::tallyforce(t_pod_1d l_fij, t_pod_1i l_ai, t_pod_1i l_aj, int Nij)
{
  auto l_f = f;
  Kokkos::parallel_for("TallyForce", Kokkos::RangePolicy<DeviceType>(0,Nij), KOKKOS_LAMBDA(int n) {
    int im = l_ai(n);
    int jm = l_aj(n);
    int n3 = 3*n;
    KK_FLOAT fx = l_fij(n3 + 0);
    KK_FLOAT fy = l_fij(n3 + 1);
    KK_FLOAT fz = l_fij(n3 + 2);
    Kokkos::atomic_add(&l_f(im, 0), fx);
    Kokkos::atomic_add(&l_f(im, 1), fy);
    Kokkos::atomic_add(&l_f(im, 2), fz);
    Kokkos::atomic_sub(&l_f(jm, 0), fx);
    Kokkos::atomic_sub(&l_f(jm, 1), fy);
    Kokkos::atomic_sub(&l_f(jm, 2), fz);
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::tallyenergy(t_pod_1d l_ei, int istart, int Ni)
{
  auto l_eatom = d_eatom;

  // For global energy tally
  if (eflag_global) {
    KK_FLOAT local_eng_vdwl = 0.0;
    Kokkos::parallel_reduce("GlobalEnergyTally", Kokkos::RangePolicy<DeviceType>(0,Ni), KOKKOS_LAMBDA(int k, KK_FLOAT& update) {
      update += l_ei(k);
    }, local_eng_vdwl);

    // Update global energy on the host after the parallel region
    eng_vdwl += local_eng_vdwl;
  }

  // For per-atom energy tally
  if (eflag_atom) {
    Kokkos::parallel_for("PerAtomEnergyTally", Kokkos::RangePolicy<DeviceType>(0,Ni), KOKKOS_LAMBDA(int k) {
      l_eatom(istart + k) += l_ei(k);
    });
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::savematrix2binfile(std::string filename, t_pod_1d d_A, int nrows, int ncols)
{
  auto A = Kokkos::create_mirror_view(d_A);
  Kokkos::deep_copy(A, d_A);

  SafeFilePtr fp = fopen(filename.c_str(), "wb");
  KK_FLOAT sz[2];
  sz[0] = (KK_FLOAT) nrows;
  sz[1] = (KK_FLOAT) ncols;
  fwrite( reinterpret_cast<char*>( sz ), sizeof(KK_FLOAT) * (2), 1, fp);
  fwrite( reinterpret_cast<char*>( A.data() ), sizeof(KK_FLOAT) * (nrows*ncols), 1, fp);
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::saveintmatrix2binfile(std::string filename, t_pod_1i d_A, int nrows, int ncols)
{
  auto A = Kokkos::create_mirror_view(d_A);
  Kokkos::deep_copy(A, d_A);

  SafeFilePtr fp = fopen(filename.c_str(), "wb");
  int sz[2];
  sz[0] = nrows;
  sz[1] = ncols;
  fwrite( reinterpret_cast<char*>( sz ), sizeof(int) * (2), 1, fp);
  fwrite( reinterpret_cast<char*>( A.data() ), sizeof(int) * (nrows*ncols), 1, fp);
}

template<class DeviceType>
int PairPODKokkos<DeviceType>::getStreamingProcessorCount()
{
  int streaming_processors = 2048; // Default
  using exec_space_t = typename DeviceType::execution_space;

  int n = exec_space_t{}.concurrency();
  if (n <= 0) n = Kokkos::DefaultExecutionSpace{}.concurrency();
  if (n <= 0) n = streaming_processors;
  if (!host_flag) n = std::max(streaming_processors, n / GPU_CONCURRENCY_DIV);

  return n;
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::savedatafordebugging()
{
  saveintmatrix2binfile("podkktypeai.bin", typeai, ni, 1);
  saveintmatrix2binfile("podkknumij.bin", numij, ni+1, 1);
  saveintmatrix2binfile("podkkai.bin", ai, nij, 1);
  saveintmatrix2binfile("podkkaj.bin", aj, nij, 1);
  saveintmatrix2binfile("podkkti.bin", ti, nij, 1);
  saveintmatrix2binfile("podkktj.bin", tj, nij, 1);
  saveintmatrix2binfile("podkkidxi.bin", idxi, nij, 1);
  savematrix2binfile("podkkrbf.bin", rbf, nrbfmax, nij);
  savematrix2binfile("podkkrbfx.bin", drbf, nrbfmax, nij);
  savematrix2binfile("podkkbd.bin", bd, ni, Mdesc);
  savematrix2binfile("podkkaccU.bin", sumU, nelements * K3 * nrbf3, ni);
  savematrix2binfile("podkkrr.bin", rr, nij, 1);
  savematrix2binfile("podkkuhat.bin", rhat, nij, 3);
  if (!POD_FUSE_FORCE_TALLY) savematrix2binfile("podkkfij.bin", fij, 3, nij);
  savematrix2binfile("podkkei.bin", ei, ni, 1);

  error->all(FLERR, "Save data and stop the run for debugging");
}

/* ----------------------------------------------------------------------
   memory usage of arrays
------------------------------------------------------------------------- */

template<class DeviceType>
double PairPODKokkos<DeviceType>::memory_usage()
{
  double bytes = 0;

  return bytes;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class PairPODKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairPODKokkos<LMPHostType>;
#endif
}
