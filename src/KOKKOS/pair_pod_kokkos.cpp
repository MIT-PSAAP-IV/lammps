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

#include "eapod.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

static constexpr int GPU_CONCURRENCY_DIV = 1;

enum{FS,FS_SHIFTEDSCALED};

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
  utils::logmesg(lmp, "Atom Block Size: {:d}\n", atomBlockSize);
  nAtomBlocks = 0;
  timing = 0;
  for (int i=0; i<100; i++) comptime[i] = 0;
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
    NeighborList(rij, numij, typeai, idxi, ai, aj, ti, tj, gi1, ni);
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

    // tally atomic force to global force
    tallyforce(fij, ai, aj, nij);

    // tally atomic stress
    if (vflag) {
      tallystress(fij, rij, ai, aj, nij);
    }
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
  ns = podptr->ns;      // number of snapshots for radial basis functions
  nl1 = podptr->nl1;  // number of one-body descriptors
  nl2 = podptr->nl2;  // number of two-body descriptors
  nl3 = podptr->nl3;  // number of three-body descriptors
  nl4 = podptr->nl4;  // number of four-body descriptors
  nl33 = podptr->nl33; // number of three-body x three-body descriptors
  nl34 = podptr->nl34; // number of three-body x four-body descriptors
  nl44 = podptr->nl44; // number of four-body x four-body descriptors
  nl = podptr->nl;   // number of local descriptors
  nrbf2 = podptr->nrbf2;
  nrbf3 = podptr->nrbf3;
  nrbf4 = podptr->nrbf4;
  nrbfmax = podptr->nrbfmax; // number of radial basis functions
  nabf3 = podptr->nabf3;     // number of three-body angular basis functions
  nabf4 = podptr->nabf4;     // number of four-body angular basis functions
  nabf3_active = podptr->nabf3_active;     // number of three-body angular basis functions
  nabf4_active = podptr->nabf4_active;     // number of four-body angular basis functions
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
  use_spline   = podptr->use_spline;
  nspline_bins = podptr->nspline_bins;
  //hat_p = podptr->hat_p;    // order of hat basis function
  //hat_q = podptr->hat_q;    // order of hat function
  //hat_q1 = podptr->hat_q1;    // order-1 of hat function
  //h_pq = podptr->h_pq;

  MemKK::realloc_kokkos(coefficients, "pair_pod:coefficients", nCoeffPerElement * nelements);
  auto h_coefficients = Kokkos::create_mirror_view(coefficients);
  for (int i=0; i<nCoeffPerElement * nelements; i++) h_coefficients[i] = podptr->coeff[i];
  Kokkos::deep_copy(coefficients, h_coefficients);

  if (use_spline) {
    const int npair = nelements * nelements;
    const size_t ncoef = (size_t)npair * (size_t)nspline_bins * (size_t)nrbfmax * 4ull;

    MemKK::realloc_kokkos(spline_r0, "pair_pod:spline_r0", npair);
    MemKK::realloc_kokkos(spline_invdr, "pair_pod:spline_invdr", npair);
    MemKK::realloc_kokkos(rbf_spline_coeffs, "pair_pod:rbf_spline_coeffs", ncoef);

    auto h_r0    = Kokkos::create_mirror_view(spline_r0);
    auto h_invdr = Kokkos::create_mirror_view(spline_invdr);
    auto h_coef  = Kokkos::create_mirror_view(rbf_spline_coeffs);

    for (int i = 0; i < npair; ++i) {
      h_r0(i)    = podptr->spline_r0[i];
      h_invdr(i) = podptr->spline_invdr[i];
    }
    for (size_t i = 0; i < ncoef; ++i) {
      h_coef(i) = podptr->rbf_spline_coeffs[i];
    }

    Kokkos::deep_copy(spline_r0, h_r0);
    Kokkos::deep_copy(spline_invdr, h_invdr);
    Kokkos::deep_copy(rbf_spline_coeffs, h_coef);
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
  MemKK::realloc_kokkos(pa4, "pa4", nabf4+1);
  MemKK::realloc_kokkos(pb4, "pb4", Q4*3);
  MemKK::realloc_kokkos(pc4, "pc4", Q4);
  

  auto h_pn3 = Kokkos::create_mirror_view(pn3);
  for (int i=0; i<nabf3+1; i++) h_pn3[i] = podptr->pn3[i];
  Kokkos::deep_copy(pn3, h_pn3);

  auto h_pq_m = Kokkos::create_mirror_view(pq_m);
  auto h_pq_d = Kokkos::create_mirror_view(pq_d);
  for (int i = 0; i < K3; i++) {
    h_pq_m[i] = podptr->pq3[i];
    h_pq_d[i] = podptr->pq3[i + K3];
  }
  Kokkos::deep_copy(pq_m, h_pq_m);
  Kokkos::deep_copy(pq_d, h_pq_d);

  auto h_pc3 = Kokkos::create_mirror_view(pc3);
  for (int i = 0; i < K3; i++) h_pc3[i] = podptr->pc3[i];
  Kokkos::deep_copy(pc3, h_pc3);

  MemKK::realloc_kokkos(p3_active, "pair_pod:p3_active", nabf3_active);
  auto h_p3_active = Kokkos::create_mirror_view(p3_active);
  for (int i=0; i<nabf3_active; i++) h_p3_active[i] = podptr->p3_active[i];
  Kokkos::deep_copy(p3_active, h_p3_active);

  MemKK::realloc_kokkos(p4_active, "pair_pod:p4_active", nabf4_active);
  auto h_p4_active = Kokkos::create_mirror_view(p4_active);
  for (int i=0; i<nabf4_active; i++) h_p4_active[i] = podptr->p4_active[i];
  Kokkos::deep_copy(p4_active, h_p4_active);

  if (nl4 > 0) {
    MemKK::realloc_kokkos(pa4, "pair_pod:pa4", nabf4+1);
    MemKK::realloc_kokkos(pb4, "pair_pod:pb4", Q4*3);
    MemKK::realloc_kokkos(pc4, "pair_pod:pc4", Q4);

    auto h_pa4 = Kokkos::create_mirror_view(pa4);
    for (int i = 0; i < nabf4+1; i++) h_pa4[i] = podptr->pa4[i];
    Kokkos::deep_copy(pa4, h_pa4);

    auto h_pb4 = Kokkos::create_mirror_view(pb4);
    for (int i = 0; i < Q4*3; i++) h_pb4[i] = podptr->pb4[i];
    Kokkos::deep_copy(pb4, h_pb4);

    auto h_pc4 = Kokkos::create_mirror_view(pc4);
    for (int i = 0; i < Q4; i++) h_pc4[i] = podptr->pc4[i];
    Kokkos::deep_copy(pc4, h_pc4);
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
      int envtmpmem = 1 + 2*nMaxActiveClusters;
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
    MemKK::realloc_kokkos(rij, "pair_pod:r_ij", 3 * nijmax);
    MemKK::realloc_kokkos(fij, "pair_pod:f_ij", 3 * nijmax);
    MemKK::realloc_kokkos(idxi, "pair_pod:idxi", nijmax);
    MemKK::realloc_kokkos(ai, "pair_pod:ai", nijmax);
    MemKK::realloc_kokkos(aj, "pair_pod:aj", nijmax);
    MemKK::realloc_kokkos(ti, "pair_pod:ti", nijmax);
    MemKK::realloc_kokkos(tj, "pair_pod:tj", nijmax);
    MemKK::realloc_kokkos(rbf, "pair_pod:rbf", nijmax * nrbfmax);
    MemKK::realloc_kokkos(rbfx, "pair_pod:rbfx", nijmax * nrbfmax);
    MemKK::realloc_kokkos(rbfy, "pair_pod:rbfy", nijmax * nrbfmax);
    MemKK::realloc_kokkos(rbfz, "pair_pod:rbfz", nijmax * nrbfmax);
    int kmax = (K3 > ns) ? K3 : ns;
    MemKK::realloc_kokkos(abf, "pair_pod:abf", nijmax * kmax);
    MemKK::realloc_kokkos(abfx, "pair_pod:abfx", nijmax * kmax);
    MemKK::realloc_kokkos(abfy, "pair_pod:abfy", nijmax * kmax);
    MemKK::realloc_kokkos(abfz, "pair_pod:abfz", nijmax * kmax);
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
void PairPODKokkos<DeviceType>::NeighborList(t_pod_1d l_rij, t_pod_1i l_numij,  t_pod_1i l_typeai,
  t_pod_1i l_idxi, t_pod_1i l_ai, t_pod_1i l_aj, t_pod_1i l_ti, t_pod_1i l_tj, int gi1, int Ni)
{
  // create local shadow views for KOKKOS_LAMBDA to pass them into parallel_for
  auto l_ilist = d_ilist;
  auto l_x = x;
  auto l_numneigh = d_numneigh;
  auto l_neighbors = d_neighbors;
  auto l_map = d_map;
  auto l_type = type;
  auto l_rcutsq = rcutsq;

  Kokkos::parallel_for("NeighborList", typename Kokkos::TeamPolicy<DeviceType>(Ni, Kokkos::AUTO), KOKKOS_LAMBDA(const typename Kokkos::TeamPolicy<DeviceType>::member_type& team) {
    int i = team.league_rank();
    int gi = l_ilist(gi1 + i);
    int itype = l_map(l_type(gi));
    l_typeai(i) = itype;
    KK_FLOAT xi0 = l_x(gi, 0);
    KK_FLOAT xi1 = l_x(gi, 1);
    KK_FLOAT xi2 = l_x(gi, 2);
    int jnum = l_numneigh(gi);
    int nij0 = l_numij(i);
    Kokkos::parallel_scan(Kokkos::TeamThreadRange(team,jnum),
        [&] (const int jj, int& offset, bool final) {
      int gj = l_neighbors(gi,jj);
      gj &= NEIGHMASK;
      int jtype = l_map(l_type(gj));
      KK_FLOAT delx = l_x(gj,0) - xi0;
      KK_FLOAT dely = l_x(gj,1) - xi1;
      KK_FLOAT delz = l_x(gj,2) - xi2;
      KK_FLOAT rsq = delx*delx + dely*dely + delz*delz;
      if (rsq >= l_rcutsq(itype,jtype)) return;
      if (final) {
        int nij1 = nij0 + offset;
        l_rij(nij1 * 3 + 0) = delx;
        l_rij(nij1 * 3 + 1) = dely;
        l_rij(nij1 * 3 + 2) = delz;
        l_idxi(nij1) = i;
        l_ai(nij1) = gi;
        l_aj(nij1) = gj;
        l_ti(nij1) = itype;
        l_tj(nij1) = jtype;
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
    t_pod_1d rbft, t_pod_1d rbftx, t_pod_1d rbfty, t_pod_1d rbftz,
    t_pod_1d l_rij, t_pod_2d l_rin, t_pod_2d l_invrdiff,
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

    const KK_FLOAT xij = l_rij(3*n + 0);
    const KK_FLOAT yij = l_rij(3*n + 1);
    const KK_FLOAT zij = l_rij(3*n + 2);

    const KK_FLOAT dij    = sqrt(xij*xij + yij*yij + zij*zij);
    const KK_FLOAT invdij = 1.0 / dij;

    const KK_FLOAT dr1 = xij * invdij;
    const KK_FLOAT dr2 = yij * invdij;
    const KK_FLOAT dr3 = zij * invdij;

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

      KK_FLOAT ix = xpi;
      for (int i = 1; i <= l_besseldegree; ++i) {
        const KK_FLOAT cosax  = cos(ix);
        const KK_FLOAT inv_i  = 1.0 / KK_FLOAT(i);
        const KK_FLOAT isinax = inv_i * sin(ix);

        const KK_FLOAT rbfv    = bf1 * isinax;
        const KK_FLOAT drbftdr = bg1 * isinax + Kf1dx * cosax;

        rbft (nij) = rbfv;
        rbftx(nij) = drbftdr * dr1;
        rbfty(nij) = drbftdr * dr2;
        rbftz(nij) = drbftdr * dr3;

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

      const KK_FLOAT rbfv    = fcut * inva;
      const KK_FLOAT drbftdr = dterm * inva;

      rbft (nij) = rbfv;
      rbftx(nij) = drbftdr * dr1;
      rbfty(nij) = drbftdr * dr2;
      rbftz(nij) = drbftdr * dr3;
      nij += Nij;
    }
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::matrixMultiply(t_pod_1d a, t_pod_1d b, t_pod_1d c, int r1, int c1, int c2)
{
    Kokkos::parallel_for("MatrixMultiply", Kokkos::RangePolicy<DeviceType>(0,r1 * c2), KOKKOS_LAMBDA(int idx) {
        int j = idx / r1;  // Calculate column index
        int i = idx % r1;  // Calculate row index
        KK_FLOAT sum = 0.0;
        for (int k = 0; k < c1; ++k) {
            sum += a(i + r1*k) * b(k + c1*j);  // Manually calculate the 1D index
        }
        c(i + r1*j) = sum;  // Manually calculate the 1D index for c
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::radialbasis_spline(
    t_pod_1d l_rbf, t_pod_1d l_rbfx, t_pod_1d l_rbfy, t_pod_1d l_rbfz,
    t_pod_1d l_rij, t_pod_1i ti, t_pod_1i tj, int N)
{
  const int ne = nelements;
  const int nb = nspline_bins;
  const int nr = nrbfmax;

  auto s_r0    = spline_r0;
  auto s_invdr = spline_invdr;
  auto s_coef  = rbf_spline_coeffs;

  Kokkos::parallel_for("radialbasis_spline",
    Kokkos::RangePolicy<DeviceType>(0, N),
    KOKKOS_LAMBDA(const int n) {

      const int itype = ti(n);
      const int jtype = tj(n);
      const int pair  = itype * ne + jtype;

      const KK_FLOAT x = l_rij(3*n + 0);
      const KK_FLOAT y = l_rij(3*n + 1);
      const KK_FLOAT z = l_rij(3*n + 2);

      const KK_FLOAT r = sqrt(x*x + y*y + z*z);
      //const KK_FLOAT invr = 1.0 / r;

      const KK_FLOAT r0    = s_r0(pair);
      const KK_FLOAT invdr = s_invdr(pair);
      const KK_FLOAT invdrr = invdr / r;

      const KK_FLOAT tg = (r - r0) * invdr;
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
        const KK_FLOAT h    = dfdt * invdrr;   // f'(r)/r

        const int idx = n + N*k;
        l_rbf (idx) = f;
        l_rbfx(idx) = h * x;
        l_rbfy(idx) = h * y;
        l_rbfz(idx) = h * z;
      }
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::angularbasis(
    t_pod_1d l_abf, t_pod_1d l_abfx, t_pod_1d l_abfy, t_pod_1d l_abfz,
    t_pod_1d l_rij, t_pod_1i l_pq_m, t_pod_1i l_pq_d, int K3, int N)
{
  Kokkos::parallel_for("angularBasis",
    Kokkos::RangePolicy<DeviceType>(0, N),
    KOKKOS_LAMBDA(const int j) {

      const KK_FLOAT x = l_rij(3*j + 0);
      const KK_FLOAT y = l_rij(3*j + 1);
      const KK_FLOAT z = l_rij(3*j + 2);

      const KK_FLOAT xx = x*x, yy = y*y, zz = z*z;
      const KK_FLOAT xy = x*y, xz = x*z, yz = y*z;

      const KK_FLOAT invdij  = 1.0 / sqrt(xx + yy + zz);
      const KK_FLOAT u = x * invdij;
      const KK_FLOAT v = y * invdij;
      const KK_FLOAT w = z * invdij;

      const KK_FLOAT invdij3 = invdij * invdij * invdij;

      const KK_FLOAT dudx = (yy + zz) * invdij3;
      const KK_FLOAT dvdy = (xx + zz) * invdij3;
      const KK_FLOAT dwdz = (xx + yy) * invdij3;

      const KK_FLOAT dudy = -xy * invdij3;
      const KK_FLOAT dudz = -xz * invdij3;
      const KK_FLOAT dvdz = -yz * invdij3;

      const KK_FLOAT dvdx = dudy;
      const KK_FLOAT dwdx = dudz;
      const KK_FLOAT dwdy = dvdz;
      
      l_abf(j)  = 1.0;
      l_abfx(j) = 0.0;
      l_abfy(j) = 0.0;
      l_abfz(j) = 0.0;

      KK_FLOAT c, dcx, dcy, dcz;
      for (int k = 1; k < K3; ++k) {
        const int d = l_pq_d(k);   // 1,2,3

        if (d == 1) {
          c = u; dcx = dudx; dcy = dudy; dcz = dudz;
        } else if (d == 2) {
          c = v; dcx = dvdx; dcy = dvdy; dcz = dvdz;
        } else { // d == 3
          c = w; dcx = dwdx; dcy = dwdy; dcz = dwdz;
        }

        const int in = j + N * l_pq_m(k);
        const KK_FLOAT tm  = l_abf(in);
        const KK_FLOAT tmx = l_abfx(in);
        const KK_FLOAT tmy = l_abfy(in);
        const KK_FLOAT tmz = l_abfz(in);
        
        const int out = j + N * k;
        l_abf(out)  = tm * c;
        l_abfx(out) = tmx * c + tm * dcx;
        l_abfy(out) = tmy * c + tm * dcy;
        l_abfz(out) = tmz * c + tm * dcz;
      }
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::radialangularsum(t_pod_1d l_sumU, t_pod_1d l_rbf, t_pod_1d l_abf, t_pod_1i l_tj,
    t_pod_1i l_numij, const int l_nelements, const int l_nrbf3, const int l_K3, const int Ni, const int Nij)
{
  int totalIterations = l_nrbf3 * l_K3 * Ni;
  if (l_nelements==1) {
    Kokkos::parallel_for("RadialAngularSum", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
      int k = idx % l_K3;
      int nrbfi = idx / l_K3;
      int m = nrbfi % l_nrbf3;
      int i = nrbfi / l_nrbf3;
      int start = l_numij(i);
      int nj = l_numij(i+1) - start;
      int Nijm = start + Nij * m;
      int Nijk = start + Nij * k;

      KK_FLOAT sum = 0.0;
      for (int j=0; j<nj; j++) {
        sum += l_rbf(j + Nijm) * l_abf(j + Nijk);
      }
      l_sumU(idx) = sum;
    });
  }
  else {
    Kokkos::parallel_for("RadialAngularSum", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
      int k = idx % l_K3;
      int nrbfi = idx / l_K3;
      int m = nrbfi % l_nrbf3;
      int i = nrbfi / l_nrbf3;
      int base = l_nelements * idx;
      int start = l_numij(i);
      int nj = l_numij(i+1) - start;
      int Nijm = start + Nij * m;
      int Nijk = start + Nij * k;

      KK_FLOAT tm[6];
      for (int e=0; e<l_nelements; e++) tm[e] = 0;

      for (int j=0; j<nj; j++)
        tm[l_tj(j + start)] += l_rbf(j + Nijm) * l_abf(j + Nijk);
      
      for (int e=0; e<l_nelements; e++) l_sumU(e + base) = tm[e];
    });
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobodydesc(t_pod_1d d2,  t_pod_1d l_rbf, t_pod_1i l_idxi, t_pod_1i l_tj,
        int l_nrbf2, const int Ni, const int Nij)
{
  int totalIterations = l_nrbf2 * Nij;
  Kokkos::parallel_for("twobodydesc", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int n = idx / l_nrbf2;
    int m = idx % l_nrbf2;
    int i2 = n + Nij * m;
    Kokkos::atomic_add(&d2(l_idxi(n) + Ni * (m + l_nrbf2 * l_tj(n))), l_rbf(i2));
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobody_forces(t_pod_1d fij, t_pod_1d cb2, t_pod_1d l_rbfx, t_pod_1d l_rbfy,
        t_pod_1d l_rbfz, t_pod_1i l_idxi, t_pod_1i l_tj, int l_nrbf2, const int Ni, const int Nij)
{
  int totalIterations = l_nrbf2 * Nij;
  Kokkos::parallel_for("twobody_forces", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int n = idx / l_nrbf2;
    int m = idx % l_nrbf2;
    int i2 = n + Nij * m;
    int i1 = 3*n;
    KK_FLOAT c = cb2(l_idxi(n) + Ni*m + Ni*l_nrbf2*l_tj(n));
    Kokkos::atomic_add(&fij(0 + i1), c*l_rbfx(i2));
    Kokkos::atomic_add(&fij(1 + i1), c*l_rbfy(i2));
    Kokkos::atomic_add(&fij(2 + i1), c*l_rbfz(i2));
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::threebodydesc(t_pod_1d d3, t_pod_1d l_sumU, t_pod_1i l_pc3, t_pod_1i l_pn3, t_pod_1i l_p3_active,
        int l_nelements, int l_nrbf3, int l_nabf3_active, int l_K3, const int Ni)
{
  int totalIterations = Ni * l_nrbf3;
  Kokkos::parallel_for("ThreeBodyDesc", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int m = idx % l_nrbf3;
    int i = idx / l_nrbf3;
    int nmi = l_nelements * l_K3 * (m + l_nrbf3 * i);
    int nRA = Ni * l_nrbf3 * l_nabf3_active;
    int iAm = i + Ni * l_nabf3_active * m;
    for (int a = 0; a < l_nabf3_active; ++a) {
      int p = l_p3_active(a);
      int n1 = l_pn3(p);
      int n2 = l_pn3(p + 1);
      int k = iAm + Ni * a;
      for (int i1 = 0; i1 < l_nelements; i1++) {
        for (int i2 = i1; i2 < l_nelements; i2++) {
          KK_FLOAT tmp=0;
          for (int q = n1; q < n2; q++) {
            tmp += l_pc3(q) * l_sumU(i1 + l_nelements * q + nmi) * l_sumU(i2 + l_nelements * q + nmi);
          }
          d3(k) = tmp;
          k += nRA;
        }
      }
    }
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::threebody_forcecoeff(t_pod_1d fb3, t_pod_1d cb3,
    t_pod_1d l_sumU, t_pod_1i l_pc3, t_pod_1i l_pn3, t_pod_1i l_p3_active,
    int l_nelements, int l_nrbf3, int l_nabf3_active, int l_K3, int Ni)
{
  int totalIterations = Ni * l_nrbf3;
  if (l_nelements==1) {
    Kokkos::parallel_for("threebody_forcecoeff1", Kokkos::RangePolicy<DeviceType>(0, totalIterations), KOKKOS_LAMBDA(int idx) {
      const int i = idx / l_nrbf3;
      const int m = idx % l_nrbf3;
      const int nib3 = i + Ni*l_nabf3_active*m;
      const int idxU = l_K3 * (m + l_nrbf3*i);
      for (int a = 0; a < l_nabf3_active; ++a) {
        const KK_FLOAT c3 = 2.0 * cb3(nib3 + Ni*a);
        const int p = l_p3_active(a);
        const int n1 = l_pn3(p);
        const int n2 = l_pn3(p + 1);
        for (int q = n1; q < n2; q++) {
          fb3(q + idxU) += c3 * l_pc3(q) * l_sumU(q + idxU);
        }
      }
    });
  }
  else {
    Kokkos::parallel_for("threebody_forcecoeff2", Kokkos::RangePolicy<DeviceType>(0, totalIterations), KOKKOS_LAMBDA(const int idx) {
      const int i = idx / l_nrbf3;
      const int m = idx % l_nrbf3;
      const int N3 = Ni * l_nabf3_active * l_nrbf3;
      const int nim = i + Ni * l_nabf3_active * m;
      const int baseKMI = l_nelements * l_K3 * (m + l_nrbf3 * i);
    
      for (int a = 0; a < l_nabf3_active; ++a) {
        const int p = l_p3_active(a);
        const int n1  = l_pn3(p);
        const int n2  = l_pn3(p + 1);
        const int jmp = nim + Ni * a;
        for (int q = n1; q < n2; ++q) {
          const KK_FLOAT pk = l_pc3[q];
          const int idxU = baseKMI + l_nelements * q;
          // em = l_elemindex(i1, i2)
          int em = jmp;
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
      }
    });
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::fourbodydesc(t_pod_1d d4,  t_pod_1d l_sumU, t_pod_1i l_pa4, t_pod_1i l_pb4,
    t_pod_1i l_pc4, int l_nelements, int l_nrbf3, int l_nrbf4, int l_nabf4, int l_K3, int l_Q4, int Ni)
{
  int totalIterations = l_nrbf4 * Ni;
  Kokkos::parallel_for("fourbodydesc", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int m = idx % l_nrbf4;
    int i = idx / l_nrbf4;
    int idxU = l_nelements * l_K3 * (m + l_nrbf3 * i);
    for (int p = 0; p < l_nabf4; p++) {
      int n1 = l_pa4(p);
      int n2 = l_pa4(p + 1);
      int k = 0;
      for (int i1 = 0; i1 < l_nelements; i1++) {
        for (int i2 = i1; i2 < l_nelements; i2++) {
          for (int i3 = i2; i3 < l_nelements; i3++) {
            KK_FLOAT tmp = 0.0;
            for (int q = n1; q < n2; q++) {
              int c = l_pc4(q);
              int j1 = l_pb4(q);
              int j2 = l_pb4(q + l_Q4);
              int j3 = l_pb4(q + 2 * l_Q4);
              tmp += c * l_sumU(idxU + i1 + l_nelements * j1) * l_sumU(idxU + i2 + l_nelements * j2) * l_sumU(idxU + i3 + l_nelements * j3);
            }
            int kk = p + l_nabf4 * m + l_nabf4 * l_nrbf4 * k;
            d4(i + Ni * kk) = tmp;
            k += 1;
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
  int totalIterations = l_nrbf4 * Ni;
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
    int N3 = Ni * l_nabf4 * l_nrbf4;
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
          KK_FLOAT c = l_pc4(q);
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

template<class DeviceType>
void PairPODKokkos<DeviceType>::allbody_forces(t_pod_1d fij, t_pod_1d l_forcecoeff, t_pod_1d l_rbf, t_pod_1d l_rbfx,
    t_pod_1d l_rbfy, t_pod_1d l_rbfz, t_pod_1d l_abf, t_pod_1d l_abfx, t_pod_1d l_abfy, t_pod_1d l_abfz,
    t_pod_1i l_idxi, t_pod_1i l_tj, int l_nelements, int l_nrbf3, int l_K3, int Nij)
{
  int totalIterations = l_nrbf3 * Nij;
  Kokkos::parallel_for("allbody_forces", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    const int j = idx / l_nrbf3;
    const int m = idx % l_nrbf3;
    const int i2 = l_tj(j) + l_nelements * l_K3 * (m + l_nrbf3*l_idxi(j));
    const int idxR = j + Nij * m;
    const KK_FLOAT rbfBase = l_rbf(idxR);
    const KK_FLOAT rbfxBase = l_rbfx(idxR);
    const KK_FLOAT rbfyBase = l_rbfy(idxR);
    const KK_FLOAT rbfzBase = l_rbfz(idxR);
    KK_FLOAT fx = 0;
    KK_FLOAT fy = 0;
    KK_FLOAT fz = 0;
    for (int k = 0; k < l_K3; k++) {
      const int idxA = j + Nij * k;
      const KK_FLOAT abfA = l_abf(idxA);
      const KK_FLOAT abfxA = l_abfx(idxA);
      const KK_FLOAT abfyA = l_abfy(idxA);
      const KK_FLOAT abfzA = l_abfz(idxA);
      const KK_FLOAT fc = l_forcecoeff(i2 + l_nelements * k);
      fx += fc * (abfxA * rbfBase + rbfxBase * abfA);
      fy += fc * (abfyA * rbfBase + rbfyBase * abfA);
      fz += fc * (abfzA * rbfBase + rbfzBase * abfA);
    }
    int ii = 3 * j;
    Kokkos::atomic_add(&fij(0 + ii), fx);
    Kokkos::atomic_add(&fij(1 + ii), fy);
    Kokkos::atomic_add(&fij(2 + ii), fz);
  });
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

  if (use_spline) {
    // Directly fills final basis rbf/rbfx/rbfy/rbfz (Phi-orthogonalized)
    begin = std::chrono::high_resolution_clock::now();
    radialbasis_spline(rbf, rbfx, rbfy, rbfz, rij, ti, tj, Nij);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[10] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  } else {
    // Original analytic path
    begin = std::chrono::high_resolution_clock::now();
    radialbasis(abf, abfx, abfy, abfz,
                rij, rin, invrdiff,
                bessel_neg_alpha, bessel_pi_inv_t1, bessel_dx_factor,
                ti, tj,
                nelements, besseldegree, inversedegree, nbesselpars, Nij);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[10] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
    
    begin = std::chrono::high_resolution_clock::now();
    matrixMultiply(abf,  Phi, rbf,  Nij, ns, nrbfmax);
    matrixMultiply(abfx, Phi, rbfx, Nij, ns, nrbfmax);
    matrixMultiply(abfy, Phi, rbfy, Nij, ns, nrbfmax);
    matrixMultiply(abfz, Phi, rbfz, Nij, ns, nrbfmax);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[11] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  begin = std::chrono::high_resolution_clock::now();
  set_array_to_zero(d2, Ni*nl2);
  twobodydesc(d2, rbf, idxi, tj, nrbf2, Ni, Nij);
  Kokkos::fence();
  end = std::chrono::high_resolution_clock::now();
  comptime[12] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

  if ((nl3 > 0) && (Nij>1)) {
    begin = std::chrono::high_resolution_clock::now();
    angularbasis(abf, abfx, abfy, abfz, rij, pq_m, pq_d, K3, Nij);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[13] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

    begin = std::chrono::high_resolution_clock::now();
    radialangularsum(sumU, rbf, abf, tj, numij, nelements, nrbf3, K3, Ni, Nij);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[14] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

    begin = std::chrono::high_resolution_clock::now();
    threebodydesc(d3, sumU, pc3, pn3, p3_active, nelements, nrbf3, nabf3_active, K3, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[15] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  if ((nl4 > 0) && (Nij>2)) {
    begin = std::chrono::high_resolution_clock::now();
    fourbodydesc(d4, sumU, pa4, pb4, pc4, nelements, nrbf3, nrbf4, nabf4, K3, Q4, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[16] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  if ((nl33>0) && (Nij>3)) {
    begin = std::chrono::high_resolution_clock::now();
    crossdesc(d33, d3, d3, ind33l, ind33r, nl33, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[17] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  if ((nl34>0) && (Nij>4)) {
    begin = std::chrono::high_resolution_clock::now();
    crossdesc(d34, d3, d4, ind34l, ind34r, nl34, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[18] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
  }

  if ((nl44>0) && (Nij>5)) {
    begin = std::chrono::high_resolution_clock::now();
    crossdesc(d44, d4, d4, ind44l, ind44r, nl44, Ni);
    Kokkos::fence();
    end = std::chrono::high_resolution_clock::now();
    comptime[19] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;
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
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::cluster_cutoff_poly_sq(const KK_FLOAT pc, const KK_FLOAT inv_rcut2, KK_FLOAT &fcut, KK_FLOAT &dfcut)
{
  const KK_FLOAT pc_inv_rcut2 = pc * inv_rcut2;
  const KK_FLOAT u            = pc * pc_inv_rcut2;      // u = pc^2 * inv_rcut2
  const KK_FLOAT u2           = u * u;
  const KK_FLOAT omu          = 1.0 - u;

  fcut  = 1.0 - u2 * u * (10.0 - u * (15.0 - 6.0 * u));

  // dfcut/dpc = -60 u^2 (1-u)^2 * (pc * inv_rcut2)
  dfcut = -60.0 * u2 * omu * omu * pc_inv_rcut2;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::cluster_cutoff_hat(const KK_FLOAT pc, const KK_FLOAT inv_rcut2, KK_FLOAT &fcut, KK_FLOAT &dfcut)
{
  constexpr int    p      = 2;
  constexpr int    q      = 4;
  constexpr KK_FLOAT TWO_PQ = -2.0 * p * q;

  KK_FLOAT pc_inv_rcut2 = pc * inv_rcut2;
  KK_FLOAT y2     = pc * pc_inv_rcut2;
  KK_FLOAT y2pm1  = Kokkos::pow(y2, p - 1);
  KK_FLOAT y2p    = y2pm1 * y2;
  KK_FLOAT omy    = 1.0 - y2p;
  KK_FLOAT omq1   = Kokkos::pow(omy, q - 1);

  fcut  = omq1 * omy;
  dfcut = TWO_PQ * pc_inv_rcut2 * y2pm1 * omq1;
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
  auto D       = Kokkos::subview(pd, std::make_pair(0, totIters));
  auto dD_dpca = Kokkos::subview(pd, std::make_pair(totIters, 2*totIters));
  auto pca     = Kokkos::subview(pd, std::make_pair(2*totIters, 2*totIters + Ni));
  auto U       = pca; // reuse buffer

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
  Kokkos::parallel_for("pca_and_active_clusters",
    Kokkos::RangePolicy<DeviceType>(0, Ni),
    KOKKOS_LAMBDA(const int i) {
      const int typei = tyai[i];
      const int ncdt  = typei * nDes;
      const int ncct  = typei * nCls;

      KK_FLOAT pcai = 0.0;
      for (int m = 0; m < nDes; ++m) {
        pcai += proj[ncdt + m] * B[i + Ni * m];
      }
      pca[i] = pcai;

      // lower_bound on redges[ncct + k], ks in [0, nCls-1]
      int lo = 0, hi = nCls;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (redges[ncct + mid] < pcai) lo = mid + 1;
        else                           hi = mid;
      }
      const int ksi = lo;

      // upper_bound on ledges[ncct + k], ke in [ks+1, min(ks+nActMax,nCls))
      hi = ksi + nActMax;
      if (hi > nCls) hi = nCls;
      lo++;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (ledges[ncct + mid] <= pcai) lo = mid + 1;
        else                            hi = mid;
      }

      ks[i] = ksi;
      kn[i] = lo - ksi;   // ke - ks
    }
  );

  // 2) D, dD/dpca, Pk, ei, cp, U
  Kokkos::parallel_for("D_dD_Pk_ei_cp_U",
    Kokkos::RangePolicy<DeviceType>(0, Ni), KOKKOS_LAMBDA(const int i) {
      const int kl    = ks[i];
      const int kni   = kn[i];
      const int typei = tyai[i];

      const int nc    = nCoeff * typei;
      const int kncct = kl + nCls * typei;
      const int knc   = 1 + nc + kl * nDes;
      
      KK_FLOAT sumfDi = 0.0;
      KK_FLOAT S      = 0.0;

      const KK_FLOAT pcai = pca[i];
      KK_FLOAT fcut, dfcut;
      for (int k = 0; k < kni; ++k) {
        const int kg = k + kncct;

        const KK_FLOAT pc        = pcai - cent[kg];
        const KK_FLOAT inv_rcut2 = (pc >= 0.0) ? invrcut2[kg] : invlcut2[kg];

        cluster_cutoff_hat(pc, inv_rcut2, fcut, dfcut);
        //cluster_cutoff_poly_sq(pc, inv_rcut2, fcut, dfcut);

        const KK_FLOAT Dk_raw = 1.0 / (pc * pc + 1e-20);
        const KK_FLOAT Dval   = fcut * Dk_raw;
        const KK_FLOAT dDval  = Dk_raw * (dfcut - 2.0 * pc * Dval);

        const int idx = i + Ni * k;
        D[idx]       = Dval;
        dD_dpca[idx] = dDval;

        sumfDi += Dval;
        S      += dDval;
      }
      sumfDi = 1.0 / sumfDi;

      KK_FLOAT eival = cefs[nc];
      KK_FLOAT T     = 0.0;
      KK_FLOAT A     = 0.0;
      for (int k = 0; k < kni; ++k) {
        const int idx = i + Ni * k;
        KK_FLOAT sumE = 0.0;
        for (int m = 0; m < nDes; ++m)
          sumE += cefs[m + k*nDes + knc] * B[i + Ni*m];
        
        const KK_FLOAT Pk  = D[idx] * sumfDi;
        const KK_FLOAT cpk = sumE * sumfDi;

        D[idx] = Pk;

        eival += sumE * Pk;
        T     += cpk * Pk;
        A     += cpk * dD_dpca[idx];
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
      const int ncdt  = nDes * typei;
      const int knc   = 1 + kl * nDes + nCoeff * typei;

      KK_FLOAT sum = U[i] * proj[m + ncdt];
      for (int k = 0; k < kni; ++k)
        sum += cefs[m + k*nDes + knc] * D[i + Ni*k];

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

  if ((nl33>0) && (Nij>3)) {
    crossdesc_reduction(cb3, cb3, cb33, d3, d3, ind33l, ind33r, nl33, Ni);
  }
  if ((nl34>0) && (Nij>4)) {
    crossdesc_reduction(cb3, cb4, cb34, d3, d4, ind34l, ind34r, nl34, Ni);
  }
  if ((nl44>0) && (Nij>5)) {
    crossdesc_reduction(cb4, cb4, cb44, d4, d4, ind44l, ind44r, nl44, Ni);
  }
  Kokkos::fence();
  end = std::chrono::high_resolution_clock::now();
  comptime[6] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

  begin = std::chrono::high_resolution_clock::now();
  set_array_to_zero(l_fij, 3*Nij);
  if (Nij>0) twobody_forces(l_fij, cb2, rbfx, rbfy, rbfz, idxi, tj, nrbf2, Ni, Nij);
  Kokkos::fence();
  end = std::chrono::high_resolution_clock::now();
  comptime[7] += std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count()/1e6;

  set_array_to_zero(forcecoeff, nelements * nrbf3 * K3 * Ni);
  if ((nl3 > 0) && (Nij>1)) threebody_forcecoeff(forcecoeff, cb3, sumU,
          pc3, pn3, p3_active, nelements, nrbf3, nabf3_active, K3, Ni);
  if ((nl4 > 0) && (Nij>2)) fourbody_forcecoeff(forcecoeff, cb4, sumU,
      pa4, pb4, pc4, nelements, nrbf3, nrbf4, nabf4, K3, Q4, Ni);
  if ((nl3 > 0) && (Nij>1)) allbody_forces(l_fij, forcecoeff, rbf, rbfx, rbfy, rbfz, abf, abfx, abfy, abfz,
          idxi, tj, nelements, nrbf3, K3, Nij);
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
void PairPODKokkos<DeviceType>::tallystress(t_pod_1d l_fij, t_pod_1d l_rij, t_pod_1i l_ai, t_pod_1i l_aj, int Nij)
{
  auto l_vatom = d_vatom;

  if (vflag_global) {
    for (int j=0; j<3; j++) {
      KK_FLOAT sum = 0.0;
      Kokkos::parallel_reduce("GlobalStressTally", Kokkos::RangePolicy<DeviceType>(0,Nij), KOKKOS_LAMBDA(int k, KK_FLOAT& update) {
        int k3 = 3*k;
        update += l_rij(j + k3) * l_fij(j + k3);
      }, sum);
      virial[j] -= sum;
    }

    KK_FLOAT sum = 0.0;
    Kokkos::parallel_reduce("GlobalStressTally", Kokkos::RangePolicy<DeviceType>(0,Nij), KOKKOS_LAMBDA(int k, KK_FLOAT& update) {
      int k3 = 3*k;
      update += l_rij(k3) * l_fij(1 + k3);
    }, sum);
    virial[3] -= sum;

    sum = 0.0;
    Kokkos::parallel_reduce("GlobalStressTally", Kokkos::RangePolicy<DeviceType>(0,Nij), KOKKOS_LAMBDA(int k, KK_FLOAT& update) {
      int k3 = 3*k;
      update += l_rij(k3) * l_fij(2 + k3);
    }, sum);
    virial[4] -= sum;

    sum = 0.0;
    Kokkos::parallel_reduce("GlobalStressTally", Kokkos::RangePolicy<DeviceType>(0,Nij), KOKKOS_LAMBDA(int k, KK_FLOAT& update) {
      int k3 = 3*k;
      update += l_rij(1+k3) * l_fij(2+k3);
    }, sum);
    virial[5] -= sum;
  }

  if (vflag_atom) {
    Kokkos::parallel_for("PerAtomStressTally", Kokkos::RangePolicy<DeviceType>(0,Nij), KOKKOS_LAMBDA(int k) {
      int i = l_ai(k);
      int j = l_aj(k);
      int k3 = 3*k;
      KK_FLOAT v_local[6];
      v_local[0] = -l_rij(k3) * l_fij(k3 + 0);
      v_local[1] = -l_rij(k3 + 1) * l_fij(k3 + 1);
      v_local[2] = -l_rij(k3 + 2) * l_fij(k3 + 2);
      v_local[3] = -l_rij(k3 + 0) * l_fij(k3 + 1);
      v_local[4] = -l_rij(k3 + 0) * l_fij(k3 + 2);
      v_local[5] = -l_rij(k3 + 1) * l_fij(k3 + 2);

      for (int d = 0; d < 6; ++d) {
        Kokkos::atomic_add(&l_vatom(i, d), 0.5 * v_local[d]);
      }

      for (int d = 0; d < 6; ++d) {
        Kokkos::atomic_add(&l_vatom(j, d), 0.5 * v_local[d]);
      }

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

  // Your MI300/APU rule: normalize large GPU concurrency by 32
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
  savematrix2binfile("podkkrbfx.bin", rbfx, nrbfmax, nij);
  savematrix2binfile("podkkrbfy.bin", rbfy, nrbfmax, nij);
  savematrix2binfile("podkkrbfz.bin", rbfz, nrbfmax, nij);
  int kmax = (K3 > ns) ? K3 : ns;
  savematrix2binfile("podkkabf.bin", abf,   kmax, nij);
  savematrix2binfile("podkkabfx.bin", abfx, kmax, nij);
  savematrix2binfile("podkkabfy.bin", abfy, kmax, nij);
  savematrix2binfile("podkkabfz.bin", abfz, kmax, nij);
  savematrix2binfile("podkkbd.bin", bd, ni, Mdesc);
  savematrix2binfile("podkkaccU.bin", sumU, nelements * K3 * nrbfmax, ni);
  savematrix2binfile("podkkrij.bin", rij, 3, nij);
  savematrix2binfile("podkkfij.bin", fij, 3, nij);
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
