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

#include <cstdlib>
#include <cstring>
#include <utility>
#include <type_traits>
#include <algorithm>
#include <vector>

#include "eapod.h"
#include "pair_pod_kokkos_hip.h"

using namespace LAMMPS_NS;
using namespace MathConst;

static constexpr int GPU_CONCURRENCY_DIV = 1;

static constexpr int POD_MAX_NRBF  = 8;   // max nrbf3 held in registers
static constexpr int POD_MAX_NELEM = 2;   // max nelements in the fused sum
static constexpr int POD_MAX_DEG   = 4;   // max monomial degree (P3)

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

// variants of the native HIP kernels
enum { HIP_RADIAL = 1,     // fused radial basis and projection
       HIP_ANGULAR = 2,    // angular basis computed on the fly
       HIP_MFMA = 4,       // matrix cores for the radial-angular sums
       HIP_MFMA_FORCE = 8,      // matrix cores for the force coefficient products
       HIP_DESC = 16,           // fused 3- and 4-body descriptors
       HIP_TALLY = 32,          // per-atom reductions in the force tally and 2-body descriptors
       HIP_PORT = 64,           // native versions of all other kernels
       HIP_FUSE = 128,          // radial Hermite spline evaluated inside its consumers
       HIP_FTALLY = 256,        // 2-body forces and force/virial tally inside the force kernel
       HIP_FUSE_FC = 512,       // 3- and 4-body force coefficients inside the force kernel
       HIP_FUSE_NEIGH = 1024,   // pair list in one pass, without reading its size back
       HIP_FIXED_PAIRS = 4096,  // ... with a fixed stride per atom, without a scan
       HIP_PAIR_ALIGN = 8192 };   // ... rounded up to a multiple of hip_palign pairs

#ifdef KOKKOS_ENABLE_HIP
namespace {
// views used by the native neighbor kernels, as raw pointers and strides
template <class VI, class VN, class VT, class VM, class VX, class VC>
PODHIP::NeighArgs pod_neigh_args(const VI &ilist, const VI &numneigh, const VN &neighbors,
                                 const VT &type, const VM &map, const VX &x, const VC &rcutsq)
{
  PODHIP::NeighArgs a;
  a.ilist = ilist.data();
  a.numneigh = numneigh.data();
  a.neighbors = neighbors.data();
  a.type = type.data();
  a.map = map.data();
  a.ns0 = neighbors.stride(0);
  a.ns1 = neighbors.stride(1);
  a.xs0 = x.stride(0);
  a.xs1 = x.stride(1);
  a.cs0 = rcutsq.stride(0);
  a.cs1 = rcutsq.stride(1);
  return a;
}

// arguments of the native 3- and 4-body descriptor kernels
template <class VD, class VI>
PODHIP::DescArgs<KK_FLOAT> pod_desc_args(const VD &d3, const VD &d4, const VI &pc3, const VI &pn3,
                                         const VI &p3_active, const VI &qrange3, const VI &pa4,
                                         const VI &pb4, const VI &pc4, int nelements, int nrbf3,
                                         int nabf3_active, int nrbf4, int nabf4, int K3, int Q4,
                                         int Ni, bool do4)
{
  PODHIP::DescArgs<KK_FLOAT> g;
  g.d3 = d3.data();
  g.d4 = d4.data();
  g.pc3 = pc3.data();
  g.pn3 = pn3.data();
  g.p3_active = p3_active.data();
  g.qrange3 = reinterpret_cast<const int2 *>(qrange3.data());
  g.pa4 = pa4.data();
  g.pb4 = pb4.data();
  g.pc4 = pc4.data();
  g.nelements = nelements;
  g.nrbf3 = nrbf3;
  g.nabf3_active = nabf3_active;
  g.nrbf4 = nrbf4;
  g.nabf4 = nabf4;
  g.K3 = K3;
  g.Q4 = Q4;
  g.Ni = Ni;
  g.do4 = do4 ? 1 : 0;
  return g;
}

// distances and unit vectors of the pairs of the current block
template <class VD>
PODHIP::PairGeom<KK_FLOAT> pod_geom(const VD &rr, const VD &rhat, int Nij, int ps)
{
  PODHIP::PairGeom<KK_FLOAT> g;
  g.rr = rr.data();
  g.rhat = rhat.data();
  g.N = Nij;
  g.ps = ps;
  return g;
}

// radial functions of the native many-body kernels: the stored rbf/drbf, or with
// hermite == true the radial Hermite spline table, evaluated by the kernels themselves
template <class VD, class VI>
PODHIP::RadialSrc<KK_FLOAT> pod_radial_src(bool hermite, const VD &rbf, const VD &drbf,
                                           const VD &coef, const VD &r0, const VD &invdr,
                                           const VI &ti, const VI &tj, int nelements, int nbins,
                                           int nrbfmax, int Nij)
{
  PODHIP::RadialSrc<KK_FLOAT> s;
  s.rbf = rbf.data();
  s.drbf = drbf.data();
  s.coef = hermite ? coef.data() : nullptr;
  s.r0v = r0.data();
  s.invdrv = invdr.data();
  s.ti = ti.data();
  s.tj = tj.data();
  s.ne = nelements;
  s.nb = nbins;
  s.nr = nrbfmax;
  s.Nij = Nij;
  return s;
}

// B-spline radial basis of the 2-body terms; with otf == false the kernels read the
// stored rbf2/drbf2/bs_base instead
template <class VD, class VI>
PODHIP::BsplineArgs<KK_FLOAT> pod_bspline_args(bool otf, const VD &r0, const VD &invh,
                                               const VI &ti, const VI &tj, int nelements,
                                               int nbins, int nrbf2, int ileft)
{
  PODHIP::BsplineArgs<KK_FLOAT> g;
  g.r0v = otf ? r0.data() : nullptr;
  g.invhv = invh.data();
  g.ti = ti.data();
  g.tj = tj.data();
  g.ne = nelements;
  g.nbm1 = nbins - 1;    // largest admissible bin index
  g.gmax = nrbf2 - 4;    // largest admissible base index
  g.ish = ileft;         // 0 if left open, 3 if clamped
  return g;
}
}    // namespace
#endif

KOKKOS_INLINE_FUNCTION int npa(int d)   { return d*(d+1)*(d+2)/6; }  // # monomials of degree < d
KOKKOS_INLINE_FUNCTION int nmono(int d) { return (d+1)*(d+2)/2;   }  // # monomials of degree == d

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

  // native HIP kernels replace some Kokkos kernels on AMD GPUs.  setting the
  // environment variable LAMMPS_POD_NATIVE_HIP=0 selects the Kokkos versions.

  native_hip = 1;
  const char *env = getenv("LAMMPS_POD_NATIVE_HIP");
  if (env && (strcmp(env, "0") == 0)) native_hip = 0;
  hip_flags = HIP_RADIAL | HIP_ANGULAR | HIP_MFMA | HIP_MFMA_FORCE | HIP_DESC | HIP_TALLY | HIP_PORT |
      HIP_FUSE | HIP_FTALLY | HIP_FUSE_FC | HIP_FUSE_NEIGH | HIP_FIXED_PAIRS | HIP_PAIR_ALIGN;
  angular_otf = false;
  rbf_deferred = false;
  bs_deferred = false;
  forces_in_fij = false;
  pairs_fused = false;
  pair_stride = 0;
  block_gi1 = 0;
  fused_abort = false;
  fuse_neigh_off = false;
  mn_value = -1;
  mn_ncalls = -1;
  mn_list = nullptr;
  mn_inum = -1;
  env = getenv("LAMMPS_POD_HIP_FLAGS");
  if (env) hip_flags = atoi(env);
  hip_palign = 8;
  env = getenv("LAMMPS_POD_PAIR_ALIGN");
  if (env) hip_palign = atoi(env);
  if (hip_palign < 1) hip_palign = 1;
  hip_jtile = 16;
  env = getenv("LAMMPS_POD_HIP_JTILE");
  if (env) hip_jtile = atoi(env);
  abf_degree = -1;
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
  // record the neighbor list style for both backends.  unlike other KOKKOS pair
  // styles the host backend does not fall back to the non-accelerated compute()
  // but runs the same kernels on a neighbor list copied over from a non-KOKKOS
  // list, and compute() reads neighflag to decide how the virial is tallied.

  neighflag = lmp->kokkos->neighflag;

  if (host_flag) {
    if (lmp->kokkos->nthreads > 1)
      error->all(FLERR,"Pair style pod/kk can currently only run on a single "
                         "CPU thread");

    PairPOD::init_style();
    return;
  }

  if (atom->tag_enable == 0) error->all(FLERR, "Pair style POD requires atom IDs");
  if (force->newton_pair == 0) error->all(FLERR, "Pair style POD requires newton pair on");

  auto request = neighbor->add_request(this, NeighConst::REQ_FULL);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL)
    error->all(FLERR,"Must use half neighbor list style with pair pod/kk");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairPODKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairPOD::init_one(i,j);

  k_cutsq.view_host()(i,j) = k_cutsq.view_host()(j,i) = static_cast<KK_FLOAT>(cutone*cutone);
  k_cutsq.modify_host();

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
      h_rin(i, j) = static_cast<KK_FLOAT>(fastpodptr->rin[i][j]);
    }

  Kokkos::deep_copy(rin, h_rin);

  MemKK::realloc_kokkos(rcut, "pair_pod:rcut", nelements, nelements);
  auto h_rcut = Kokkos::create_mirror_view(rcut);
  for (int i = 0; i < nelements; i++)
    for (int j = 0; j < nelements; j++) {
      h_rcut(i, j) = static_cast<KK_FLOAT>(fastpodptr->rcut[i][j]);
    }

  Kokkos::deep_copy(rcut, h_rcut);

  MemKK::realloc_kokkos(invrdiff, "pair_pod:invrdiff", nelements, nelements);
  auto h_invrdiff = Kokkos::create_mirror_view(invrdiff);
  for (int i = 0; i < nelements; i++)
    for (int j = 0; j < nelements; j++) {
      h_invrdiff(i, j) = static_cast<KK_FLOAT>(fastpodptr->invrdiff[i][j]);
    }

  Kokkos::deep_copy(invrdiff, h_invrdiff);

  MemKK::realloc_kokkos(rcutsq, "pair_pod:rcutsq", nelements, nelements);
  auto h_rcutsq = Kokkos::create_mirror_view(rcutsq);
  for (int i = 0; i < nelements; i++)
    for (int j = 0; j < nelements; j++) {
      h_rcutsq(i, j) = static_cast<KK_FLOAT>(fastpodptr->rcutsq[i][j]);
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

namespace {
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
}    // namespace

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
    maxneighs = -1;
#ifdef KOKKOS_ENABLE_HIP
    if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
      if (native_hip && (hip_flags & HIP_PORT)) {
        // the neighbor counts change only when the list is rebuilt, so the maximum is reused
        // (saving a reduction and a wait for the GPU) until the next build
        if ((neighbor->ago != 0) && (neighbor->ncalls == mn_ncalls) && (list == mn_list) &&
            (inum == mn_inum) && (mn_value >= 0)) {
          maxneighs = mn_value;
        } else {
          if ((int) hip_iscratch.extent(0) < 1) MemKK::realloc_kokkos(hip_iscratch, "pair_pod:hip_iscratch", 1024);
          maxneighs = PODHIP::max_neighs(Kokkos::HIP().hip_stream(), hip_iscratch.data(),
                                         d_ilist.data(), d_numneigh.data(), inum);
          mn_value = maxneighs;
          mn_ncalls = neighbor->ncalls;
          mn_list = list;
          mn_inum = inum;
        }
      }
    }
#endif
    if (maxneighs < 0)
      Kokkos::parallel_reduce("PairPODKokkos::find_max_neighs",inum, FindMaxNumNeighs<DeviceType>(k_list), Kokkos::Max<int>(maxneighs));
    maxneigh = maxneighs;
  }

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
  // room for the fixed pair stride, rounded up to a multiple of hip_palign
  grow_pairs(nmax * (((maxneigh + hip_palign - 1) / hip_palign) * hip_palign));

  hip_acc_used = false;
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT)) {
      if ((int) hip_acc.extent(0) < 8) MemKK::realloc_kokkos(hip_acc, "pair_pod:hip_acc", 8);
      hip_acc_used = (hipMemsetAsync(hip_acc.data(), 0, 8 * sizeof(KK_ACC_FLOAT),
                                     Kokkos::HIP().hip_stream()) == hipSuccess);
    }
  }
#endif

  for (int block=0; block<nAtomBlocks; block++) {
    int gi1 = atomBlocks[block]-1;
    int gi2 = atomBlocks[block+1]-1;
    ni = gi2 - gi1; // total number of atoms in the current atom block
    block_gi1 = gi1;

    // the neighbors within rcut, in one pass if every stage reads them atom by atom
    // (nij is then an upper bound of the number of pairs), otherwise counted first
    nij = FusedNeighborList(gi1, ni);
    if (nij < 0) {
      nij = NeighborCount(numij, gi1, ni);
      NeighborList(rr, rhat, numij, typeai, idxi, ai, aj, ti, tj, gi1, ni, nij);
    }

    // compute atomic energy and force for the current atom block
    fused_abort = false;
    blockatom_energyforce(ei, fij, ni, nij);

    // a stage that reads the pairs as one array needs their exact number: compute the
    // block again, and the following ones, with the counted list
    if (fused_abort) {
      fused_abort = false;
      fuse_neigh_off = true;
      nij = exact_pairs(gi1);
      blockatom_energyforce(ei, fij, ni, nij);
    }

    // tally atomic energy to global energy
    tallyenergy(ei, gi1, ni);

    // tally the pair forces, if they were not tallied by allbody_forces()
    if (forces_in_fij) {
      if (pairs_fused) nij = exact_pairs(gi1);
      tallyforce(fij, ai, aj, nij);
      if (vflag) tallystress(fij, rr, rhat, ai, aj, nij);
    } else if (!POD_FUSE_FORCE_TALLY) {
      if (pairs_fused) nij = exact_pairs(gi1);
      tallyforce(fij, ai, aj, nij);    // benchmarking variant of allbody_forces()
    }
  }

#ifdef KOKKOS_ENABLE_HIP
  // global energy and virial accumulated on the device by the native tallies
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (hip_acc_used && (eflag_global || vflag_global)) {
      KK_ACC_FLOAT h_acc[8];
      if ((hipMemcpyAsync(h_acc, hip_acc.data(), 8 * sizeof(KK_ACC_FLOAT), hipMemcpyDeviceToHost,
                          Kokkos::HIP().hip_stream()) != hipSuccess) ||
          (hipStreamSynchronize(Kokkos::HIP().hip_stream()) != hipSuccess))
        error->one(FLERR, "pair pod/kk: reading the energy and virial from the GPU failed");
      if (eflag_global) eng_vdwl += static_cast<double>(h_acc[0]);
      if (vflag_global)
        for (int d = 0; d < 6; ++d) virial[d] -= static_cast<double>(h_acc[1 + d]);
    }
  }
#endif

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
  nActiveClusters = static_cast<KK_FLOAT>(podptr->nActiveClusters); // average number of active clusters
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

  // the size limits of the portable radial-angular and force kernels (POD_MAX_*) are
  // checked where those kernels are used, since the native HIP kernels have none

  MemKK::realloc_kokkos(coefficients, "pair_pod:coefficients", nCoeffPerElement * nelements);
  auto h_coefficients = Kokkos::create_mirror_view(coefficients);
  for (int i=0; i<nCoeffPerElement * nelements; i++) h_coefficients[i] = static_cast<KK_FLOAT>(podptr->coeff[i]);
  Kokkos::deep_copy(coefficients, h_coefficients);

  if (use_bspline) {
    const int npair = nelements * nelements;

    MemKK::realloc_kokkos(bs_r0, "pair_pod:bs_r0", npair);
    MemKK::realloc_kokkos(bs_invh, "pair_pod:bs_invh", npair);

    auto h_bsr0   = Kokkos::create_mirror_view(bs_r0);
    auto h_bsinvh = Kokkos::create_mirror_view(bs_invh);

    for (int i = 0; i < npair; ++i) {
      h_bsr0(i)   = static_cast<KK_FLOAT>(podptr->bs_r0[i]);
      h_bsinvh(i) = static_cast<KK_FLOAT>(podptr->bs_invh[i]);
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
      h_r0(i)    = static_cast<KK_FLOAT>(podptr->hermite_r0[i]);
      h_invdr(i) = static_cast<KK_FLOAT>(podptr->hermite_invdr[i]);
    }
    for (size_t i = 0; i < ncoef; ++i) {
      h_coef(i) = static_cast<KK_FLOAT>(podptr->rbf_hermite_coeffs[i]);
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
          h_neg_alpha[idx] = static_cast<KK_FLOAT>(bc[j].neg_alpha);
          h_pi_inv_t1[idx] = static_cast<KK_FLOAT>(bc[j].pi_inv_t1);
          h_dx_factor[idx] = static_cast<KK_FLOAT>(bc[j].dx_factor);
        }
      }
    }
    Kokkos::deep_copy(bessel_neg_alpha, h_neg_alpha);
    Kokkos::deep_copy(bessel_pi_inv_t1, h_pi_inv_t1);
    Kokkos::deep_copy(bessel_dx_factor, h_dx_factor);

    MemKK::realloc_kokkos(Phi, "pair_pod:Phi", ns*ns);
    auto h_Phi = Kokkos::create_mirror_view(Phi);
    for (int i=0; i<ns*ns; i++) h_Phi[i] = static_cast<KK_FLOAT>(podptr->Phi[i]);
    Kokkos::deep_copy(Phi, h_Phi);
  }

  if (eapod) {
    MemKK::realloc_kokkos(Proj, "pair_pod:Proj",  Mdesc * nComponents * nelements);
    auto h_Proj = Kokkos::create_mirror_view(Proj);
    for (int i=0; i<Mdesc * nComponents * nelements; i++) h_Proj[i] = static_cast<KK_FLOAT>(podptr->Proj[i]);
    Kokkos::deep_copy(Proj, h_Proj);

    int ncce = nClusters * nComponents * nelements;
    MemKK::realloc_kokkos(Centroids, "pair_pod:Centroids", ncce);
    auto h_Centroids = Kokkos::create_mirror_view(Centroids);
    for (int i=0; i<ncce; i++) h_Centroids[i] = static_cast<KK_FLOAT>(podptr->Centroids[i]);
    Kokkos::deep_copy(Centroids, h_Centroids);

    if (localeapod) {
      MemKK::realloc_kokkos(invLeftClusterRcut2, "pair_pod:invLeftClusterRcut2", ncce);
      MemKK::realloc_kokkos(invRightClusterRcut2, "pair_pod:invRightClusterRcut2", ncce);
      MemKK::realloc_kokkos(leftClusterEdges, "pair_pod:leftClusterEdges", ncce);
      MemKK::realloc_kokkos(rightClusterEdges, "pair_pod:rightClusterEdges", ncce);

      auto h_invLeftClusterRcut2 = Kokkos::create_mirror_view(invLeftClusterRcut2);
      for (int i=0; i<ncce; i++) h_invLeftClusterRcut2[i] = static_cast<KK_FLOAT>(podptr->invLeftClusterRcut2[i]);
      Kokkos::deep_copy(invLeftClusterRcut2, h_invLeftClusterRcut2);

      auto h_invRightClusterRcut2 = Kokkos::create_mirror_view(invRightClusterRcut2);
      for (int i=0; i<ncce; i++) h_invRightClusterRcut2[i] = static_cast<KK_FLOAT>(podptr->invRightClusterRcut2[i]);
      Kokkos::deep_copy(invRightClusterRcut2, h_invRightClusterRcut2);

      auto h_leftClusterEdges = Kokkos::create_mirror_view(leftClusterEdges);
      for (int i=0; i<ncce; i++) h_leftClusterEdges[i] = static_cast<KK_FLOAT>(podptr->leftClusterEdges[i]);
      Kokkos::deep_copy(leftClusterEdges, h_leftClusterEdges);

      auto h_rightClusterEdges = Kokkos::create_mirror_view(rightClusterEdges);
      for (int i=0; i<ncce; i++) h_rightClusterEdges[i] = static_cast<KK_FLOAT>(podptr->rightClusterEdges[i]);
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

  // check that the pq3 recursion generates each monomial u^a v^b w^c with
  // a+b+c <= L exactly once, as assumed by kernels that compute them directly

  abf_degree = 0;
  while ((abf_degree + 1) * (abf_degree + 2) * (abf_degree + 3) / 6 < K3) abf_degree++;
  if ((abf_degree + 1) * (abf_degree + 2) * (abf_degree + 3) / 6 != K3) abf_degree = -1;
  if (abf_degree >= 0) {
    const int L1 = abf_degree + 1;
    std::vector<int> ex(3 * K3, 0), seen(L1 * L1 * L1, 0);
    for (int k = 0; k < K3; k++) {
      if (k > 0) {
        const int m = h_pq_m[k], d = h_pq_d[k];
        if ((m < 0) || (m >= k) || (d < 1) || (d > 3)) { abf_degree = -1; break; }
        for (int c = 0; c < 3; c++) ex[3 * k + c] = ex[3 * m + c];
        ex[3 * k + d - 1]++;
      }
      const int a = ex[3 * k], b = ex[3 * k + 1], c = ex[3 * k + 2];
      if ((a + b + c > abf_degree) || seen[(a * L1 + b) * L1 + c]++) { abf_degree = -1; break; }
    }
  }

  MemKK::realloc_kokkos(p3_active, "p3_active", nabf3_active);
  auto h_p3_active = Kokkos::create_mirror_view(p3_active);
  for (int i=0; i<nabf3_active; i++) h_p3_active[i] = podptr->p3_active[i];
  Kokkos::deep_copy(p3_active, h_p3_active);

  // range of the 3-body angular terms of each active channel, for the native kernels
  MemKK::realloc_kokkos(p3_qrange, "pair_pod:p3_qrange", 2 * nabf3_active);
  auto h_p3_qrange = Kokkos::create_mirror_view(p3_qrange);
  for (int a = 0; a < nabf3_active; a++) {
    h_p3_qrange[2 * a] = podptr->pn3[podptr->p3_active[a]];
    h_p3_qrange[2 * a + 1] = podptr->pn3[podptr->p3_active[a] + 1];
  }
  Kokkos::deep_copy(p3_qrange, h_p3_qrange);
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
    std::vector<int> tmp(std::max(P3, 1));
    int n = 0;
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
    MemKK::realloc_kokkos(rr,   "pair_pod:rr",   nijmax);
    MemKK::realloc_kokkos(rhat, "pair_pod:rhat", 3 * nijmax);
    bool need_fij = !POD_FUSE_FORCE_TALLY;
#ifdef KOKKOS_ENABLE_HIP
    // the native force kernels accumulate the pair forces before tallying them
    if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>)
      need_fij = need_fij || native_hip;
#endif
    if (need_fij) MemKK::realloc_kokkos(fij, "pair_pod:f_ij", 3 * nijmax);

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
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT)) {
      const int naux = (Ni + PODHIP::SCAN_BLOCK - 1) / PODHIP::SCAN_BLOCK + 1;
      if ((int) hip_iscratch.extent(0) < naux) MemKK::realloc_kokkos(hip_iscratch, "pair_pod:hip_iscratch", naux + 1024);
      const int total = PODHIP::neighbor_count<KK_FLOAT>(Kokkos::HIP().hip_stream(), l_numij.data(),
          hip_iscratch.data(), x.data(), rcutsq.data(), pod_neigh_args(d_ilist, d_numneigh, d_neighbors, type, d_map, x, rcutsq), gi1, Ni);
      if (total >= 0) return total;
    }
  }
#endif

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

/* ----------------------------------------------------------------------
   pairs of the block from one native kernel (HIP_FUSE_NEIGH), without reading their number
   back.  only used when every stage of the block reads the pairs atom by atom through numij:
   the pair arrays then have the stride ni * maxneigh (returned), an upper bound of the
   number of pairs.  returns -1 if not used
------------------------------------------------------------------------- */

template<class DeviceType>
int PairPODKokkos<DeviceType>::FusedNeighborList(int gi1, int Ni)
{
  pairs_fused = false;
  pair_stride = 0;
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    // the radial bases must be deferred to their consumers or computed atom by atom
    const bool radial = use_hermite &&
        (use_bspline ? ((hip_flags & HIP_FUSE) && (hip_flags & HIP_ANGULAR) && (abf_degree >= 0))
                     : (nelements * nrbf2 <= 256));
    if (!native_hip || fuse_neigh_off || !(hip_flags & HIP_FUSE_NEIGH) ||
        !(hip_flags & HIP_PORT) || !(hip_flags & HIP_FTALLY) || !hip_acc_used || !radial ||
        (nl3 <= 0) || (Ni <= 0))
      return -1;
    const size_t bytes = PODHIP::neighbor_fused_scratch(Ni);
    const int nint = (int) ((bytes + sizeof(int) - 1) / sizeof(int));
    if ((int) hip_lookback.extent(0) < nint)
      MemKK::realloc_kokkos(hip_lookback, "pair_pod:hip_lookback", nint);
    const int N = Ni * maxneigh;
    // fixed stride: the pairs of atom i at i*ps (maxneigh, rounded up to a multiple of
    // hip_palign with HIP_PAIR_ALIGN), written without a scan
    if ((hip_flags & HIP_FIXED_PAIRS) && (maxneigh > 0)) {
      const int ps = (hip_flags & HIP_PAIR_ALIGN)
          ? ((maxneigh + hip_palign - 1) / hip_palign) * hip_palign : maxneigh;
      const int Nf = Ni * ps;
      if (PODHIP::neighbor_fixed<KK_FLOAT>(Kokkos::HIP().hip_stream(), rr.data(), rhat.data(),
              typeai.data(), aj.data(), ti.data(), tj.data(), numij.data(), x.data(),
              rcutsq.data(), pod_neigh_args(d_ilist, d_numneigh, d_neighbors, type, d_map, x,
              rcutsq), gi1, Ni, Nf, ps, nelements)) {
        pairs_fused = true;
        pair_stride = ps;
        return Nf;
      }
    }
    if (PODHIP::neighbor_fused<KK_FLOAT>(Kokkos::HIP().hip_stream(), rr.data(), rhat.data(),
            typeai.data(), idxi.data(), ai.data(), aj.data(), ti.data(), tj.data(), numij.data(),
            hip_lookback.data(), x.data(), rcutsq.data(),
            pod_neigh_args(d_ilist, d_numneigh, d_neighbors, type, d_map, x, rcutsq), gi1, Ni, N)) {
      pairs_fused = true;
      return N;
    }
  }
#endif
  return -1;
}

// the pairs of the block from NeighborCount() and NeighborList(), in the same order as those
// of FusedNeighborList() but with the exact number of pairs as stride; returns that number
template<class DeviceType>
int PairPODKokkos<DeviceType>::exact_pairs(int gi1)
{
  pairs_fused = false;
  pair_stride = 0;
  const int n = NeighborCount(numij, gi1, ni);
  NeighborList(rr, rhat, numij, typeai, idxi, ai, aj, ti, tj, gi1, ni, n);
  return n;
}

// called before a stage that reads the pairs as one array of nij entries: with the pairs of
// FusedNeighborList() the block has to be computed again (fused_abort), returns true
template<class DeviceType>
bool PairPODKokkos<DeviceType>::fused_abort_needed()
{
  if (!pairs_fused) return false;
  fused_abort = true;
  return true;
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::NeighborList(t_pod_1d l_rr, t_pod_1d l_rhat, t_pod_1i l_numij,
  t_pod_1i l_typeai, t_pod_1i l_idxi, t_pod_1i l_ai, t_pod_1i l_aj, t_pod_1i l_ti, t_pod_1i l_tj,
  int gi1, int Ni, int Nij)
{
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT) &&
        PODHIP::neighbor_list<KK_FLOAT>(Kokkos::HIP().hip_stream(), l_rr.data(), l_rhat.data(),
            l_typeai.data(), l_idxi.data(), l_ai.data(), l_aj.data(), l_ti.data(), l_tj.data(),
            l_numij.data(), x.data(), rcutsq.data(),
            pod_neigh_args(d_ilist, d_numneigh, d_neighbors, type, d_map, x, rcutsq), gi1, Ni, Nij))
      return;
  }
#endif

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
  const KK_FLOAT y3 = y * y2 - static_cast<KK_FLOAT>(1.0);
  const KK_FLOAT y4 = y3 * y3 + static_cast<KK_FLOAT>(1.0e-6);
  const KK_FLOAT y5 = Kokkos::sqrt(y4);

  fcut  = e_v * Kokkos::exp(static_cast<KK_FLOAT>(-1.0) / y5);  // argument is always negative
  dfcut = fcut * (static_cast<KK_FLOAT>(3.0) * invrmax * y2 * y3) / (y4 * y5);
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
    const KK_FLOAT invr    = static_cast<KK_FLOAT>(1.0) / r;
    const KK_FLOAT y       = r * invrmax;

    // cutoff function for radial basis
    KK_FLOAT fcut, dfcut;
    cutoff_exp(y, invrmax, E_V, fcut, dfcut);

    // Shared Bessel constants
    const KK_FLOAT f1   = fcut * invr;
    const KK_FLOAT g1   = (dfcut - f1) * invr;
    const KK_FLOAT bfac = Kokkos::sqrt(static_cast<KK_FLOAT>(2.0) * invrmax);
    const KK_FLOAT bf1  = bfac * f1;
    const KK_FLOAT bg1  = bfac * g1;

    int nij = n;

    // Bessel radial basis
    for (int j = 0; j < l_nbesselpars; ++j) {
      const int bcj = bc_base + j;

      const KK_FLOAT mt2   = Kokkos::expm1(l_neg_alpha(bcj) * y);
      const KK_FLOAT xpi   = mt2 * l_pi_inv_t1(bcj);
      const KK_FLOAT pi_dx = l_dx_factor(bcj) * (static_cast<KK_FLOAT>(1.0) + mt2);
      const KK_FLOAT Kf1dx = bf1 * pi_dx;

      KK_ACC_FLOAT ix = static_cast<KK_ACC_FLOAT>(xpi);
      for (int i = 1; i <= l_besseldegree; ++i) {
        const KK_FLOAT cosax  = static_cast<KK_FLOAT>(Kokkos::cos(ix));
        const KK_FLOAT inv_i  = static_cast<KK_FLOAT>(1.0) / static_cast<KK_FLOAT>(i);
        const KK_FLOAT isinax = inv_i * static_cast<KK_FLOAT>(Kokkos::sin(ix));

        const KK_FLOAT rbfv    = bf1 * isinax;
        const KK_FLOAT drbftdr = bg1 * isinax + Kf1dx * cosax;

        rbft (nij) = rbfv;
        drbft(nij) = drbftdr * invdij;  // R'/r

        ix  += static_cast<KK_ACC_FLOAT>(xpi);
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
      drbft(nij) = dterm * inva * invdij;  // R'/r
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
            sum1 += static_cast<KK_ACC_FLOAT>(rbft(i + Nij * k) * phi_kj);
            sum2 += static_cast<KK_ACC_FLOAT>(drbft(i + Nij * k) * phi_kj);
        }

        rbf(i + Nij * j) = static_cast<KK_FLOAT>(sum1);
        drbf(i + Nij * j) = static_cast<KK_FLOAT>(sum2);
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::radialbasis_hermite(
    t_pod_1d l_rbf, t_pod_1d l_drbf, t_pod_1d l_rr, t_pod_1i ti, t_pod_1i tj, int N)
{
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT)) {
      // with use_bspline == 0 the 2-body descriptors are computed in the same pass
      const bool fuse = !use_bspline && (nelements * nrbf2 <= 256);
      if (fuse) {
        d2_fused = PODHIP::launch(PODHIP::radialbasis_hermite_kernel<KK_FLOAT, KK_ACC_FLOAT, true>,
            ni, 64, 0, Kokkos::HIP().hip_stream(), l_rbf.data(), l_drbf.data(), pod_geom(l_rr, rhat, N, pair_stride),
            ti.data(), tj.data(), hermite_r0.data(), hermite_invdr.data(),
            rbf_hermite_coeffs.data(), nelements, nhermite_bins, nrbfmax, N, numij.data(),
            bd.data(), nrbf2, ni);
        if (d2_fused) return;
      }
      if (fused_abort_needed()) return;
      if (PODHIP::launch(PODHIP::radialbasis_hermite_kernel<KK_FLOAT, KK_ACC_FLOAT, false>,
            PODHIP::nblocks_for(N, 256), 256, 0, Kokkos::HIP().hip_stream(), l_rbf.data(),
            l_drbf.data(), pod_geom(l_rr, rhat, N, pair_stride), ti.data(), tj.data(), hermite_r0.data(),
            hermite_invdr.data(), rbf_hermite_coeffs.data(), nelements, nhermite_bins, nrbfmax, N,
            (const int *) nullptr, (KK_FLOAT *) nullptr, 0, 0))
        return;
    }
  }
#endif

  const int ne = nelements;
  const int nb = nhermite_bins;
  const int nr = nrbfmax;

  auto s_r0    = hermite_r0;
  auto s_invdr = hermite_invdr;
  auto s_coef  = rbf_hermite_coeffs;

  Kokkos::parallel_for("radialbasis_hermite",
    Kokkos::RangePolicy<DeviceType>(0, N),
    KOKKOS_LAMBDA(const int n) {

      const int itype = ti(n);
      const int jtype = tj(n);
      const int pair  = itype * ne + jtype;

      const KK_FLOAT r = l_rr(n);

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
        const KK_FLOAT dfdt = c1 + t * (static_cast<KK_FLOAT>(2.0)*c2 + static_cast<KK_FLOAT>(3.0)*c3*t);

        const int idx = n + N*k;
        l_rbf (idx) = f;
        l_drbf(idx) = dfdt * invdrr;   // f'(r)/r
      }
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::ensure_rbf(int N)
{
  // computes the Hermite radial basis that was deferred in blockatom_base_descriptors()
  if (!rbf_deferred) return;
  rbf_deferred = false;
  radialbasis_hermite(rbf, drbf, rr, ti, tj, N);
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::ensure_bspline(int N)
{
  // computes the B-spline radial basis that was deferred in blockatom_base_descriptors()
  if (!bs_deferred) return;
  bs_deferred = false;
  radialbasis_bspline(rbf2, drbf2, rr, bs_base, ti, tj, N);
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
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT) &&
        PODHIP::launch(PODHIP::radialbasis_bspline_kernel<KK_FLOAT>, PODHIP::nblocks_for(N, 256), 256,
            0, Kokkos::HIP().hip_stream(), l_rbf2.data(), l_drbf2.data(), l_base.data(),
            pod_geom(l_rr, rhat, N, pair_stride), pod_bspline_args(true, bs_r0, bs_invh, ti, tj, nelements, bs_nb, nrbf2,
                                           bs_ileft), N))
      return;
  }
#endif

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
      const KK_FLOAT dscale = inside ? invh / r : (KK_FLOAT)0.0;
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

      l_drbf2(n      ) = sel4(sh    , dB0, dB1, dB2, dB3);   // R'(r)/r
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
  if ((l_nelements > POD_MAX_NELEM) || (nrbf3 > POD_MAX_NRBF) || (P3 > POD_MAX_DEG))
    error->one(FLERR, "Pair style pod/kk supports at most {} elements, {} three-body radial "
               "basis functions and a three-body angular degree of {} without the native AMD "
               "GPU kernels", POD_MAX_NELEM, POD_MAX_NRBF, POD_MAX_DEG);
  pod_dispatch_int<1, POD_MAX_NELEM>(l_nelements, [&](auto NE) {
    pod_dispatch_int<1, POD_MAX_NRBF>(nrbf3, [&](auto NR) {
      pod_dispatch_int<1, POD_MAX_DEG>(P3, [&](auto MD) {
        this->template radialangularsum_impl<decltype(NE)::value,
                                             decltype(NR)::value,
                                             decltype(MD)::value>(
            l_sumU, l_rbf, l_rhat, l_tj, l_numij, K3, Ni, Nij);
      });
    });
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobodydesc(t_pod_1d d2, t_pod_1d l_rbf2, t_pod_1i l_idxi, t_pod_1i l_tj,
        int l_nrbf2, const int Ni, const int Nij)
{
#ifdef KOKKOS_ENABLE_HIP
  // d2 is written, not accumulated, by the native kernel; it was zeroed before
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_TALLY) &&
        PODHIP::twobodydesc<KK_FLOAT, KK_ACC_FLOAT>(Kokkos::HIP().hip_stream(), d2.data(),
                                                     l_rbf2.data(), l_tj.data(), numij.data(),
                                                     nelements, l_nrbf2, Ni, Nij, pair_stride))
      return;
  }
#endif

  if (fused_abort_needed()) return;
  int totalIterations = l_nrbf2 * Nij;
  Kokkos::parallel_for("twobodydesc", Kokkos::RangePolicy<DeviceType>(0,totalIterations), KOKKOS_LAMBDA(int idx) {
    int n = idx / l_nrbf2;
    int m = idx % l_nrbf2;
    Kokkos::atomic_add(&d2(l_idxi(n) + Ni * (m + l_nrbf2 * l_tj(n))), l_rbf2(n + Nij * m));
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobody_forces(
    t_pod_1d fij, t_pod_1d cb2, t_pod_1d l_drbf2, t_pod_1d l_rhat, t_pod_1d l_rr,
    t_pod_1i l_idxi, t_pod_1i l_tj, int l_nrbf2, const int Ni, const int Nij)
{
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT) &&
        PODHIP::launch(PODHIP::twobody_forces_kernel<KK_FLOAT, KK_ACC_FLOAT>,
            PODHIP::nblocks_for(Nij, 256), 256, 0, Kokkos::HIP().hip_stream(), fij.data(),
            cb2.data(), l_drbf2.data(), pod_geom(l_rr, l_rhat, Nij, pair_stride), (const int *) nullptr, l_idxi.data(),
            l_tj.data(), l_nrbf2, l_nrbf2, Ni, Nij, PODHIP::BsplineArgs<KK_FLOAT>{}))
      return;
  }
#endif

  Kokkos::parallel_for("twobody_forces",
    Kokkos::RangePolicy<DeviceType>(0, Nij),
    KOKKOS_LAMBDA(const int n) {
      const int ii = l_idxi(n) + Ni * l_nrbf2 * l_tj(n);

      KK_ACC_FLOAT cR = 0.0;
      for (int m = 0; m < l_nrbf2; ++m) {
        cR += static_cast<KK_ACC_FLOAT>(cb2(ii + Ni * m) * l_drbf2(n + Nij * m));
      }
      cR *= static_cast<KK_ACC_FLOAT>(l_rr(n));

      const int i1 = 3 * n;
      fij(i1 + 0) += static_cast<KK_FLOAT>(cR * static_cast<KK_ACC_FLOAT>(l_rhat(n)));
      fij(i1 + 1) += static_cast<KK_FLOAT>(cR * static_cast<KK_ACC_FLOAT>(l_rhat(n + Nij)));
      fij(i1 + 2) += static_cast<KK_FLOAT>(cR * static_cast<KK_ACC_FLOAT>(l_rhat(n + 2*Nij)));
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobodydesc_bspline(t_pod_1d d2, t_pod_1d l_rbf2, t_pod_1i l_base,
        t_pod_1i l_idxi, t_pod_1i l_tj, int l_nrbf2, const int Ni, const int Nij)
{
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    // d2 is written, not accumulated, by the native kernel; it was zeroed before
    if (native_hip && (hip_flags & HIP_PORT) &&
        PODHIP::launch(PODHIP::twobodydesc_bspline_kernel<KK_FLOAT, KK_ACC_FLOAT>, Ni, 64,
            (size_t) nelements * l_nrbf2 * sizeof(KK_ACC_FLOAT), Kokkos::HIP().hip_stream(),
            d2.data(), l_rbf2.data(), l_base.data(), l_tj.data(), numij.data(), nelements,
            l_nrbf2, Ni, Nij, pod_bspline_args(bs_deferred, bs_r0, bs_invh, ti, l_tj, nelements,
                                               bs_nb, l_nrbf2, bs_ileft), pod_geom(rr, rhat, Nij, pair_stride)))
      return;
  }
#endif

  if (fused_abort_needed()) return;
  ensure_bspline(Nij);

  Kokkos::parallel_for("twobodydesc_bspline",
    Kokkos::RangePolicy<DeviceType>(0, Nij), KOKKOS_LAMBDA(const int n) {
      const int i0 = l_idxi(n) + Ni * (l_base(n) + l_nrbf2 * l_tj(n));
      for (int s = 0; s < 4; ++s)
        Kokkos::atomic_add(&d2(i0 + Ni * s), l_rbf2(n + Nij * s));
    });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::twobody_forces_bspline(
    t_pod_1d fij, t_pod_1d cb2, t_pod_1d l_drbf2, t_pod_1d l_rhat, t_pod_1d l_rr,
    t_pod_1i l_base, t_pod_1i l_idxi, t_pod_1i l_tj, int l_nrbf2, const int Ni, const int Nij)
{
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT) &&
        PODHIP::launch(PODHIP::twobody_forces_kernel<KK_FLOAT, KK_ACC_FLOAT>,
            PODHIP::nblocks_for(Nij, 256), 256, 0, Kokkos::HIP().hip_stream(), fij.data(),
            cb2.data(), l_drbf2.data(), pod_geom(l_rr, l_rhat, Nij, pair_stride), l_base.data(), l_idxi.data(),
            l_tj.data(), l_nrbf2, 4, Ni, Nij,
            pod_bspline_args(bs_deferred, bs_r0, bs_invh, ti, l_tj, nelements, bs_nb, l_nrbf2,
                             bs_ileft)))
      return;
  }
#endif

  ensure_bspline(Nij);

  Kokkos::parallel_for("twobody_forces_bspline",
    Kokkos::RangePolicy<DeviceType>(0, Nij), KOKKOS_LAMBDA(const int n) {
      const int ii = l_idxi(n) + Ni * (l_base(n) + l_nrbf2 * l_tj(n));

      KK_ACC_FLOAT cR = 0.0;
      for (int s = 0; s < 4; ++s)
        cR += cb2(ii + Ni * s) * l_drbf2(n + Nij * s);
      cR *= static_cast<KK_ACC_FLOAT>(l_rr(n));

      const int i1 = 3 * n;
      fij(i1 + 0) += static_cast<KK_FLOAT>(cR * static_cast<KK_ACC_FLOAT>(l_rhat(n)));
      fij(i1 + 1) += static_cast<KK_FLOAT>(cR * static_cast<KK_ACC_FLOAT>(l_rhat(n + Nij)));
      fij(i1 + 2) += static_cast<KK_FLOAT>(cR * static_cast<KK_ACC_FLOAT>(l_rhat(n + 2*Nij)));
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
        tmp += static_cast<KK_ACC_FLOAT>(static_cast<KK_FLOAT>(l_pc3(q)) * su * su);
      }
      d3(i + Ni * (a + l_nabf3_active * m)) = static_cast<KK_FLOAT>(tmp);
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
            tmp += static_cast<KK_ACC_FLOAT>(static_cast<KK_FLOAT>(l_pc3(q)) * l_sumU(i1 + l_nelements * q + nmi) * l_sumU(i2 + l_nelements * q + nmi));
          }
          d3(k) = static_cast<KK_FLOAT>(tmp);
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
        fb3(q + idxU) += c3 * static_cast<KK_FLOAT>(l_pc3(q)) * l_sumU(q + idxU);
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
            const KK_FLOAT w = static_cast<KK_FLOAT>(pk) * cb3[em];
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
              tmp += static_cast<KK_ACC_FLOAT>(static_cast<KK_FLOAT>(c) * l_sumU(idxU + i1 + l_nelements * j1) * l_sumU(idxU + i2 + l_nelements * j2) * l_sumU(idxU + i3 + l_nelements * j3));
            }
            d4(k) = static_cast<KK_FLOAT>(tmp);
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
          KK_FLOAT c = c4 * static_cast<KK_FLOAT>(l_pc4(q));
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
                KK_FLOAT c4 = static_cast<KK_FLOAT>(c) * cb4[k];
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

template<class DeviceType, bool BSPLINE, bool FUSEF, bool VGLOBAL>
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

    KK_FLOAT rb[POD_MAX_NRBF], drb[POD_MAX_NRBF];
    #pragma unroll
    for (int m = 0; m < POD_MAX_NRBF; ++m) {
      const bool on = (m < nr3);
      rb[m]  = on ? rbf (j + Nij*m) : (KK_FLOAT) 0.0;
      drb[m] = on ? drbf(j + Nij*m) : (KK_FLOAT) 0.0;
    }

    // two-body (drbf = R'/r): A00 = 1 -> accumulate to fcAdR
    KK_ACC_FLOAT fcAdR = 0.0;
    if constexpr (BSPLINE) {
      const int i2 = ia + Ni*(base(j) + nr2*te);
      #pragma unroll
      for (int s = 0; s < 4; ++s)
        fcAdR += cb2(i2 + Ni*s) * drbf2(j + Nij*s);
    } else {
      const int i2 = ia + Ni*nr2*te;
      #pragma unroll
      for (int m = 0; m < POD_MAX_NRBF; ++m)
        if (m < nr3) fcAdR += cb2(i2 + Ni*m) * drb[m];
      for (int m = nr3; m < nr2; ++m)
        fcAdR += cb2(i2 + Ni*m) * drbf2(j + Nij*m);
    }

    // many-body: fcRdA + fcAdR
    const int NeK3 = ne * K3;
    const int ii   = te + NeK3 * nr3 * ia;

    KK_ACC_FLOAT Du = 0.0, Dv = 0.0, Dw = 0.0;
    KK_FLOAT wr = 1.0, wrm1 = 0.0;
    for (int r = 0; r <= P3; ++r) {
      KK_FLOAT vq = 1.0, vqm1 = 0.0;
      for (int q = 0; q + r <= P3; ++q) {
        const KK_FLOAT vw  = vq * wr;
        const KK_FLOAT vwv = (KK_FLOAT) q * vqm1 * wr;
        const KK_FLOAT vww = (KK_FLOAT) r * vq * wrm1;

        int d  = q + r;
        int k  = npa(d) + (d+1)*r - (r*(r-1))/2 + q;
        int nm = nmono(d);
        KK_FLOAT up = 1.0, upm1 = 0.0;

        for (int p = 0; p + q + r <= P3; ++p) {
          const int b = ii + ne * k;
          KK_ACC_FLOAT fcR = 0.0, fcdR = 0.0;
          #pragma unroll
          for (int m = 0; m < POD_MAX_NRBF; ++m) {
            if (m < nr3) {
              const KK_FLOAT c = fc(b + NeK3*m);
              fcR  += c * rb[m];
              fcdR += c * drb[m];
            }
          }
          fcAdR += fcdR * (up * vw);
          Du += fcR * ((KK_FLOAT) p * upm1 * vw);
          Dv += fcR * (up * vwv);
          Dw += fcR * (up * vww);

          upm1 = up;  up *= u;
          k += nm + r;  nm += d + 2;  ++d;
        }
        vqm1 = vq;  vq *= v;
      }
      wrm1 = wr;  wr *= w;
    }

    // Jacobian: J = d(u,v,w) / d(x,y,z) = (I - rhat rhat^T) / r
    // f_ij = fc * d(R * A) = J * Dr_hat + fc * A * r_hat
    //      = D/r + (fcAdR r - (D*u)/r) u
    const KK_ACC_FLOAT g  = Du*u + Dv*v + Dw*w;
    const KK_ACC_FLOAT sr = fcAdR*dij - g*invr;
    const KK_ACC_FLOAT fx = Du*invr + sr*u;
    const KK_ACC_FLOAT fy = Dv*invr + sr*v;
    const KK_ACC_FLOAT fz = Dw*invr + sr*w;

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
  if (nrbf3 > POD_MAX_NRBF)
    error->one(FLERR, "Pair style pod/kk supports at most {} three-body radial basis functions "
               "without the native AMD GPU kernels", POD_MAX_NRBF);
  const bool vg = (vflag_global != 0);
  if (use_bspline) {
    if (vg) allbody_forces_impl<true,  true >(l_fij, l_cb2, Ni, Nij);
    else    allbody_forces_impl<true,  false>(l_fij, l_cb2, Ni, Nij);
  } else {
    if (vg) allbody_forces_impl<false, true >(l_fij, l_cb2, Ni, Nij);
    else    allbody_forces_impl<false, false>(l_fij, l_cb2, Ni, Nij);
  }
}

template<class DeviceType>
template<bool BSPLINE, bool VGLOBAL>
void PairPODKokkos<DeviceType>::allbody_forces_impl(t_pod_1d l_fij, t_pod_1d l_cb2,
                                                    const int Ni, const int Nij)
{
  PODAllBodyForceFunctor<DeviceType, BSPLINE, POD_FUSE_FORCE_TALLY, VGLOBAL> fn;
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
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT) &&
        PODHIP::launch(PODHIP::crossdesc_kernel<KK_FLOAT>, PODHIP::nblocks_for((long) n12 * Ni, 256),
            256, 0, Kokkos::HIP().hip_stream(), d12.data(), d1.data(), d2.data(), ind1.data(),
            ind2.data(), n12, Ni))
      return;
  }
#endif

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
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT) &&
        PODHIP::launch(PODHIP::crossdesc_reduction_kernel<KK_FLOAT>,
            PODHIP::nblocks_for((long) n12 * Ni, 256), 256, 0, Kokkos::HIP().hip_stream(),
            cb1.data(), cb2.data(), c12.data(), d1.data(), d2.data(), ind1.data(), ind2.data(),
            n12, Ni))
      return;
  }
#endif

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
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    constexpr int V = 16 / sizeof(KK_FLOAT);
    if (native_hip && (hip_flags & HIP_PORT) &&
        PODHIP::launch(PODHIP::zero_kernel<KK_FLOAT>, PODHIP::nblocks_for(N / V + 1, 256), 256, 0,
                       Kokkos::HIP().hip_stream(), a.data(), (long) N))
      return;
  }
#endif

  Kokkos::parallel_for("initialize_array", Kokkos::RangePolicy<DeviceType>(0,N), KOKKOS_LAMBDA(int i) {
    a(i) = 0.0;
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::blockatom_base_descriptors(t_pod_1d bd, int Ni, int Nij)
{

  auto d2 = Kokkos::subview(bd, std::make_pair(0, Ni * nl2));
  auto d3 = Kokkos::subview(bd, std::make_pair(Ni * nl2, Ni * (nl2 + nl3)));
  auto d4 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3), Ni * (nl2 + nl3 + nl4)));
  auto d33 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3 + nl4), Ni * (nl2 + nl3 + nl4 + nl33)));
  auto d34 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3 + nl4 + nl33), Ni * (nl2 + nl3 + nl4 + nl33 + nl34)));
  auto d44 = Kokkos::subview(bd, std::make_pair(Ni * (nl2 + nl3 + nl4 + nl33 + nl34), Ni * (nl2 + nl3 + nl4 + nl33 + nl34 + nl44)));

  d2_fused = false;

  rbf_deferred = false;
  if (use_hermite) {
#ifdef KOKKOS_ENABLE_HIP
    // the native many-body kernels evaluate the Hermite spline of a pair when they need it.
    // with B-splines for the 2-body terms nothing else reads rbf/drbf, so they are only
    // computed (by ensure_rbf()) if one of those kernels falls back to a stored basis.
    if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>)
      rbf_deferred = native_hip && (hip_flags & HIP_FUSE) && (hip_flags & HIP_ANGULAR) &&
          (hip_flags & HIP_PORT) && use_bspline && (abf_degree >= 0);
#endif
    // Directly fills final basis rbf/drbf (Phi-orthogonalized)
    if (!rbf_deferred) radialbasis_hermite(rbf, drbf, rr, ti, tj, Nij);
  } else {
    // Original analytic path
    bool done = false;
#ifdef KOKKOS_ENABLE_HIP
    if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
      if (native_hip && (hip_flags & HIP_RADIAL))
        done = PODHIP::radialbasis_phi<KK_FLOAT, KK_ACC_FLOAT>(
            Kokkos::HIP().hip_stream(), rbf.data(), drbf.data(), pod_geom(rr, rhat, Nij, pair_stride), rin.data(),
            invrdiff.data(), rin.stride(0), rin.stride(1), invrdiff.stride(0), invrdiff.stride(1),
            bessel_neg_alpha.data(), bessel_pi_inv_t1.data(), bessel_dx_factor.data(), Phi.data(),
            ti.data(), tj.data(), nelements, besseldegree, inversedegree, nbesselpars, ns,
            nrbfmax, Nij);
    }
#endif
    if (!done) {
      radialbasis(rbft, drbft, rr, rin, invrdiff,
                  bessel_neg_alpha, bessel_pi_inv_t1, bessel_dx_factor, ti, tj,
                  nelements, besseldegree, inversedegree, nbesselpars, Nij);

      radialPhi(rbft, drbft, Phi, rbf, drbf, Nij, ns, nrbfmax);
    }
  }

  bs_deferred = false;
  if (use_bspline) {
#ifdef KOKKOS_ENABLE_HIP
    // the native 2-body kernels evaluate the B-splines of a pair themselves; the basis is
    // only stored (by ensure_bspline()) if one of them falls back to the portable version
    if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>)
      bs_deferred = native_hip && (hip_flags & HIP_FUSE) && (hip_flags & HIP_PORT);
#endif
    if (!bs_deferred) radialbasis_bspline(rbf2, drbf2, rr, bs_base, ti, tj, Nij);
  }

  if (!d2_fused) {
    set_array_to_zero(d2, Ni*nl2);
    if (use_bspline) twobodydesc_bspline(d2, rbf2, bs_base, idxi, tj, nrbf2, Ni, Nij);
    else             twobodydesc(d2, rbf2, idxi, tj, nrbf2, Ni, Nij);
  }

  if ((nl3 > 0) && (Nij>1)) {
    // with the native HIP kernels the angular basis is computed on the fly by its two
    // users, the radial-angular sums and the forces
    angular_otf = false;
#ifdef KOKKOS_ENABLE_HIP
    if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
      if (native_hip && (hip_flags & HIP_ANGULAR) && (abf_degree >= 0)) {
        const auto rsrc = pod_radial_src(rbf_deferred, rbf, drbf, rbf_hermite_coeffs, hermite_r0,
                                         hermite_invdr, ti, tj, nelements, nhermite_bins, nrbfmax, Nij);
        angular_otf = PODHIP::radialangularsum_otf<KK_FLOAT, KK_ACC_FLOAT>(
            Kokkos::HIP().hip_stream(), sumU.data(), rsrc, pod_geom(rr, rhat, Nij, pair_stride), tj.data(),
            numij.data(), pq_m.data(), pq_d.data(), nelements, nrbf3, K3, Ni, Nij,
            (hip_flags & HIP_MFMA) != 0, hip_jtile);
      }
    }
#endif
    if (!angular_otf) {
      if (fused_abort_needed()) return;
      ensure_rbf(Nij);
      radialangularsum(sumU, rbf, rhat, tj, numij, nelements, nrbf3, K3, P3, Ni, Nij);
    }
  }

  bool desc_done = false;
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_DESC) && (nl3 > 0) && (Nij > 1))
      desc_done = PODHIP::desc34<KK_FLOAT, KK_ACC_FLOAT>(
          Kokkos::HIP().hip_stream(), pod_desc_args(d3, d4, pc3, pn3, p3_active, p3_qrange, pa4, pb4, pc4, nelements, nrbf3, nabf3_active,
                          nrbf4, nabf4, K3, Q4, Ni, (nl4 > 0) && (Nij > 2)),
          sumU.data());
  }
#endif

  if (!desc_done) {
    if ((nl3 > 0) && (Nij>1))
      threebodydesc(d3, sumU, pc3, pn3, p3_active, nelements, nrbf3, nabf3_active, K3, Ni);

    if ((nl4 > 0) && (Nij>2))
      fourbodydesc(d4, sumU, pa4, pb4, pc4, nelements, nrbf3, nrbf4, nabf4, K3, Q4, Ni);
  }

  if ((nl33>0) && (Nij>3)) {
    crossdesc(d33, d3, d3, ind33l, ind33r, nl33, Ni);
  }

  if ((nl34>0) && (Nij>4)) {
    crossdesc(d34, d3, d4, ind34l, ind34r, nl34, Ni);
  }

  if ((nl44>0) && (Nij>5)) {
    crossdesc(d44, d4, d4, ind44l, ind44r, nl44, Ni);
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::blockatom_base_coefficients(t_pod_1d ei, t_pod_1d cb, t_pod_1d B, int Ni)
{
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT) &&
        PODHIP::launch(PODHIP::base_coefficients_kernel<KK_FLOAT>, PODHIP::nblocks_for(Ni, 256), 256,
            0, Kokkos::HIP().hip_stream(), ei.data(), cb.data(), B.data(), coefficients.data(),
            typeai.data(), Mdesc, nCoeffPerElement, Ni))
      return;
  }
#endif

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
  const KK_FLOAT omu = static_cast<KK_FLOAT>(1.0) - u;

  fcut  = static_cast<KK_FLOAT>(1.0) - u2 * u * (static_cast<KK_FLOAT>(10.0) - u * (static_cast<KK_FLOAT>(15.0) - static_cast<KK_FLOAT>(6.0) * u));

  // dfcut := (1/(2pc)) * d(fcut)/dpc
  dfcut = static_cast<KK_FLOAT>(-30.0) * u2 * omu * omu * inv_rcut2;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
void PairPODKokkos<DeviceType>::cluster_cutoff_hat(
    const KK_FLOAT pc, const KK_FLOAT inv_rcut2, KK_FLOAT &fcut, KK_FLOAT &dfcut)
{
  const KK_FLOAT y2   = (pc * pc) * inv_rcut2; // (pc/r)^2
  const KK_FLOAT omy  = static_cast<KK_FLOAT>(1.0) - y2 * y2;         // 1-(pc/r)^4
  const KK_FLOAT omy2 = omy * omy;

  fcut = omy2 * omy2; // (1-y^4)^4

  // dfcut := (1/(2pc)) * d(fcut)/dpc
  dfcut = static_cast<KK_FLOAT>(-8.0) * inv_rcut2 * y2 * omy2 * omy;
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

#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT)) {
      const hipStream_t stream = Kokkos::HIP().hip_stream();
      const PODHIP::LocalEnvArgs la{nCls, nDes, nCoeff, nActMax};
      const bool split = (nActMax <= PODHIP::LP_KMAX);
      const bool ok = split
          ? (PODHIP::launch(PODHIP::localenv_pca_split_kernel<KK_FLOAT, KK_ACC_FLOAT>,
                 PODHIP::nblocks_for(Ni, 64), 256, 0, stream, pca.data(), ks.data(), kn.data(),
                 B.data(), proj.data(), ledges.data(), redges.data(), tyai.data(), la, Ni) &&
             PODHIP::launch(PODHIP::localenv_atom_split_kernel<KK_FLOAT, KK_ACC_FLOAT>,
                 PODHIP::nblocks_for(Ni, 64), 256, 0, stream, ei.data(), D.data(),
                 dD_dpca.data(), U.data(), pca.data(), ks.data(), kn.data(), B.data(),
                 cent.data(), cefs.data(), invlcut2.data(), invrcut2.data(), tyai.data(), la, Ni))
          : (PODHIP::launch(PODHIP::localenv_pca_kernel<KK_FLOAT, KK_ACC_FLOAT>,
                 PODHIP::nblocks_for(Ni, 256), 256, 0, stream, pca.data(), ks.data(), kn.data(),
                 B.data(), proj.data(), ledges.data(), redges.data(), tyai.data(), la, Ni) &&
             PODHIP::launch(PODHIP::localenv_atom_kernel<KK_FLOAT, KK_ACC_FLOAT>,
                 PODHIP::nblocks_for(Ni, 256), 256, 0, stream, ei.data(), D.data(),
                 dD_dpca.data(), U.data(), pca.data(), ks.data(), kn.data(), B.data(),
                 cent.data(), cefs.data(), invlcut2.data(), invrcut2.data(), tyai.data(), la, Ni));
      if (ok &&
          PODHIP::launch(PODHIP::localenv_cb_kernel<KK_FLOAT, KK_ACC_FLOAT>,
              PODHIP::nblocks_for((long) Ni * nDes, 256), 256, 0, stream, cb.data(), D.data(),
              U.data(), ks.data(), kn.data(), proj.data(), cefs.data(), tyai.data(), la, Ni))
        return;
    }
  }
#endif

  // 1) PCA + active cluster search
  Kokkos::parallel_for("pca_and_active_clusters",
    Kokkos::RangePolicy<DeviceType>(0, Ni),
    KOKKOS_LAMBDA(const int i) {
      const int typei = tyai[i];
      const int ncdt  = typei * nDes;
      const int ncct  = typei * nCls;

      KK_ACC_FLOAT pcai = 0.0;
      for (int m = 0; m < nDes; ++m) {
        pcai += static_cast<KK_ACC_FLOAT>(proj[ncdt + m] * B[i + Ni * m]);
      }
      pca[i] = static_cast<KK_FLOAT>(pcai);

      // lower_bound on redges[ncct + k], ks in [0, nCls-1]
      int lo = 0, hi = nCls;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (static_cast<KK_ACC_FLOAT>(redges[ncct + mid]) < pcai) lo = mid + 1;
        else                           hi = mid;
      }
      const int ksi = lo;

      // upper_bound on ledges[ncct + k], ke in [ks+1, min(ks+nActMax,nCls))
      hi = ksi + nActMax;
      if (hi > nCls) hi = nCls;
      lo++;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (static_cast<KK_ACC_FLOAT>(ledges[ncct + mid]) <= pcai) lo = mid + 1;
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

      KK_ACC_FLOAT sumfDi = 0.0;
      KK_ACC_FLOAT S      = 0.0;

      const KK_FLOAT pcai = pca[i];
      KK_FLOAT fcut, dfcut;
      for (int k = 0; k < kni; ++k) {
        const int kg = k + kncct;

        const KK_FLOAT pc        = pcai - cent[kg];
        const KK_FLOAT inv_rcut2 = (pc >= static_cast<KK_FLOAT>(0.0)) ? invrcut2[kg] : invlcut2[kg];

        cluster_cutoff_hat(pc, inv_rcut2, fcut, dfcut);
        //cluster_cutoff_poly_sq(pc, inv_rcut2, fcut, dfcut);

        const KK_FLOAT invDk = static_cast<KK_FLOAT>(1.0) / (pc * pc + static_cast<KK_FLOAT>(1e-20));
        const KK_FLOAT fDk   = fcut * invDk;
        const KK_FLOAT dfDkdpc = static_cast<KK_FLOAT>(2.0) * pc * invDk * (dfcut - fDk);

        const int idx = i + Ni * k;
        D[idx]       = fDk;
        dD_dpca[idx] = dfDkdpc;

        sumfDi += static_cast<KK_ACC_FLOAT>(fDk);
        S      += static_cast<KK_ACC_FLOAT>(dfDkdpc);
      }
      sumfDi = static_cast<KK_ACC_FLOAT>(1.0) / sumfDi;

      KK_ACC_FLOAT eival = static_cast<KK_ACC_FLOAT>(cefs[nc]);
      KK_ACC_FLOAT T     = 0.0;
      KK_ACC_FLOAT A     = 0.0;
      for (int k = 0; k < kni; ++k) {
        int idx = knc + k*nDes;
        KK_ACC_FLOAT sumE = 0.0;
        for (int m = 0; m < nDes; ++m)
          sumE += static_cast<KK_ACC_FLOAT>(cefs[m + idx] * B[i + Ni*m]);

        idx = i + Ni * k;
        const KK_FLOAT Pk  = static_cast<KK_FLOAT>(static_cast<KK_ACC_FLOAT>(D[idx]) * sumfDi);
        const KK_FLOAT cpk = static_cast<KK_FLOAT>(sumE * sumfDi);

        D[idx] = Pk;

        eival += sumE * static_cast<KK_ACC_FLOAT>(Pk);
        T     += static_cast<KK_ACC_FLOAT>(cpk * Pk);
        A     += static_cast<KK_ACC_FLOAT>(cpk * dD_dpca[idx]);
      }
      ei[i] = static_cast<KK_FLOAT>(eival);
      U[i]  = static_cast<KK_FLOAT>(A - S * T);
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

      KK_ACC_FLOAT sum = static_cast<KK_ACC_FLOAT>(U[i] * proj[m + nDes*typei]);
      for (int k = 0; k < kni; ++k)
        sum += static_cast<KK_ACC_FLOAT>(cefs[knc + k*nDes] * D[i + Ni*k]);

      cb[i + Ni*m] = static_cast<KK_FLOAT>(sum);
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

#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_PORT)) {
      const hipStream_t stream = Kokkos::HIP().hip_stream();
      const PODHIP::EnvArgs ea{nCom, nCls, nDes, nCoeff};
      if (PODHIP::launch(PODHIP::env_pca_kernel<KK_FLOAT>, PODHIP::nblocks_for((long) Ni * nCom, 256),
              256, 0, stream, pca.data(), B.data(), proj.data(), tyai.data(), ea, Ni) &&
          PODHIP::launch(PODHIP::env_prob_energy_kernel<KK_FLOAT>, PODHIP::nblocks_for(Ni, 256), 256,
              0, stream, ei.data(), P.data(), D.data(), sumD.data(), pca.data(), B.data(),
              cent.data(), cefs.data(), tyai.data(), ea, Ni) &&
          PODHIP::launch(PODHIP::env_cp_kernel<KK_FLOAT>, PODHIP::nblocks_for((long) Ni * nCls, 256),
              256, 0, stream, cp.data(), B.data(), cefs.data(), tyai.data(), ea, Ni) &&
          PODHIP::launch(PODHIP::env_cb_kernel<KK_FLOAT>, PODHIP::nblocks_for((long) Ni * nDes, 256),
              256, 0, stream, cb.data(), P.data(), D.data(), sumD.data(), cp.data(), pca.data(),
              cent.data(), proj.data(), cefs.data(), tyai.data(), ea, Ni))
        return;
    }
  }
#endif

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
    KK_FLOAT sum = static_cast<KK_FLOAT>(1e-20);
    for (int k = 0; k < nCom; k++) {
      KK_FLOAT c = cent[k + j * nCom + nCls*nCom*typei];
      KK_FLOAT p = pca[i + Ni*k];
      sum += (p - c) * (p - c);
    }
    D[i + Ni*j] = static_cast<KK_FLOAT>(1.0) / sum;
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

  // calculate base descriptors and their derivatives with respect to atom coordinates
  blockatom_base_descriptors(bd, Ni, Nij);
  if (fused_abort) return;

  if (localeapod) blockatom_local_environment_descriptors(l_ei, cb, bd, Ni);
  else if (eapod) blockatom_environment_descriptors(l_ei, cb, bd, Ni);
  else            blockatom_base_coefficients(l_ei, cb, bd, Ni);

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

  // force coefficients of the 3- and 4-body descriptors.  with HIP_FUSE_FC the native force
  // kernel computes them itself (fc_deferred), and they are only computed here if it does not run
  bool fc_done = false, fc_deferred = false;
#ifdef KOKKOS_ENABLE_HIP
  PODHIP::FcArgs<KK_FLOAT> fca{};
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (nl3 > 0) && (Nij > 1)) {
      fca = PODHIP::FcArgs<KK_FLOAT>{cb3.data(), cb4.data(), sumU.data(), pc3.data(), pn3.data(),
                                     p3_active.data(), pa4.data(), pb4.data(), pc4.data(),
                                     nelements, nrbf3, nabf3_active, nrbf4, nabf4, K3, Q4, Ni,
                                     ((nl4 > 0) && (Nij > 2)) ? 1 : 0};
      fc_deferred = angular_otf && (hip_flags & HIP_FUSE_FC);
      if (!fc_deferred)
        fc_done =
            PODHIP::forcecoeff34<KK_FLOAT>(Kokkos::HIP().hip_stream(), forcecoeff.data(), fca);
    }
  }
#endif
  auto compute_forcecoeff = [&]() {
#ifdef KOKKOS_ENABLE_HIP
    if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
      if (fc_deferred)
        fc_done =
            PODHIP::forcecoeff34<KK_FLOAT>(Kokkos::HIP().hip_stream(), forcecoeff.data(), fca);
      fc_deferred = false;
    }
#endif
    if (!fc_done && (Nij > 1)) {
      set_array_to_zero(forcecoeff, nelements * nrbf3 * K3 * Ni);
      if (nl3 > 0) threebody_forcecoeff(forcecoeff, cb3, sumU, pc3, pn3, p3_active, nelements, nrbf3, nabf3_active, K3, Ni);
      if ((nl4 > 0) && (Nij>2)) fourbody_forcecoeff(forcecoeff, cb4, sumU, pa4, pb4, pc4, nelements, nrbf3, nrbf4, nabf4, K3, Q4, Ni);
    }
  };
  if (!fc_deferred) compute_forcecoeff();

  // pair forces.  the native kernels add the many-body and 2-body forces of each pair to
  // fij, which tallyforce() and tallystress() then add to the atoms; otherwise
  // allbody_forces() computes both and tallies the forces and the virial itself
  forces_in_fij = false;
  bool tallied = false;
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (angular_otf && (nl3 > 0) && (Nij > 1)) {
      // with HIP_FTALLY the force kernel also adds the 2-body forces, and tallies the pair
      // forces and the virial into the atoms, so that fij is not used
      const bool ftally = hip_acc_used && (hip_flags & HIP_FTALLY);
      PODHIP::ForceTally<KK_FLOAT, KK_ACC_FLOAT> ft{};
      if (ftally) {
        if (vflag_global && ((int) hip_vir.extent(0) < 6 * Ni))
          MemKK::realloc_kokkos(hip_vir, "pair_pod:hip_vir", 6 * Ni);
        ft.f = f.data();
        ft.fs0 = f.stride(0);
        ft.fs1 = f.stride(1);
        ft.vir = vflag_global ? hip_vir.data() : nullptr;
        ft.vatom = vflag_atom ? d_vatom.data() : nullptr;
        ft.vs0 = d_vatom.stride(0);
        ft.vs1 = d_vatom.stride(1);
        ft.ai = ai.data();
        ft.aj = aj.data();
        ft.gia = d_ilist.data() + block_gi1;
        ft.cb2 = cb2.data();
        ft.drbf2 = drbf2.data();
        ft.base2 = use_bspline ? bs_base.data() : nullptr;
        ft.bs = pod_bspline_args(use_bspline && bs_deferred, bs_r0, bs_invh, ti, tj, nelements,
                                 bs_nb, nrbf2, bs_ileft);
        ft.nrbf2 = nrbf2;
        ft.nm2 = use_bspline ? 4 : nrbf2;
        ft.Ni = Ni;
      } else {
        set_array_to_zero(l_fij, 3*Nij);
      }
      const auto rsrc = pod_radial_src(rbf_deferred, rbf, drbf, rbf_hermite_coeffs, hermite_r0,
                                       hermite_invdr, ti, tj, nelements, nhermite_bins, nrbfmax, Nij);
      const bool done = PODHIP::allbody_forces_otf<KK_FLOAT, KK_ACC_FLOAT>(
          Kokkos::HIP().hip_stream(), l_fij.data(), ft, forcecoeff.data(),
          fc_deferred ? &fca : nullptr, rsrc,
          pod_geom(rr, rhat, Nij, pair_stride), tj.data(), numij.data(), pq_m.data(), pq_d.data(), nelements,
          nrbf3, K3, Ni, Nij, (hip_flags & HIP_MFMA_FORCE) != 0);
      if (done) fc_deferred = false;
      if (done && ftally) {
        tallied = true;
        if (vflag_global &&
            !PODHIP::virial_reduce<KK_ACC_FLOAT>(Kokkos::HIP().hip_stream(), hip_acc.data(),
                                                 hip_vir.data(), Ni))
          error->one(FLERR, "pair pod/kk: summing the virial on the GPU failed");
      } else if (done) {
        forces_in_fij = true;
        if (use_bspline) twobody_forces_bspline(l_fij, cb2, drbf2, rhat, rr, bs_base, idxi, tj, nrbf2, Ni, Nij);
        else             twobody_forces(l_fij, cb2, drbf2, rhat, rr, idxi, tj, nrbf2, Ni, Nij);
      }
    }
  }
#endif
  if (!forces_in_fij && !tallied) {
    if (fused_abort_needed()) return;
    if (fc_deferred) compute_forcecoeff();
    ensure_rbf(Nij);
    ensure_bspline(Nij);
    allbody_forces(l_fij, cb2, Ni, Nij);
  }
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::tallyforce(t_pod_1d l_fij, t_pod_1i l_ai, t_pod_1i l_aj, int Nij)
{
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (native_hip && (hip_flags & HIP_TALLY) &&
        PODHIP::tallyforce<KK_FLOAT, KK_ACC_FLOAT>(Kokkos::HIP().hip_stream(), f.data(),
                                                    f.stride(0), f.stride(1), l_fij.data(),
                                                    l_ai.data(), l_aj.data(), numij.data(), ni))
      return;
  }
#endif

  auto l_f = f;
  Kokkos::parallel_for("TallyForce", Kokkos::RangePolicy<DeviceType>(0,Nij), KOKKOS_LAMBDA(int n) {
    int im = l_ai(n);
    int jm = l_aj(n);
    int n3 = 3*n;
    KK_FLOAT fx = l_fij(n3 + 0);
    KK_FLOAT fy = l_fij(n3 + 1);
    KK_FLOAT fz = l_fij(n3 + 2);
    Kokkos::atomic_add(&l_f(im, 0), static_cast<KK_ACC_FLOAT>(fx));
    Kokkos::atomic_add(&l_f(im, 1), static_cast<KK_ACC_FLOAT>(fy));
    Kokkos::atomic_add(&l_f(im, 2), static_cast<KK_ACC_FLOAT>(fz));
    Kokkos::atomic_sub(&l_f(jm, 0), static_cast<KK_ACC_FLOAT>(fx));
    Kokkos::atomic_sub(&l_f(jm, 1), static_cast<KK_ACC_FLOAT>(fy));
    Kokkos::atomic_sub(&l_f(jm, 2), static_cast<KK_ACC_FLOAT>(fz));
  });
}

template<class DeviceType>
void PairPODKokkos<DeviceType>::tallyenergy(t_pod_1d l_ei, int istart, int Ni)
{
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (hip_acc_used && !eflag_global && !eflag_atom) return;
    if (hip_acc_used &&
        PODHIP::launch(PODHIP::tallyenergy_kernel<KK_FLOAT, KK_ACC_FLOAT>,
            MIN(PODHIP::nblocks_for(Ni, 256), 128), 256, 0, Kokkos::HIP().hip_stream(), hip_acc.data(),
            eflag_atom ? d_eatom.data() : (KK_ACC_FLOAT *) nullptr, l_ei.data(), istart, Ni,
            eflag_global ? 1 : 0))
      return;
  }
#endif

  auto l_eatom = d_eatom;

  // For global energy tally
  if (eflag_global) {
    KK_FLOAT local_eng_vdwl = 0.0;
    Kokkos::parallel_reduce("GlobalEnergyTally", Kokkos::RangePolicy<DeviceType>(0,Ni), KOKKOS_LAMBDA(int k, KK_FLOAT& update) {
      update += l_ei(k);
    }, local_eng_vdwl);

    // Update global energy on the host after the parallel region
    eng_vdwl += static_cast<double>(local_eng_vdwl);
  }

  // For per-atom energy tally
  if (eflag_atom) {
    Kokkos::parallel_for("PerAtomEnergyTally", Kokkos::RangePolicy<DeviceType>(0,Ni), KOKKOS_LAMBDA(int k) {
      l_eatom(istart + k) += static_cast<KK_ACC_FLOAT>(l_ei(k));
    });
  }
}

template<class DeviceType, class View1D>
struct PODGlobalStressFunctor {
  View1D l_fij, l_rr, l_rhat;
  int Nij;

  PODGlobalStressFunctor(View1D fij, View1D rr, View1D rhat, int n) :
    l_fij(fij), l_rr(rr), l_rhat(rhat), Nij(n) {}

  struct value_type {
    KK_ACC_FLOAT v[6];
  };

  KOKKOS_INLINE_FUNCTION
  void init(value_type& dst) const {
    for (int d = 0; d < 6; ++d) dst.v[d] = 0;
  }

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dst, const value_type& src) const {
    for (int d = 0; d < 6; ++d) dst.v[d] += src.v[d];
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k, value_type& upd) const {
    const KK_FLOAT r = l_rr(k);
    const KK_FLOAT rx = r * l_rhat(k);
    const KK_FLOAT ry = r * l_rhat(k + Nij);
    const KK_FLOAT rz = r * l_rhat(k + 2*Nij);

    const int k3 = 3 * k;
    const KK_FLOAT fx = l_fij(k3 + 0);
    const KK_FLOAT fy = l_fij(k3 + 1);
    const KK_FLOAT fz = l_fij(k3 + 2);

    upd.v[0] += static_cast<KK_ACC_FLOAT>(rx * fx);
    upd.v[1] += static_cast<KK_ACC_FLOAT>(ry * fy);
    upd.v[2] += static_cast<KK_ACC_FLOAT>(rz * fz);
    upd.v[3] += static_cast<KK_ACC_FLOAT>(rx * fy);
    upd.v[4] += static_cast<KK_ACC_FLOAT>(rx * fz);
    upd.v[5] += static_cast<KK_ACC_FLOAT>(ry * fz);
  }
};

// virial of the pair forces in fij, used when allbody_forces() did not tally them
template<class DeviceType>
void PairPODKokkos<DeviceType>::tallystress(
    t_pod_1d l_fij, t_pod_1d l_rr, t_pod_1d l_rhat, t_pod_1i l_ai, t_pod_1i l_aj, int Nij)
{
#ifdef KOKKOS_ENABLE_HIP
  if constexpr (std::is_same_v<typename DeviceType::execution_space, Kokkos::HIP>) {
    if (hip_acc_used && !vflag_global && !vflag_atom) return;
    if (hip_acc_used &&
        PODHIP::launch(PODHIP::tallystress_kernel<KK_FLOAT, KK_ACC_FLOAT>, PODHIP::nblocks_for(Nij, 256),
            256, 0, Kokkos::HIP().hip_stream(), hip_acc.data(),
            vflag_atom ? d_vatom.data() : (KK_ACC_FLOAT *) nullptr, (int) d_vatom.stride(0),
            (int) d_vatom.stride(1), l_fij.data(), pod_geom(l_rr, l_rhat, Nij, pair_stride), l_ai.data(),
            l_aj.data(), Nij, vflag_global ? 1 : 0))
      return;
  }
#endif

  auto l_vatom = d_vatom;

  if (vflag_global) {
    using Functor = PODGlobalStressFunctor<DeviceType, t_pod_1d>;
    typename Functor::value_type sumv;

    Kokkos::parallel_reduce("GlobalStressTally",
      Kokkos::RangePolicy<DeviceType>(0, Nij),
      Functor(l_fij, l_rr, l_rhat, Nij),
      sumv
    );

    for (int d = 0; d < 6; ++d) virial[d] -= static_cast<double>(sumv.v[d]);
  }

  if (vflag_atom) {
    Kokkos::parallel_for("PerAtomStressTally",
      Kokkos::RangePolicy<DeviceType>(0, Nij), KOKKOS_LAMBDA(const int k) {
        const int i  = l_ai(k);
        const int j  = l_aj(k);
        const KK_FLOAT h = static_cast<KK_FLOAT>(-0.5) * l_rr(k);
        const KK_FLOAT rx = h * l_rhat(k);
        const KK_FLOAT ry = h * l_rhat(k + Nij);
        const KK_FLOAT rz = h * l_rhat(k + 2*Nij);

        const int k3 = 3 * k;
        const KK_FLOAT fx = l_fij(k3 + 0);
        const KK_FLOAT fy = l_fij(k3 + 1);
        const KK_FLOAT fz = l_fij(k3 + 2);

        KK_FLOAT v_local[6];
        v_local[0] = rx * fx;
        v_local[1] = ry * fy;
        v_local[2] = rz * fz;
        v_local[3] = rx * fy;
        v_local[4] = rx * fz;
        v_local[5] = ry * fz;

        for (int d = 0; d < 6; ++d) {
          const KK_FLOAT v_l = v_local[d];
          Kokkos::atomic_add(&l_vatom(i, d), static_cast<KK_ACC_FLOAT>(v_l));
          Kokkos::atomic_add(&l_vatom(j, d), static_cast<KK_ACC_FLOAT>(v_l));
        }
      });
  }
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
