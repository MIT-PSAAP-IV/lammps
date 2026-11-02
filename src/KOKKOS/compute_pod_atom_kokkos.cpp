// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   (license header as in the other files)
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Kokkos port of compute pod/atom, UQ output of local EA-POD.
   A private PairPODKokkos object is the kernel engine. Per-pair vectors
   J_ij*c are obtained without bdd: the engine's force view points at a
   scratch buffer and ai/aj are remapped so that pair n writes row n.
------------------------------------------------------------------------- */

#include "compute_pod_atom_kokkos.h"
#include "pair_pod_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "neighbor_kokkos.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "eapod.h"

using namespace LAMMPS_NS;

static constexpr int UQ_MAX_ACT  = 8;   // POD_MAX_ACT_CLS in pair_pod_kokkos.cpp
static constexpr int UQ_MAX_COMM = 8;   // max committee size m

// cluster hat (p=2,q=4), same as MD; dfcut = (1/(2pc)) dfcut/dpc
KOKKOS_INLINE_FUNCTION
static void uq_cluster_hat(const double pc2, const double inv_cut2, double &fcut, double &dfcut)
{
  const double y2   = pc2 * inv_cut2;
  const double omy  = 1.0 - y2 * y2;
  const double omy2 = omy * omy;
  fcut  = omy2 * omy2;
  dfcut = -8.0 * inv_cut2 * y2 * omy2 * omy;
}

// W = L^{-1}, both row-packed lower triangular (row r at offset r(r+1)/2)
static void invert_packed_lower(const double *L, double *W, int M)
{
  for (int j = 0; j < M; j++) {
    const int jj = j*(j+1)/2;
    W[jj + j] = 1.0 / L[jj + j];
    for (int r = j+1; r < M; r++) {
      const int ro = r*(r+1)/2;
      double s = 0.0;
      for (int c = j; c < r; c++) s += L[ro + c] * W[c*(c+1)/2 + j];
      W[ro + j] = -s / L[ro + r];
    }
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputePODAtomKokkos<DeviceType>::ComputePODAtomKokkos(LAMMPS *lmp, int narg, char **arg) :
  ComputePODAtom(lmp, narg, arg), podkk(nullptr)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
  host_flag = (execution_space == HostKK);

  have_gram = 0;
  ncomm = podptr->ncommittee;
  nimax_uq = nijmax_uq = 0;

  // device path: UQ output of local EA-POD; everything else uses the CPU code
  use_kk = (!host_flag && podptr->uncertaintyflag && podptr->localeapod) ? 1 : 0;
  if (!use_kk) {
    if (comm->me == 0)
      error->warning(FLERR, "Compute {}: using the CPU code path", style);
    return;
  }
  if (podptr->nMaxActiveClusters > UQ_MAX_ACT)
    error->all(FLERR, "Compute {}: nMaxActiveClusters > {}", style, UQ_MAX_ACT);
  if (ncomm > UQ_MAX_COMM)
    error->all(FLERR, "Compute {}: committee_size > {}", style, UQ_MAX_COMM);

  setup_engine();
  setup_gram();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputePODAtomKokkos<DeviceType>::~ComputePODAtomKokkos()
{
  if (copymode) return;
  if (use_kk) memoryKK->destroy_kokkos(k_pod, pod);   // nulls pod -> base dtor skips it
  delete podkk;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::init()
{
  ComputePODAtom::init();     // checks + occasional full list request
  if (!use_kk) return;        // CPU path keeps a plain host list

  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
}

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::setup_engine()
{
  podkk = new PairPODKokkos<DeviceType>(lmp);
  podkk->copy_from_pod_class(podptr);   // coeffs, Phi/Hermite, PCA, edges, index tables

  // engine->compute() is never called: keep its virial branches off
  podkk->vflag_global = 0;
  podkk->vflag_atom = 0;

  const int nt = atom->ntypes, ne = podptr->nelements;
  MemKK::realloc_kokkos(podkk->d_map, "pod/atom/kk:map", nt + 1);
  auto h_map = Kokkos::create_mirror_view(podkk->d_map);
  for (int i = 0; i <= nt; i++) h_map(i) = map[i];
  Kokkos::deep_copy(podkk->d_map, h_map);

  auto copy2d = [&](typename PairPODKokkos<DeviceType>::t_pod_2d &d, double **src,
                    const char *name) {
    MemKK::realloc_kokkos(d, name, ne, ne);
    auto h = Kokkos::create_mirror_view(d);
    for (int i = 0; i < ne; i++)
      for (int j = 0; j < ne; j++) h(i, j) = src[i][j];
    Kokkos::deep_copy(d, h);
  };
  copy2d(podkk->rin,      podptr->rin,      "pod/atom/kk:rin");
  copy2d(podkk->rcut,     podptr->rcut,     "pod/atom/kk:rcut");
  copy2d(podkk->invrdiff, podptr->invrdiff, "pod/atom/kk:invrdiff");
  copy2d(podkk->rcutsq,   podptr->rcutsq,   "pod/atom/kk:rcutsq");
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::setup_gram()
{
  have_gram = (podptr->Lgram != nullptr) ? 1 : 0;   // EAPOD already warned if missing
  if (!have_gram) {
    MemKK::realloc_kokkos(d_Linv, "pod/atom/kk:Linv", 1);
    MemKK::realloc_kokkos(d_uqref, "pod/atom/kk:uqref", 1);
    return;
  }
  const int M  = podptr->Mdesc;
  const int ng = podptr->nelements * podptr->nClusters;
  const size_t npack = (size_t) M * (M + 1) / 2;

  MemKK::realloc_kokkos(d_Linv, "pod/atom/kk:Linv", ng * npack);
  MemKK::realloc_kokkos(d_uqref, "pod/atom/kk:uqref", 3 * ng);
  auto h_Linv = Kokkos::create_mirror_view(d_Linv);
  auto h_ref  = Kokkos::create_mirror_view(d_uqref);
  for (int g = 0; g < ng; g++)
    invert_packed_lower(&podptr->Lgram[g * npack], h_Linv.data() + g * npack, M);
  for (int q = 0; q < 3 * ng; q++) h_ref(q) = podptr->uqref[q];
  Kokkos::deep_copy(d_Linv, h_Linv);
  Kokkos::deep_copy(d_uqref, h_ref);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::grow_uq_atoms(int Ni)
{
  if (Ni <= nimax_uq) return;
  nimax_uq = Ni;
  const int nac = podkk->nMaxActiveClusters;
  MemKK::realloc_kokkos(d_U,    "pod/atom/kk:U",    Ni);
  MemKK::realloc_kokkos(d_sF2,  "pod/atom/kk:sF2",  Ni);
  MemKK::realloc_kokkos(d_phi2, "pod/atom/kk:phi2", Ni);
  MemKK::realloc_kokkos(d_P,    "pod/atom/kk:P",    Ni * nac);
  MemKK::realloc_kokkos(d_lev,  "pod/atom/kk:lev",  Ni * nac);
  MemKK::realloc_kokkos(d_ks,   "pod/atom/kk:ks",   Ni);
  MemKK::realloc_kokkos(d_kn,   "pod/atom/kk:kn",   Ni);
  MemKK::realloc_kokkos(d_win,  "pod/atom/kk:win",  Ni * ncomm);
}

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::grow_uq_pairs(int Nij)
{
  if (Nij <= nijmax_uq) return;
  nijmax_uq = Nij;
  MemKK::realloc_kokkos(d_fij, "pod/atom/kk:fij", 2 * Nij);
}

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::compute_peratom()
{
  if (!use_kk) {
    atomKK->sync(Host, X_MASK | TYPE_MASK | MASK_MASK);
    ComputePODAtom::compute_peratom();
    return;
  }

  invoked_peratom = update->ntimestep;

  if (atom->nmax > nmax) {
    memoryKK->destroy_kokkos(k_pod, pod);
    nmax = atom->nmax;
    memoryKK->create_kokkos(k_pod, pod, nmax, size_peratom_cols, "pod/atom/kk:pod");
    array_atom = pod;
  }
  d_pod = k_pod.template view<DeviceType>();
  Kokkos::deep_copy(d_pod, 0.0);

  neighbor->build_one(list);
  auto k_list = static_cast<NeighListKokkos<DeviceType> *>(list);

  atomKK->sync(execution_space, X_MASK | TYPE_MASK | MASK_MASK);
  podkk->x    = atomKK->k_x.view<DeviceType>();
  podkk->type = atomKK->k_type.view<DeviceType>();
  mask        = atomKK->k_mask.view<DeviceType>();
  podkk->d_ilist     = k_list->d_ilist;
  podkk->d_numneigh  = k_list->d_numneigh;
  podkk->d_neighbors = k_list->d_neighbors;

  const int inum = list->inum;
  if (inum > 0) {
    int blocks[101];
    int nblocks = podkk->calculateNumberOfIntervals(inum, podkk->atomBlockSize);
    if (nblocks > 100) nblocks = 100;
    podkk->divideInterval(blocks, inum, nblocks);
    int nimax = 0;
    for (int b = 0; b < nblocks; b++) nimax = std::max(nimax, blocks[b+1] - blocks[b]);
    podkk->grow_atoms(nimax);
    grow_uq_atoms(nimax);

    for (int b = 0; b < nblocks; b++) {
      const int gi1 = blocks[b] - 1;
      const int Ni  = blocks[b+1] - blocks[b];

      const int Nij = podkk->NeighborCount(podkk->numij, gi1, Ni);
      if (Nij == 0) continue;                       // isolated atoms -> zeros (as CPU)
      podkk->grow_pairs(Nij);
      grow_uq_pairs(Nij);
      podkk->NeighborList(podkk->rr, podkk->rhat, podkk->numij, podkk->typeai, podkk->idxi,
                          podkk->ai, podkk->aj, podkk->ti, podkk->tj, gi1, Ni, Nij);

      podkk->blockatom_base_descriptors(podkk->bd, Ni, Nij);   // bd(i+Ni*m), sumU

      uq_gate(gi1, Ni);                    // E_i, sigma_E, ext, k*, ks, kn, U
      if (have_gram) uq_leverage(Ni);      // gamma terms

      redirect_pair_forces(Nij);           // pair n -> row n of d_fij
      for (int w = 0; w <= ncomm; w++) {   // m committee passes + 1 gate pass
        fill_cb(w, Ni);
        pair_force_pass(Ni, Nij);
        accumulate_norm2((w < ncomm) ? d_sF2 : d_phi2, Ni);
      }
      uq_finalize(gi1, Ni);                // sigma_F, gamma, phi
    }
  }

  k_pod.template modify<DeviceType>();
  k_pod.sync_host();
}

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::uq_gate(int gi1, int Ni)
{
  const int M = podkk->Mdesc, K = podkk->nClusters, nCoeff = podkk->nCoeffPerElement;
  const int NAC = podkk->nMaxActiveClusters, m = ncomm, gram = have_gram;
  const int l_groupbit = groupbit;

  auto bd = podkk->bd;          auto coef = podkk->coefficients;
  auto proj = podkk->Proj;      auto cent = podkk->Centroids;
  auto ledges = podkk->leftClusterEdges;   auto redges = podkk->rightClusterEdges;
  auto invl = podkk->invLeftClusterRcut2;  auto invr = podkk->invRightClusterRcut2;
  auto typeai = podkk->typeai;  auto numij = podkk->numij;  auto ilist = podkk->d_ilist;
  auto l_mask = mask;  auto pod = d_pod;  auto uqref = d_uqref;
  auto U = d_U, P = d_P, sF2 = d_sF2, phi2 = d_phi2;
  auto ks_ = d_ks, kn_ = d_kn, win = d_win;

  Kokkos::parallel_for("ComputePODAtomKokkos::uq_gate", Kokkos::RangePolicy<DeviceType>(0, Ni),
    KOKKOS_LAMBDA(const int i) {
      const int gi = ilist(gi1 + i);
      U(i) = 0.0;  sF2(i) = 0.0;  phi2(i) = 0.0;  ks_(i) = 0;  kn_(i) = 0;
      if (!(l_mask(gi) & l_groupbit) || (numij(i+1) == numij(i))) return;

      const int typei = typeai(i);
      const int ncct  = K * typei;         // nComponents == 1 (local EA-POD)
      const int cn    = nCoeff * typei;    // first coefficient of eleement i
      const int c0    = 1 + cn;            // first coefficient of cluster 0

      double pca = 0.0;
      for (int j = 0; j < M; j++) pca += proj(M*typei + j) * bd(i + Ni*j);

      // gate support (same search as the MD kernel)
      int lo = 0, hi = K;
      while (lo < hi) { const int mid = (lo+hi) >> 1; if (redges(ncct+mid) < pca) lo = mid+1; else hi = mid; }
      const int ks = lo;
      hi = (ks + NAC < K) ? ks + NAC : K;
      lo = ks + 1;
      while (lo < hi) { const int mid = (lo+hi) >> 1; if (ledges(ncct+mid) <= pca) lo = mid+1; else hi = mid; }
      const int kn = lo - ks;

      double Pk[UQ_MAX_ACT], dPk[UQ_MAX_ACT];
      double S = 0.0, dS = 0.0;
      for (int k = 0; k < kn; k++) {
        const int kg = ncct + ks + k;
        const double pc = pca - cent(kg);
        const double inv_cut2 = (pc >= 0.0) ? invr(kg) : invl(kg);
        double fcut, dfcut;
        uq_cluster_hat(pc*pc, inv_cut2, fcut, dfcut);
        const double invD = 1.0 / (pc*pc + 1e-20);
        Pk[k]  = fcut * invD;
        dPk[k] = 2.0 * pc * invD * (dfcut - Pk[k]);
        S += Pk[k];  dS += dPk[k];
      }
      const double invS = 1.0 / S;

      double Ebar = 0.0, Ui = 0.0;
      for (int k = 0; k < kn; k++) {
        Pk[k] *= invS;
        dPk[k] = (dPk[k] - Pk[k]*dS) * invS;
        const int cbase = c0 + (ks + k)*M;
        double e = 0.0;
        for (int j = 0; j < M; j++) e += coef(cbase + j) * bd(i + Ni*j);
        Ebar += Pk[k] * e;
        Ui   += dPk[k] * e;
        P(i + Ni*k) = Pk[k];
      }

      // committee window: m nearest sorted centroids, gate independent
      lo = 0; hi = K;
      while (lo < hi) { const int mid = (lo+hi) >> 1; if (cent(ncct+mid) < pca) lo = mid+1; else hi = mid; }
      int r = lo, l = lo - 1;
      double ew[UQ_MAX_COMM], emean = 0.0;
      for (int w = 0; w < m; w++) {
        int kw;
        if ((r >= K) || ((l >= 0) && (pca - cent(ncct+l) <= cent(ncct+r) - pca))) kw = l--;
        else kw = r++;
        win(i + Ni*w) = kw;
        double e = 0.0;
        for (int j = 0; j < M; j++) e += coef(c0 + kw*M + j) * bd(i + Ni*j);
        ew[w] = e;  emean += e;
      }
      emean /= m;
      double sE2 = 0.0;
      for (int w = 0; w < m; w++) sE2 += (ew[w] - emean) * (ew[w] - emean);
      const int kstar = win(i);
      double sigE = Kokkos::sqrt(sE2 / m);
      if (gram) sigE /= uqref(3*(ncct + kstar) + 1);

      // latent-axis extrapolation
      const double clo = cent(ncct), chi = cent(ncct + K - 1);
      double ext = 0.0;
      if (pca < clo)      { const double d = cent(ncct+1) - clo;   ext = (clo - pca) / (d > 1e-20 ? d : 1e-20); }
      else if (pca > chi) { const double d = chi - cent(ncct+K-2); ext = (pca - chi) / (d > 1e-20 ? d : 1e-20); }

      pod(gi, 0) = coef(cn) + Ebar;
      pod(gi, 1) = sigE;
      pod(gi, 5) = ext;
      pod(gi, 6) = pca;
      pod(gi, 7) = kstar;
      pod(gi, 8) = ks;
      pod(gi, 9) = kn;
      U(i) = Ui;  ks_(i) = ks;  kn_(i) = kn;
    });
}

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::uq_leverage(int Ni)
{
  const int M = podkk->Mdesc, K = podkk->nClusters, NAC = podkk->nMaxActiveClusters;
  const long npack = (long) M * (M + 1) / 2;
  auto bd = podkk->bd;  auto typeai = podkk->typeai;
  auto P = d_P, lev = d_lev, Linv = d_Linv, uqref = d_uqref;
  auto ks = d_ks, kn = d_kn;

  Kokkos::parallel_for("ComputePODAtomKokkos::uq_leverage",
    Kokkos::RangePolicy<DeviceType>(0, Ni * NAC), KOKKOS_LAMBDA(const int idx) {
      const int i = idx % Ni, k = idx / Ni;
      double val = 0.0;
      if ((k < kn(i)) && (P(i + Ni*k) >= 1e-6)) {
        const int g = K*typeai(i) + ks(i) + k;
        const long off = g * npack;
        double sum = 0.0;
        for (int r = 0; r < M; r++) {
          const long ro = off + (long) r*(r+1)/2;
          double s = 0.0;
          for (int c = 0; c <= r; c++) s += Linv(ro + c) * bd(i + Ni*c);
          sum += s * s;
        }
        val = P(i + Ni*k) * sum / uqref(3*g);
      }
      lev(idx) = val;
    });
}

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::redirect_pair_forces(int Nij)
{
  // allbody_forces: f(ai(n)) += f_ij, f(aj(n)) -= f_ij  ->  row n holds f_ij,
  // rows Nij..2Nij-1 are a sink. ai/aj are not read by the descriptor kernels.
  auto ai = podkk->ai;
  auto aj = podkk->aj;
  Kokkos::parallel_for("ComputePODAtomKokkos::redirect", Kokkos::RangePolicy<DeviceType>(0, Nij),
    KOKKOS_LAMBDA(const int n) { ai(n) = n; aj(n) = Nij + n; });
  podkk->f = d_fij;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::fill_cb(int w, int Ni)
{
  // w < m : cb = c_{win(w)} - mean_w' c_{win(w')}   (sigma_F pass)
  // w == m: cb = Proj * U                            (phi pass)
  const int M = podkk->Mdesc, nCoeff = podkk->nCoeffPerElement, m = ncomm;
  auto cb = podkk->cb;  auto coef = podkk->coefficients;  auto proj = podkk->Proj;
  auto typeai = podkk->typeai;  auto U = d_U;  auto win = d_win;  auto kn = d_kn;

  Kokkos::parallel_for("ComputePODAtomKokkos::fill_cb", Kokkos::RangePolicy<DeviceType>(0, Ni * M),
    KOKKOS_LAMBDA(const int idx) {
      const int n = idx % Ni, j = idx / Ni;   // n fastest -> coalesced writes
      double c = 0.0;
      if (kn(n) > 0) {
        const int typei = typeai(n);
        if (w < m) {
          const int base = 1 + nCoeff*typei + j;
          double cbar = 0.0;
          for (int v = 0; v < m; v++) cbar += coef(base + M*win(n + Ni*v));
          c = coef(base + M*win(n + Ni*w)) - cbar / m;
        } else {
          c = proj(M*typei + j) * U(n);
        }
      }
      cb(n + Ni*j) = c;
    });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::pair_force_pass(int Ni, int Nij)
{
  auto p = podkk;
  using t_1d = typename PairPODKokkos<DeviceType>::t_pod_1d;
  const int o3 = p->nl2, o4 = o3 + p->nl3, o33 = o4 + p->nl4;
  const int o34 = o33 + p->nl33, o44 = o34 + p->nl34, oend = o44 + p->nl44;
  auto seg = [Ni](t_1d v, int a, int b) -> t_1d {
    return Kokkos::subview(v, std::make_pair(Ni * a, Ni * b));
  };
  t_1d d3 = seg(p->bd, o3, o4), d4 = seg(p->bd, o4, o33);
  t_1d cb2 = seg(p->cb, 0, o3), cb3 = seg(p->cb, o3, o4), cb4 = seg(p->cb, o4, o33);
  t_1d cb33 = seg(p->cb, o33, o34), cb34 = seg(p->cb, o34, o44), cb44 = seg(p->cb, o44, oend);

  if (p->nl33 > 0) p->crossdesc_reduction(cb3, cb3, cb33, d3, d3, p->ind33l, p->ind33r, p->nl33, Ni);
  if (p->nl34 > 0) p->crossdesc_reduction(cb3, cb4, cb34, d3, d4, p->ind34l, p->ind34r, p->nl34, Ni);
  if (p->nl44 > 0) p->crossdesc_reduction(cb4, cb4, cb44, d4, d4, p->ind44l, p->ind44r, p->nl44, Ni);

  p->set_array_to_zero(p->forcecoeff, p->nelements * p->nrbf3 * p->K3 * Ni);
  if (p->nl3 > 0)
    p->threebody_forcecoeff(p->forcecoeff, cb3, p->sumU, p->pc3, p->pn3, p->p3_active,
                            p->nelements, p->nrbf3, p->nabf3_active, p->K3, Ni);
  if (p->nl4 > 0)
    p->fourbody_forcecoeff(p->forcecoeff, cb4, p->sumU, p->pa4, p->pb4, p->pc4, p->nelements,
                           p->nrbf3, p->nrbf4, p->nabf4, p->K3, p->Q4, Ni);

  Kokkos::deep_copy(d_fij, 0.0);
  p->allbody_forces(cb2, Ni, Nij);     // fused 2-body + many-body -> d_fij rows
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::accumulate_norm2(t_uq_1d acc, int Ni)
{
  auto fij = d_fij;  auto numij = podkk->numij;  auto kn = d_kn;
  Kokkos::parallel_for("ComputePODAtomKokkos::norm2", Kokkos::RangePolicy<DeviceType>(0, Ni),
    KOKKOS_LAMBDA(const int i) {
      if (kn(i) == 0) return;
      double s = 0.0;
      for (int n = numij(i); n < numij(i+1); n++) {
        const double fx = fij(n,0), fy = fij(n,1), fz = fij(n,2);
        s += fx*fx + fy*fy + fz*fz;
      }
      acc(i) += s;
    });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputePODAtomKokkos<DeviceType>::uq_finalize(int gi1, int Ni)
{
  const int K = podkk->nClusters, NAC = podkk->nMaxActiveClusters, m = ncomm, gram = have_gram;
  auto ilist = podkk->d_ilist;  auto typeai = podkk->typeai;
  auto pod = d_pod;  auto uqref = d_uqref;  auto lev = d_lev;
  auto sF2 = d_sF2, phi2 = d_phi2;  auto kn = d_kn, win = d_win;

  Kokkos::parallel_for("ComputePODAtomKokkos::uq_finalize", Kokkos::RangePolicy<DeviceType>(0, Ni),
    KOKKOS_LAMBDA(const int i) {
      if (kn(i) == 0) return;
      const int gi = ilist(gi1 + i);
      const int g  = K*typeai(i) + win(i);     // k*
      double sigF = Kokkos::sqrt(sF2(i) / m);
      double gam = -1.0;
      if (gram) {
        sigF /= uqref(3*g + 2);
        gam = 0.0;
        for (int k = 0; k < NAC; k++) gam += lev(i + Ni*k);   // zero for k >= kn
      }
      pod(gi, 2) = sigF;
      pod(gi, 3) = gam;
      pod(gi, 4) = Kokkos::sqrt(phi2(i));
    });
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double ComputePODAtomKokkos<DeviceType>::memory_usage()
{
  double bytes = 0.0;
  if (use_kk) {
    bytes += (double) nmax * size_peratom_cols * sizeof(double);
    bytes += (double) d_Linv.span() * sizeof(double);
    bytes += (double) nimax_uq * (3 + 2*UQ_MAX_ACT) * sizeof(double);
    bytes += (double) nimax_uq * (2 + ncomm) * sizeof(int);
    bytes += (double) 6 * nijmax_uq * sizeof(KK_ACC_FLOAT);
  }
  return bytes;   // engine buffers not counted
}

namespace LAMMPS_NS {
template class ComputePODAtomKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputePODAtomKokkos<LMPHostType>;
#endif
}













