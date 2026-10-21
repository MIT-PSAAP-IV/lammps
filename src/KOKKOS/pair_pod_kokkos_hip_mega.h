/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// STUDY CODE (branch mega-study): the whole native pair pod/kk step of a
// one-element model with Hermite many-body radial functions, B-spline 2-body
// functions, local EA-POD and 3+4-body terms (the Hf benchmark) in one
// kernel, one wavefront per atom.  The stages are the bodies of the
// existing native kernels, with the pairs of the atom and all per-atom
// intermediates kept in LDS:
//
//   0 neighbors      -> pair cache (r, rhat, j) of atom i          [P]
//   1 2-body desc    -> descriptors d2                             [D]
//   2 radial-angular -> sumU                                       [U]
//   3 3/4-body desc  -> descriptors d3, d4                         [D]
//   4 local EA-POD   -> energy ei (global), coefficients cb        [D]
//   5 force coeffs   -> forcecoeff                                 [F]
//   6 forces         -> f (global atomics), per-atom virial (global)
//
// mega_kernel<S0, S1> runs stages S0..S1.  A range that does not start at 0
// reads its live inputs from a global spill area, one that does not end at 6
// writes its live outputs there, so the pipeline can be split at any stage
// boundary (with the same code) to measure what fusion saves and costs.

#ifndef LMP_PAIR_POD_KOKKOS_HIP_MEGA_H
#define LMP_PAIR_POD_KOKKOS_HIP_MEGA_H

#ifdef KOKKOS_ENABLE_HIP

namespace LAMMPS_NS {
namespace PODHIP {

enum { MS_NEIGH = 0, MS_TWOBODY, MS_RAS, MS_DESC, MS_LOCALENV, MS_FC, MS_FORCE, MS_N };

static constexpr int MEGA_RTM = 2;    // K3 <= 32 (angular degree <= 3)
static constexpr int MEGA_CSM = 2;    // nelements * nrbf3 <= 8
static constexpr int MEGA_FT = 12;                // 4-body terms per monomial (FREG=2 recipe)
static constexpr int MEGA_FRI = 2 + 3 * MEGA_FT;  // ints per monomial in the recipe
static constexpr int MEGA_FRD = 1 + MEGA_FT;      // doubles per monomial in the recipe

// pairs beyond cap (study of the code variants): 0 none (dropped, reported), 1 pool (pool full:
// dropped, reported), 2 pool, and when it is full the windows found again from the neighbor list
#ifndef LMP_POD_MEGA_SPILLV
#define LMP_POD_MEGA_SPILLV 1
#endif
// M1/M2 variants: NOZ 1 = no LDS array of zero element indices (one element), S6POW / S2POW
// 1 = power tables in LDS for the force / radial-angular stage, 2 = the same without the
// constant x^0 = 1 entries, 0 = monomials formed per lane
#ifndef LMP_POD_MEGA_NOZ
#define LMP_POD_MEGA_NOZ 1
#endif
#ifndef LMP_POD_MEGA_S6POW
#define LMP_POD_MEGA_S6POW 2
#endif
#ifndef LMP_POD_MEGA_S2POW
#define LMP_POD_MEGA_S2POW 1
#endif
// N1: S2FILL 1 = the radial-angular tile is filled by all lanes, one (pair, radial function)
// or (pair, direction) per lane (old path a539c81a39), 0 = one lane per pair
#ifndef LMP_POD_MEGA_S2FILL
#define LMP_POD_MEGA_S2FILL 1
#endif
// N2: local EA-POD with at most MEGA_KM active clusters; LPRE = stage at whose start the
// cluster tables and the projection are loaded into registers (2, 3 or 4; 0 = not preloaded,
// read from memory in stage 4) (old path c89843f8b6)
#ifndef LMP_POD_MEGA_LPRE
#define LMP_POD_MEGA_LPRE 3
#endif
static constexpr int MEGA_KM = 4;
// N4: COLD 1 = the reload of a pair window (atoms with more than cap pairs) is a call to a
// function that is not inlined, so that its registers do not add to the force stage's;
// S6OVL 1 = the force stage scratch starts after the 2-body coefficients cb2, over cb3/cb4
// (dead once the force coefficients are in registers) and the stage scratch
#ifndef LMP_POD_MEGA_COLD
#define LMP_POD_MEGA_COLD 0
#endif
#ifndef LMP_POD_MEGA_S6OVL
#define LMP_POD_MEGA_S6OVL 0
#endif

// copy nw pairs of a window from the pool to the LDS cache (rhat strides cap and pn)
template <typename T>
__device__ __attribute__((noinline)) void mega_reload(T *r_s, T *rh_s, int *aj_s, const T *gr,
                                                      const T *gh, const int *ga, const int nw,
                                                      const int cap, const int pn)
{
  for (int t = threadIdx.x; t < nw; t += 64) {
    r_s[t] = gr[t];
    rh_s[t] = gh[t];
    rh_s[t + cap] = gh[t + pn];
    rh_s[t + 2 * cap] = gh[t + 2 * pn];
    aj_s[t] = ga[t];
  }
}
#ifndef LMP_POD_MEGA_SPILLS
#define LMP_POD_MEGA_SPILLS 0x46    // stages with extra windows (bit s: stage s)
#endif

// x^n by repeated multiplication from 1, as the power tables (bit-identical values)
template <typename T> __device__ inline T pod_ipow(const T x, const int n)
{
  T p = static_cast<T>(1.0);
  for (int q = 0; q < n; q++) p *= x;
  return p;
}

// pod_monomial() with the powers of (u, v, w) formed here instead of read from a table
template <typename T>
__device__ inline void pod_monomial_uvw(const T u, const T v, const T w, const int ex, T &P, T &Pu,
                                        T &Pv, T &Pw)
{
  const int a = ex & 255, b = (ex >> 8) & 255, c = (ex >> 16) & 255;
  const T pa1 = pod_ipow(u, a > 0 ? a - 1 : 0), pb1 = pod_ipow(v, b > 0 ? b - 1 : 0),
          pc1 = pod_ipow(w, c > 0 ? c - 1 : 0);
  const T pa = (a > 0) ? pa1 * u : pa1, pb = (b > 0) ? pb1 * v : pb1, pc = (c > 0) ? pc1 * w : pc1;
  const T pbc = pb * pc;
  P = pa * pbc;
  Pu = static_cast<T>(a) * pa1 * pbc;
  Pv = static_cast<T>(b) * pa * pb1 * pc;
  Pw = static_cast<T>(c) * pa * pb * pc1;
}

// x^n from a table t[n - 1] = x^n (n >= 1) without the entry x^0 = 1
template <typename T> __device__ inline T pod_pw1(const T *t, const int n)
{
  const T x = t[(n > 0) ? n - 1 : 0];
  return (n > 0) ? x : static_cast<T>(1.0);
}

// pod_monomial() for the powers [u^n][v^n][w^n], n = 1..L
template <typename T>
__device__ inline void pod_monomial1(const T *pd, const int ex, const int L, T &P, T &Pu, T &Pv,
                                     T &Pw)
{
  const int a = ex & 255, b = (ex >> 8) & 255, c = (ex >> 16) & 255;
  const T pa = pod_pw1(pd, a), pb = pod_pw1(pd + L, b), pc = pod_pw1(pd + 2 * L, c);
  const T pbc = pb * pc;
  P = pa * pbc;
  Pu = static_cast<T>(a) * pod_pw1(pd, a > 0 ? a - 1 : 0) * pbc;
  Pv = static_cast<T>(b) * pa * pod_pw1(pd + L, b > 0 ? b - 1 : 0) * pc;
  Pw = static_cast<T>(c) * pa * pb * pod_pw1(pd + 2 * L, c > 0 ? c - 1 : 0);
}

template <typename T, typename TA> struct MegaArgs {
  // neighbors
  NeighArgs na;
  const T *x, *rcutsq;
  int gi1, Ni;
  int cap;        // pair capacity per atom
  int pglobal;    // 1: the pair cache is the global spill area, 0: LDS

  // global spill area: pairs [i*cap + p] (rhat [3*cap*i + d*cap + p]), number of pairs,
  // descriptors / coefficients [i*nDes + m], sumU and forcecoeff [i*nout + t]
  T *pr, *prh;
  int *paj, *pnp;
  T *sd, *su, *sf;

  // radial functions
  RadialSrc<T> rs;     // Hermite table (coef, r0v, invdrv, ne, nb, nr)
  BsplineArgs<T> bs;   // B-spline 2-body basis
  int nrbf2;
  const int *pq_m, *pq_d;
  int nrbf3, K3, L, jtile;

  DescArgs<T> dg;
  FcArgs<T> fc;

  // local EA-POD
  const T *proj, *ledges, *redges, *cent, *cefs, *invlcut2, *invrcut2;
  int nCls, nDes, nCoeff, nActMax, nl2, nl3, nl4;

  // outputs
  T *ei;
  ForceTally<T, TA> ft;
  int *flag;    // flag[0]: largest number of pairs seen above cap

  int stop;     // last stage to run (stage timing); MS_N - 1 normally
  int pad;      // extra LDS bytes per block (occupancy experiments)
  int lenv;     // 1: local EA-POD with all lanes of the wave, 0: as the multi-kernel path
  // freg == 2: per monomial k the terms of forcecoeff34_atom() for one element, built on the
  // host: frec_i[MEGA_FRI k] = active 3-body index or -1, number of 4-body terms, then per
  // term (p4, iA, iB); frec_d[MEGA_FRD k] = pc3[k], then pc4 of each term
  const int *frec_i;
  const T *frec_d;
  int sdt;      // 1: spill the descriptors as bd [i + Ni*m] (input of the multi-kernel path)
  int freg;     // 1: forcecoeff in registers during the force stage, its LDS shared with the tile
  int *pai, *ptype;    // sdt: atom index of the first pair slot [i*cap], element of atom i

  // pairs beyond cap (stages 0-6 in one kernel, pairs in LDS): an atom with more than cap pairs
  // stores all of them in a per-step pool, pool_n pairs (rhat stride pool_n), at an offset
  // claimed from *pool_ctr (zeroed before each launch).  if the pool is full, the windows of
  // cap pairs are found again from the neighbor list instead.  spill == 0: pairs beyond cap
  // are dropped and flag[0] reports them (the run then stops)
  T *gpr, *gprh;
  int *gpaj, *pool_ctr;
  int pool_n, spill;
  int *hist;    // nullptr, or the number of atoms with n pairs (n < 64) and with >= 64
};

// byte offsets of the LDS regions of mega_kernel<S0, S1>
struct MegaLayout {
  int z, p, d, f, w, w6, total;
  bool needP, needD, needU, needF;
};

__host__ __device__ constexpr bool mega_has(int S0, int S1, int s) { return (S0 <= s) && (s <= S1); }

__host__ __device__ inline int mega_round(int b) { return (b + 15) & ~15; }
__host__ __device__ inline int mega_max(int a, int b) { return (a > b) ? a : b; }

template <int S0, int S1, typename T, typename TA>
__host__ __device__ inline MegaLayout mega_layout(const MegaArgs<T, TA> &a)
{
  MegaLayout lo{};
  const int nout = a.nrbf3 * a.K3;
  const int L1 = a.L + 1;
  lo.needP = mega_has(S0, S1, 0) || mega_has(S0, S1, 1) || mega_has(S0, S1, 2) ||
      mega_has(S0, S1, 6);
  lo.needD = mega_has(S0, S1, 1) || mega_has(S0, S1, 3) || mega_has(S0, S1, 4) ||
      mega_has(S0, S1, 5) || mega_has(S0, S1, 6);
  lo.needU = mega_has(S0, S1, 2) || mega_has(S0, S1, 3) || mega_has(S0, S1, 5);
  // freg == 2 with stages 5 and 6 in one kernel: the force coefficients are formed per lane in
  // the force stage from sumU and cb, and are never stored
  const bool flane = (a.freg == 2) && mega_has(S0, S1, 5) && mega_has(S0, S1, 6);
  lo.needF = (mega_has(S0, S1, 5) || mega_has(S0, S1, 6)) && !flane;
  int o = 0;
  lo.z = o;    // zeros (element indices of the pairs) and the atom index of i
  if (lo.needP) o += LMP_POD_MEGA_NOZ ? 16 : mega_round((a.cap + 1) * (int) sizeof(int));
  lo.p = o;
  if (lo.needP && !a.pglobal) o += mega_round(a.cap * (4 * (int) sizeof(T) + (int) sizeof(int)));
  lo.d = o;
  if (lo.needD) o += mega_round(a.nDes * (int) sizeof(T));
  lo.f = o;
  if (lo.needF && !a.freg) o += mega_round(nout * (int) sizeof(T));
  lo.w = o;
  int w = lo.needU ? nout * (int) sizeof(T) : 0;
  if (lo.needF && a.freg) {
    // forcecoeff after sumU and the scratch of forcecoeff34_atom in the stage scratch
    const int fo = mega_round(nout * (int) sizeof(T)) + mega_round((a.K3 + a.fc.Q4) * (int) sizeof(int));
    lo.f = lo.w + fo;
    w = mega_max(w, fo + nout * (int) sizeof(T));
  }
  if (mega_has(S0, S1, 1)) w = mega_max(w, a.nrbf2 * (int) sizeof(TA));
  if (mega_has(S0, S1, 2)) {
    const int jt = a.jtile;
    const int np2 = (LMP_POD_MEGA_S2POW == 1) ? 3 * L1 : ((LMP_POD_MEGA_S2POW == 2) ? 3 * a.L : 0);
    w = mega_max(w, ((np2 + a.nrbf3) * jt + a.nrbf3) * (int) sizeof(T) +
                (jt + a.K3) * (int) sizeof(int));
  }
  if (mega_has(S0, S1, 4) && !a.lenv)
    w = mega_max(w, mega_round(nout * (int) sizeof(T)) + (4 + 4 * LP_KMAX + 3 * LP_KMAX + 2) * (int) sizeof(TA));
  if (mega_has(S0, S1, 5) && !flane)
    w = mega_max(w, mega_round(nout * (int) sizeof(T)) + (a.K3 + a.fc.Q4) * (int) sizeof(int));
  if (flane) w = mega_max(w, nout * (int) sizeof(T));
  int end6 = 0;
  lo.w6 = lo.w;
  if (mega_has(S0, S1, 6)) {
    const int t6 = mega_round(9 * (int) sizeof(TA)) +
        (LMP_POD_MEGA_S6POW ? MFMA_PAIR_TILE * 3 * (LMP_POD_MEGA_S6POW == 2 ? a.L : L1) *
                                  (int) sizeof(T) +
                              MFMA_PAIR_TILE * (int) sizeof(int)
                            : 0) +
        a.K3 * (int) sizeof(int);
    if (LMP_POD_MEGA_S6OVL && lo.needD) {
      lo.w6 = lo.d + mega_round(a.nl2 * (int) sizeof(T));
      end6 = lo.w6 + mega_round(t6);
    } else
      w = mega_max(w, t6);
  }
  o += mega_round(w);
  lo.total = mega_max(o, end6) + a.pad;
  return lo;
}

// stages S0..S1 of atom i.  PERSIST: the energy and virial of the atom are added to eacc and
// vacc (valid in lane 0) instead of being written per atom
template <typename T, typename TA, int S0, int S1, bool PERSIST, bool PG>
__device__ inline void mega_atom(const MegaArgs<T, TA> &a, char *lds_raw, const int i, TA &eacc,
                                 TA (&vacc)[6])
{
  const MegaLayout lo = mega_layout<S0, S1>(a);
  const int lane = threadIdx.x;
  const int lr = lane & 15, lq = lane >> 4;
  const int cap = a.cap;
  const int nelements = 1;
  const int nrbf3 = a.nrbf3, K3 = a.K3, L1 = a.L + 1;
  const int nout = nrbf3 * K3;
  const int nDes = a.nDes;

  int *z_s = reinterpret_cast<int *>(lds_raw + lo.z);    // [cap] zeros (unless NOZ), then gi
  int *gi_s = z_s + (LMP_POD_MEGA_NOZ ? 0 : cap);
  int *const zt = LMP_POD_MEGA_NOZ ? nullptr : z_s;      // element indices of the pairs
  T *d_s = reinterpret_cast<T *>(lds_raw + lo.d);        // descriptors, then cb
  T *f_s = reinterpret_cast<T *>(lds_raw + lo.f);        // forcecoeff [m][k][e]
  char *w_s = lds_raw + lo.w;                            // stage scratch
  T *u_s = reinterpret_cast<T *>(w_s);                   // sumU [m][k][e] at the start of w

  const int gi = a.na.ilist[a.gi1 + i];
  const int itype = a.na.map[a.na.type[gi]];

  // pair cache: LDS, or the global spill area
  T *r_s, *rh_s;
  int *aj_s;
  constexpr bool needP = mega_has(S0, S1, 0) || mega_has(S0, S1, 1) || mega_has(S0, S1, 2) ||
      mega_has(S0, S1, 6);
  // global layout: pair p of atom i is j = i*cap + p, rhat[j + d*NG] (as rr/rhat of the
  // multi-kernel path with a fixed stride of cap pairs per atom)
  const int NG = cap * a.Ni;
  int PN;
  if constexpr (PG || !needP) {
    r_s = a.pr + (size_t) cap * i;
    rh_s = a.prh + (size_t) cap * i;
    aj_s = a.paj + (size_t) cap * i;
    PN = NG;
  } else {
    r_s = reinterpret_cast<T *>(lds_raw + lo.p);
    rh_s = r_s + cap;
    aj_s = reinterpret_cast<int *>(rh_s + 3 * cap);
    PN = cap;
  }
  const PairGeom<T> pg{r_s, rh_s, PN};
  if (lo.needP) {
    if (!LMP_POD_MEGA_NOZ)
      for (int t = lane; t < cap; t += 64) z_s[t] = 0;
    if (lane == 0) gi_s[0] = gi;
  }
  RadialSrc<T> rs = a.rs;
  rs.ti = zt;
  rs.tj = zt;

  int np = 0;

  // the pairs beyond cap (see MegaArgs::spill): poff = offset of the pairs of atom i in the
  // pool or -1, curw = first pair held in the LDS cache.  the stages that read the pairs walk
  // them in windows of cap pairs; with at most cap pairs there is a single window
  constexpr bool SPILL = (S0 == 0) && (S1 == MS_N - 1) && !PG && (LMP_POD_MEGA_SPILLV > 0);
  int poff = -1, curw = 0;

  // the pairs n of the neighbor list inside the cutoff with lo <= n < hi, to (pr, prh,
  // paj)[n - lo] (rhat stride ps), in the order of the list; returns the number of pairs
  auto scan = [&](const int lo, const int hi, T *pr, T *prh, int *paj, const int ps) -> int {
    const T xi0 = a.x[gi * a.na.xs0];
    const T xi1 = a.x[gi * a.na.xs0 + a.na.xs1];
    const T xi2 = a.x[gi * a.na.xs0 + 2 * a.na.xs1];
    const int jnum = a.na.numneigh[gi];
    const unsigned long long below = (1ull << lane) - 1ull;
    int cnt = 0;
    for (int j0 = 0; j0 < jnum; j0 += 64) {
      const int jj = j0 + lane;
      bool in = false;
      int gj = 0, jtype = 0;
      T dx = T(0), dy = T(0), dz = T(0);
      if (jj < jnum)
        in = pod_neigh_test<T>(a.na, a.x, a.rcutsq, gi, itype, xi0, xi1, xi2, jj, gj, jtype, dx,
                               dy, dz);
      const unsigned long long mask = __ballot(in);
      if (in) {
        const int n = cnt + __popcll(mask & below);
        if ((n >= lo) && (n < hi)) {
          // as in neighbor_fused_kernel
          const T r = sqrt(dx * dx + dy * dy + dz * dz);
          const T sinv = static_cast<T>(1.0) / r;
          const int q = n - lo;
          pr[q] = r;
          prh[q] = dx * sinv;
          prh[q + ps] = dy * sinv;
          prh[q + 2 * ps] = dz * sinv;
          paj[q] = gj;
        }
      }
      cnt += __popcll(mask);
    }
    return cnt;
  };

  // the LDS cache holds the window of pairs that starts at wb
  auto ensure = [&](const int wb) {
    if constexpr (SPILL) {
      if (wb == curw) return;
      __syncthreads();
      const int nw = min(cap, np - wb);
      if (poff >= 0) {
        const int pn = a.pool_n;
        const T *gr = a.gpr + poff + wb;
        const T *gh = a.gprh + poff + wb;
        const int *ga = a.gpaj + poff + wb;
        if (LMP_POD_MEGA_COLD)
          mega_reload<T>(r_s, rh_s, aj_s, gr, gh, ga, nw, cap, pn);
        else
          for (int t = lane; t < nw; t += 64) {
            r_s[t] = gr[t];
            rh_s[t] = gh[t];
            rh_s[t + cap] = gh[t + pn];
            rh_s[t + 2 * cap] = gh[t + 2 * pn];
            aj_s[t] = ga[t];
          }
      }
#if LMP_POD_MEGA_SPILLV > 1
      else
        scan(wb, wb + cap, r_s, rh_s, aj_s, cap);
#endif
      __syncthreads();
      curw = __builtin_amdgcn_readfirstlane(wb);
    } else
      (void) wb;
  };

  // ---- inputs of a range that does not start at stage 0
  if constexpr (S0 > 0) {
    np = a.pnp[i];
    if constexpr (needP && !PG) {
      const T *gr = a.pr + (size_t) cap * i;
      const T *gh = a.prh + (size_t) cap * i;
      const int *ga = a.paj + (size_t) cap * i;
      for (int t = lane; t < np; t += 64) {
        r_s[t] = gr[t];
        rh_s[t] = gh[t];
        rh_s[t + cap] = gh[t + NG];
        rh_s[t + 2 * cap] = gh[t + 2 * NG];
        aj_s[t] = ga[t];
      }
    }
    if (lo.needD && (S0 >= 2))
      for (int t = lane; t < nDes; t += 64)
        d_s[t] = a.sdt ? a.sd[i + (size_t) a.Ni * t] : a.sd[(size_t) nDes * i + t];
    if (lo.needU && (S0 >= 3))
      for (int t = lane; t < nout; t += 64) u_s[t] = a.su[(size_t) nout * i + t];
    if (lo.needF && (S0 >= 6))
      for (int t = lane; t < nout; t += 64) f_s[t] = a.sf[(size_t) nout * i + t];
    __syncthreads();
  }

  // ---- stage 0: the pairs of atom i inside the cutoff, in the order of the neighbor list
  if constexpr (mega_has(S0, S1, MS_NEIGH)) {
    const int cnt = scan(0, cap, r_s, rh_s, aj_s, PN);
    if (a.hist && (lane == 0)) atomicAdd(a.hist + min(cnt, 64), 1);
    bool spilled = false;
    if constexpr (SPILL) {
      if (a.spill && (cnt > cap)) {
        int off = -1;
        if (lane == 0) {
          const int o = atomicAdd(a.pool_ctr, cnt);
          off = (o + cnt <= a.pool_n) ? o : -1;
          atomicAdd(a.flag + 5, 1);                 // atoms with pairs beyond cap
          if (off < 0) atomicAdd(a.flag + 6, 1);    // of which found again (pool full)
          if (cnt > __hip_atomic_load(a.flag + 7, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
            atomicMax(a.flag + 7, cnt);
        }
        poff = __builtin_amdgcn_readfirstlane(__shfl(off, 0));    // wave-uniform: SGPR
        if (poff >= 0) scan(0, cnt, a.gpr + poff, a.gprh + poff, a.gpaj + poff, a.pool_n);
        // pool full: found again window by window (2), or dropped and reported (1)
        spilled = (poff >= 0) || (LMP_POD_MEGA_SPILLV > 1);
        if (spilled) np = cnt;
      }
    }
    if (!spilled) {
      if ((cnt > cap) && (lane == 0) &&
          (cnt > __hip_atomic_load(a.flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)))
        atomicMax(a.flag, cnt);
      np = min(cnt, cap);
    }
    if (lane == 0) a.pnp[i] = np;
    __syncthreads();
  }

  // ---- stage 1: 2-body descriptors d2(m) = sum_j B_m(r_j), as twobodydesc_bspline_kernel
  if constexpr (mega_has(S0, S1, MS_TWOBODY)) {
    if (MS_TWOBODY > a.stop) return;
    TA *s_s = reinterpret_cast<TA *>(w_s);
    BsplineArgs<T> bs = a.bs;
    bs.ti = zt;
    bs.tj = zt;
    for (int t = lane; t < a.nrbf2; t += 64) s_s[t] = TA(0);
    __syncthreads();
    // the pairs of the LDS window (nw of them); the first window is the only one unless the
    // atom has more than cap pairs, then the others follow (same below)
    auto win1 = [&](const int nw) {
      for (int n = lane; n < nw; n += 64) {
        T Bq[4], dBq[4];
        TA *se = s_s + pod_bspline_pair(bs, pg, n, Bq, dBq);
#pragma unroll
        for (int q = 0; q < 4; q++) atomicAdd(se + q, static_cast<TA>(Bq[q]));
      }
    };
    win1(min(np, cap));
    if constexpr (SPILL && (LMP_POD_MEGA_SPILLS & 2)) {
      if (np > cap)
        for (int wb = cap; wb < np; wb += cap) {
          ensure(wb);
          win1(min(cap, np - wb));
        }
    }
    __syncthreads();
    for (int t = lane; t < a.nrbf2; t += 64) d_s[t] = static_cast<T>(s_s[t]);
    __syncthreads();
  }

  // cluster tables of the element (lane c: cluster c) and the projection (lanes m, m + 64) in
  // registers for stage 4, loaded at the start of stage LPRE_AT to hide their latency
  constexpr int LPRE_AT = (!mega_has(S0, S1, MS_LOCALENV) || (LMP_POD_MEGA_LPRE == 0))
      ? -1
      : ((LMP_POD_MEGA_LPRE < S0) ? S0 : LMP_POD_MEGA_LPRE);
  T lre = T(0), lle = T(0), lce = T(0), lirc = T(0), lilc = T(0), lpv0 = T(0), lpv1 = T(0);
  auto lpre_load = [&]() {
    const int ncct = itype * a.nCls;
    if (lane < a.nCls) {
      lre = a.redges[ncct + lane];
      lle = a.ledges[ncct + lane];
      lce = a.cent[ncct + lane];
      lirc = a.invrcut2[ncct + lane];
      lilc = a.invlcut2[ncct + lane];
    }
    const T *pr = a.proj + itype * nDes;
    if (lane < nDes) lpv0 = pr[lane];
    if (lane + 64 < nDes) lpv1 = pr[lane + 64];
  };

  // ---- stage 2: radial x angular sums on the matrix cores, as radialangularsum_mfma_kernel
  if constexpr (mega_has(S0, S1, MS_RAS)) {
    if (MS_RAS > a.stop) return;
    if constexpr (LPRE_AT == MS_RAS) lpre_load();
    constexpr int RTM = MEGA_RTM, CTM = 1;
    const int jtile = a.jtile;
    const int NC = nelements * nrbf3;
    const int RT = (K3 + 15) / 16, CT = (NC + 15) / 16;
    // power table [jtile][3][PL]: x^0..x^L (S2POW 1) or x^1..x^L (S2POW 2)
    const int PL = (LMP_POD_MEGA_S2POW == 2) ? L1 - 1 : L1;
    T *pow_s = reinterpret_cast<T *>(w_s);
    const int ldr = jtile + 1;
    T *rbf_s = pow_s + (LMP_POD_MEGA_S2POW ? jtile * 3 * PL : 0);    // [nrbf3][ldr]
    int *tj_s = reinterpret_cast<int *>(rbf_s + nrbf3 * ldr);    // [jtile]
    int *ex_s = tj_s + jtile;                                    // [K3]

    if (lane == 0) {
      ex_s[0] = 0;
      for (int k = 1; k < K3; k++) ex_s[k] = ex_s[a.pq_m[k]] + (1 << (8 * (a.pq_d[k] - 1)));
    }
    __syncthreads();
    // table offsets of the powers (S2POW), or the exponents
    int oa[RTM], ob[RTM], oc[RTM];
#pragma unroll
    for (int rt = 0; rt < RTM; rt++) {
      const int k = rt * 16 + lr;
      const int ex = (k < K3) ? ex_s[k] : 0;
      const int o = LMP_POD_MEGA_S2POW ? PL : 0;
      oa[rt] = ex & 255;
      ob[rt] = o + ((ex >> 8) & 255);
      oc[rt] = 2 * o + ((ex >> 16) & 255);
    }
    int bm[CTM], be[CTM];
#pragma unroll
    for (int ct = 0; ct < CTM; ct++) {
      const int c = ct * 16 + lr;
      bm[ct] = (c < NC) ? c % nrbf3 : -1;
      be[ct] = (c < NC) ? c / nrbf3 : -1;
    }
    typedef typename Mfma<TA>::acc_t acc_t;
    acc_t acc[RTM][CTM];
#pragma unroll
    for (int rt = 0; rt < RTM; rt++)
#pragma unroll
      for (int ct = 0; ct < CTM; ct++) acc[rt][ct] = acc_t{0, 0, 0, 0};

    if constexpr (LMP_POD_MEGA_SPILLS & 4) ensure(0);
    auto win2 = [&](const int nj) {
    for (int j0 = 0; j0 < nj; j0 += jtile) {
      const int nt = min(jtile, nj - j0);
      const int nt4 = (nt + 3) & ~3;
      __syncthreads();
      if (LMP_POD_MEGA_S2FILL && LMP_POD_MEGA_S2POW && (nt4 * (nrbf3 + 3) <= 64)) {
        // lanes [0, nt4*nrbf3): radial function m of pair jj; then direction d of pair jj
        const int nr = nt4 * nrbf3;
        if (lane < nr) {
          const int jj = lane / nrbf3, m = lane - jj * nrbf3;
          T f = T(0), df;
          if (jj < nt) {
            const PairRadial<T> prad(rs, pg, j0 + jj);
            prad(m, f, df);
          }
          rbf_s[m * ldr + jj] = f;
          if (m == 0) tj_s[jj] = (jj < nt) ? 0 : -1;
        } else if (lane < nr + 3 * nt4) {
          const int q = lane - nr, jj = q / 3, d = q - 3 * jj;
          T *pw = pow_s + jj * 3 * PL + d * PL;
          if (jj < nt) {
            const T x = pg.rhat[j0 + jj + d * pg.N];
            T p = static_cast<T>(1.0);
            if (LMP_POD_MEGA_S2POW == 2) p *= x;
            for (int n = 0; n < PL; n++) {
              pw[n] = p;
              p *= x;
            }
          } else
            for (int n = 0; n < PL; n++) pw[n] = T(0);
        }
      } else
      for (int jj = lane; jj < nt4; jj += 64) {
        T *pw = pow_s + jj * 3 * PL;
        if (jj < nt) {
          const int j = j0 + jj;
          if (LMP_POD_MEGA_S2POW) {
            T u, v, w;
            pg.uvw(j, u, v, w);
            T pu = static_cast<T>(1.0), pv = pu, pz = pu;
            if (LMP_POD_MEGA_S2POW == 2) {
              pu *= u;
              pv *= v;
              pz *= w;
            }
            for (int n = 0; n < PL; n++) {
              pw[n] = pu;
              pw[PL + n] = pv;
              pw[2 * PL + n] = pz;
              pu *= u;
              pv *= v;
              pz *= w;
            }
          }
          pair_radial<0>(rs, pg, j, nrbf3, [&](int m, T f, T) { rbf_s[m * ldr + jj] = f; });
          tj_s[jj] = 0;
        } else {
          if (LMP_POD_MEGA_S2POW)
            for (int n = 0; n < 3 * PL; n++) pw[n] = T(0);
          for (int m = 0; m < nrbf3; m++) rbf_s[m * ldr + jj] = T(0);
          tj_s[jj] = -1;
        }
      }
      __syncthreads();

      for (int q = 0; q < nt4; q += 4) {
        const int jj = q + lq;
        const int tjj = tj_s[jj];
        TA av[RTM], b[CTM];
#if LMP_POD_MEGA_S2POW == 1
        const T *pw = pow_s + jj * 3 * L1;
#pragma unroll
        for (int rt = 0; rt < RTM; rt++)
          av[rt] = (rt * 16 + lr < K3) ? static_cast<TA>(pw[oa[rt]] * pw[ob[rt]] * pw[oc[rt]])
                                       : TA(0);
#elif LMP_POD_MEGA_S2POW == 2
        // padding pairs give nonzero powers here, but their radial functions (B) are zero
        const T *pw = pow_s + jj * 3 * PL;
#pragma unroll
        for (int rt = 0; rt < RTM; rt++)
          av[rt] = (rt * 16 + lr < K3)
              ? static_cast<TA>(pod_pw1(pw, oa[rt]) * pod_pw1(pw + PL, ob[rt] - PL) *
                                pod_pw1(pw + 2 * PL, oc[rt] - 2 * PL))
              : TA(0);
#else
        // u^a v^b w^c of pair jj from the pair cache (padding pairs: 0, as the zero table rows)
        T u = T(0), v = T(0), w = T(0);
        if (jj < nt) pg.uvw(j0 + jj, u, v, w);
#pragma unroll
        for (int rt = 0; rt < RTM; rt++)
          av[rt] = ((rt * 16 + lr < K3) && (jj < nt))
              ? static_cast<TA>(pod_ipow(u, oa[rt]) * pod_ipow(v, ob[rt]) * pod_ipow(w, oc[rt]))
              : TA(0);
#endif
#pragma unroll
        for (int ct = 0; ct < CTM; ct++)
          b[ct] = ((bm[ct] >= 0) && (be[ct] == tjj)) ? static_cast<TA>(rbf_s[bm[ct] * ldr + jj])
                                                     : TA(0);
#pragma unroll
        for (int rt = 0; rt < RTM; rt++)
          if (rt < RT)
#pragma unroll
            for (int ct = 0; ct < CTM; ct++)
              if (ct < CT) acc[rt][ct] = Mfma<TA>::mma(av[rt], b[ct], acc[rt][ct]);
      }
    }
    };
    win2(min(np, cap));
    if constexpr (SPILL && (LMP_POD_MEGA_SPILLS & 4)) {
      if (np > cap)
        for (int wb = cap; wb < np; wb += cap) {
          ensure(wb);
          win2(min(cap, np - wb));
        }
    }
    __syncthreads();    // u_s overlaps the tile
#pragma unroll
    for (int rt = 0; rt < RTM; rt++)
#pragma unroll
      for (int ct = 0; ct < CTM; ct++)
#pragma unroll
        for (int v = 0; v < 4; v++) {
          const int k = rt * 16 + Mfma<TA>::row(lane, v);
          if ((rt < RT) && (ct < CT) && (k < K3) && (bm[ct] >= 0))
            u_s[be[ct] + nelements * (k + K3 * bm[ct])] = static_cast<T>(acc[rt][ct][v]);
        }
    __syncthreads();
  }

  // ---- stage 3: 3- and 4-body descriptors from sumU, as desc34_kernel
  if constexpr (mega_has(S0, S1, MS_DESC)) {
    if (MS_DESC > a.stop) return;
    if constexpr (LPRE_AT == MS_DESC) lpre_load();
    DescArgs<T> g = a.dg;
    g.d3 = d_s + a.nl2;
    g.d4 = d_s + a.nl2 + a.nl3;
    g.Ni = 1;
    desc34_atom<T, TA>(g, u_s, 0);
    __syncthreads();
  }

  // ---- stage 4: local EA-POD (localenv_pca_split, localenv_atom_split, localenv_cb).
  // a.lenv == 1: the wave works on its one atom with all lanes (wave sums for the
  // projections, ballots instead of the binary searches); a.lenv == 0: the same
  // decomposition as the multi-kernel path (4 partial sums, binary searches, lane 0)
  if constexpr (mega_has(S0, S1, MS_LOCALENV)) {
    if (MS_LOCALENV > a.stop) return;
    if constexpr (LPRE_AT == MS_LOCALENV) lpre_load();
    if (a.lenv) {
      // as localenv_wave_atom<KM = 4, PRE> of the old path: one wave, wave sums, ballots
      constexpr int KM = MEGA_KM;
      constexpr bool PRE = (LPRE_AT >= 0);
      const T *B = d_s;
      const int nCls = a.nCls;
      const int ncct = itype * nCls;
      const T *pr = a.proj + itype * nDes;
      TA p = TA(0);
      if (PRE) {
        if (lane < nDes) p += static_cast<TA>(lpv0 * B[lane]);
        if (lane + 64 < nDes) p += static_cast<TA>(lpv1 * B[lane + 64]);
      } else
        for (int m = lane; m < nDes; m += 64) p += static_cast<TA>(pr[m] * B[m]);
      const TA pcai = wave_sum(p);
      // number of right edges below pcai (lower bound), then of the left edges of the next
      // clusters at or below it, as the binary searches of localenv_pca_kernel
      int kl = 0, kni = 1;
      if (PRE) {
        kl = __popcll(__ballot((lane < nCls) && (static_cast<TA>(lre) < pcai)));
        const int hi = min(kl + a.nActMax, nCls);
        kni += __popcll(__ballot((lane > kl) && (lane < hi) && (static_cast<TA>(lle) <= pcai)));
      } else {
        for (int c0 = 0; c0 < nCls; c0 += 64) {
          const int cc = c0 + lane;
          kl += __popcll(__ballot((cc < nCls) && (static_cast<TA>(a.redges[ncct + cc]) < pcai)));
        }
        const int hi = min(kl + a.nActMax, nCls);
        const int cc = kl + 1 + lane;
        kni += __popcll(__ballot((cc < hi) && (static_cast<TA>(a.ledges[ncct + cc]) <= pcai)));
      }
      const int nc = a.nCoeff * itype;
      const int knc = 1 + nc + kl * nDes;
      // coefficients of the active clusters, loaded once for the sums and the force coefficients
      T cf[KM][2];
#pragma unroll
      for (int k = 0; k < KM; k++)
#pragma unroll
        for (int q = 0; q < 2; q++) {
          const int m = lane + 64 * q;
          cf[k][q] = ((k < kni) && (m < nDes)) ? a.cefs[knc + k * nDes + m] : T(0);
        }
      TA sumE[KM];
#pragma unroll
      for (int k = 0; k < KM; k++) sumE[k] = TA(0);
#pragma unroll
      for (int q = 0; q < 2; q++) {
        const int m = lane + 64 * q;
        const T b = (m < nDes) ? B[m] : T(0);
#pragma unroll
        for (int k = 0; k < KM; k++)
          if (k < kni) sumE[k] += static_cast<TA>(cf[k][q] * b);
      }
#pragma unroll
      for (int k = 0; k < KM; k++)
        if (k < kni) sumE[k] = wave_sum(sumE[k]);
      const T pcaiT = static_cast<T>(pcai);
      const int kncct = kl + ncct;
      TA sumfDi = 0.0, S = 0.0;
      T Dk[KM], dDk[KM], Pk[KM];
#pragma unroll
      for (int k = 0; k < KM; k++) {
        Dk[k] = dDk[k] = Pk[k] = T(0);
        if (k < kni) {
          const int kg = k + kncct;
          T ck, irk, ilk;
          if (PRE && (kl + k < nCls)) {
            ck = __shfl(lce, kl + k);
            irk = __shfl(lirc, kl + k);
            ilk = __shfl(lilc, kl + k);
          } else {
            ck = a.cent[kg];
            irk = a.invrcut2[kg];
            ilk = a.invlcut2[kg];
          }
          const T pc = pcaiT - ck;
          const T inv_rcut2 = (pc >= static_cast<T>(0.0)) ? irk : ilk;
          const T y2 = (pc * pc) * inv_rcut2;
          const T omy = static_cast<T>(1.0) - y2 * y2;
          const T omy2 = omy * omy;
          const T fcut = omy2 * omy2;
          const T dfcut = static_cast<T>(-8.0) * inv_rcut2 * y2 * omy2 * omy;
          const T invDk = static_cast<T>(1.0) / (pc * pc + static_cast<T>(1e-20));
          const T fDk = fcut * invDk;
          Dk[k] = fDk;
          dDk[k] = static_cast<T>(2.0) * pc * invDk * (dfcut - fDk);
          sumfDi += static_cast<TA>(fDk);
          S += static_cast<TA>(dDk[k]);
        }
      }
      sumfDi = static_cast<TA>(1.0) / sumfDi;
      TA eival = static_cast<TA>(a.cefs[nc]);
      TA Tt = 0.0, A = 0.0;
#pragma unroll
      for (int k = 0; k < KM; k++)
        if (k < kni) {
          Pk[k] = static_cast<T>(static_cast<TA>(Dk[k]) * sumfDi);
          const T cpk = static_cast<T>(sumE[k] * sumfDi);
          eival += sumE[k] * static_cast<TA>(Pk[k]);
          Tt += static_cast<TA>(cpk * Pk[k]);
          A += static_cast<TA>(cpk * dDk[k]);
        }
      if (lane == 0) {
        if constexpr (PERSIST) eacc += static_cast<TA>(static_cast<T>(eival));
        else a.ei[i] = static_cast<T>(eival);
      }
      const T Ui = static_cast<T>(A - S * Tt);
      T cbm[2] = {T(0), T(0)};
#pragma unroll
      for (int q = 0; q < 2; q++) {
        const int m = lane + 64 * q;
        if (m < nDes) {
          const T pm = PRE ? (q ? lpv1 : lpv0) : pr[m];
          TA sum = static_cast<TA>(Ui * pm);
#pragma unroll
          for (int k = 0; k < KM; k++)
            if (k < kni) sum += static_cast<TA>(cf[k][q] * Pk[k]);
          cbm[q] = static_cast<T>(sum);
        }
      }
      __syncthreads();
#pragma unroll
      for (int q = 0; q < 2; q++) {
        const int m = lane + 64 * q;
        if (m < nDes) d_s[m] = cbm[q];
      }
      __syncthreads();
    } else {
    TA *part_s = reinterpret_cast<TA *>(w_s + mega_round(nout * (int) sizeof(T)));    // [LP]
    TA *pe_s = part_s + LP;                                                          // [LP_KMAX][LP]
    T *pk_s = reinterpret_cast<T *>(pe_s + LP * LP_KMAX);                            // [LP_KMAX]
    T *dk_s = pk_s + LP_KMAX;                                                        // [LP_KMAX]
    T *ddk_s = dk_s + LP_KMAX;                                                       // [LP_KMAX]
    TA *uv_s = reinterpret_cast<TA *>(ddk_s + LP_KMAX);                              // U
    const int nCls = a.nCls, nActMax = a.nActMax;
    const T *B = d_s;
    if (lane < LP) {
      const T *pr = a.proj + itype * nDes;
      TA part = 0.0;
      for (int m = lane; m < nDes; m += LP) part += static_cast<TA>(pr[m] * B[m]);
      part_s[lane] = part;
    }
    __syncthreads();
    TA pcai = part_s[0];
    for (int q = 1; q < LP; q++) pcai += part_s[q];
    const int ncct = itype * nCls;
    int lo2 = 0, hi = nCls;
    while (lo2 < hi) {
      const int mid = (lo2 + hi) >> 1;
      if (static_cast<TA>(a.redges[ncct + mid]) < pcai) lo2 = mid + 1;
      else hi = mid;
    }
    const int kl = lo2;
    hi = kl + nActMax;
    if (hi > nCls) hi = nCls;
    lo2++;
    while (lo2 < hi) {
      const int mid = (lo2 + hi) >> 1;
      if (static_cast<TA>(a.ledges[ncct + mid]) <= pcai) lo2 = mid + 1;
      else hi = mid;
    }
    const int kni = lo2 - kl;
    const int nc = a.nCoeff * itype;
    const int knc = 1 + nc + kl * nDes;
    // partial sumE(k) over the descriptors m = lp (mod LP), lane = lp + LP*k
    {
      const int k = lane / LP, lp = lane % LP;
      if (k < kni) {
        const T *c = a.cefs + knc + k * nDes;
        TA part = 0.0;
        for (int m = lp; m < nDes; m += LP) part += static_cast<TA>(c[m] * B[m]);
        pe_s[k * LP + lp] = part;
      }
    }
    __syncthreads();
    if (lane == 0) {
      const T pcaiT = static_cast<T>(pcai);
      const int kncct = kl + nCls * itype;
      TA sumfDi = 0.0, S = 0.0;
      for (int k = 0; k < kni; ++k) {
        const int kg = k + kncct;
        const T pc = pcaiT - a.cent[kg];
        const T inv_rcut2 = (pc >= static_cast<T>(0.0)) ? a.invrcut2[kg] : a.invlcut2[kg];
        const T y2 = (pc * pc) * inv_rcut2;
        const T omy = static_cast<T>(1.0) - y2 * y2;
        const T omy2 = omy * omy;
        const T fcut = omy2 * omy2;
        const T dfcut = static_cast<T>(-8.0) * inv_rcut2 * y2 * omy2 * omy;
        const T invDk = static_cast<T>(1.0) / (pc * pc + static_cast<T>(1e-20));
        const T fDk = fcut * invDk;
        const T dfDkdpc = static_cast<T>(2.0) * pc * invDk * (dfcut - fDk);
        dk_s[k] = fDk;
        ddk_s[k] = dfDkdpc;
        sumfDi += static_cast<TA>(fDk);
        S += static_cast<TA>(dfDkdpc);
      }
      sumfDi = static_cast<TA>(1.0) / sumfDi;
      TA eival = static_cast<TA>(a.cefs[nc]);
      TA Tt = 0.0, A = 0.0;
      for (int k = 0; k < kni; ++k) {
        TA sumE = pe_s[k * LP];
        for (int q = 1; q < LP; q++) sumE += pe_s[k * LP + q];
        const T Pk = static_cast<T>(static_cast<TA>(dk_s[k]) * sumfDi);
        const T cpk = static_cast<T>(sumE * sumfDi);
        pk_s[k] = Pk;
        eival += sumE * static_cast<TA>(Pk);
        Tt += static_cast<TA>(cpk * Pk);
        A += static_cast<TA>(cpk * ddk_s[k]);
      }
      if constexpr (PERSIST) eacc += static_cast<TA>(static_cast<T>(eival));
      else a.ei[i] = static_cast<T>(eival);
      // U is stored as T in the multi-kernel path
      uv_s[0] = static_cast<TA>(static_cast<T>(A - S * Tt));
    }
    __syncthreads();
    // cb(m) = U proj(m) + sum_k cefs(m, k) P_k, overwriting the descriptors
    const T Ui = static_cast<T>(uv_s[0]);
    T cbm[2] = {T(0), T(0)};
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int m = lane + 64 * q;
      if (m < nDes) {
        const int kncm = 1 + m + kl * nDes + a.nCoeff * itype;
        TA sum = static_cast<TA>(Ui * a.proj[m + nDes * itype]);
        for (int k = 0; k < kni; ++k) sum += static_cast<TA>(a.cefs[kncm + k * nDes] * pk_s[k]);
        cbm[q] = static_cast<T>(sum);
      }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int m = lane + 64 * q;
      if (m < nDes) d_s[m] = cbm[q];
    }
    __syncthreads();
    }
  }

  // ---- stage 5: 3- and 4-body force coefficients (forcecoeff34_atom)
  if constexpr (mega_has(S0, S1, MS_FC)) {
    if (MS_FC > a.stop) return;
    const bool flane = (a.freg == 2) && mega_has(S0, S1, MS_FORCE);
    FcArgs<T> g = a.fc;
    g.cb3 = d_s + a.nl2;
    g.cb4 = d_s + a.nl2 + a.nl3;
    g.sumU = u_s;
    g.Ni = 1;
    int *qa_s = reinterpret_cast<int *>(w_s + mega_round(nout * (int) sizeof(T)));
    if (!flane) forcecoeff34_atom<T>(g, f_s, u_s, qa_s, qa_s + K3, 0);
  }

  // ---- stage 6: forces and tally, as allbody_forces_mfma_kernel<CSM = 2> with the 2-body
  // forces (HIP_FTALLY)
  if constexpr (mega_has(S0, S1, MS_FORCE)) {
    if (MS_FORCE > a.stop) return;
    constexpr int CSM = MEGA_CSM, JT = MFMA_PAIR_TILE;
    const int PL = (LMP_POD_MEGA_S6POW == 2) ? L1 - 1 : L1;    // powers per direction
    const int PD = 3 * PL;
    const int NeK3 = nelements * K3;
    const int NC = nelements * nrbf3;
    const int CS = (NC + 3) / 4;
    const int KT = (K3 + 15) / 16;
    char *w6_s = lds_raw + lo.w6;    // force stage scratch (S6OVL: over cb3/cb4 and U)
    TA *acc_s = reinterpret_cast<TA *>(w6_s);
    T *pd_s = reinterpret_cast<T *>(w6_s + mega_round(9 * (int) sizeof(TA)));    // [JT][PD] (S6POW)
    int *tj_s = reinterpret_cast<int *>(pd_s + (LMP_POD_MEGA_S6POW ? JT * PD : 0));    // [JT]
    int *ex_s = tj_s + (LMP_POD_MEGA_S6POW ? JT : 0);                              // [K3]
    const T *fc_s = f_s;
    int nj = 0;

    ForceTally<T, TA> ft = a.ft;
    ft.ai = gi_s;    // ai[0] = gi
    ft.aj = aj_s;
    ft.cb2 = d_s;
    ft.Ni = 1;
    ft.bs.ti = zt;
    ft.bs.tj = zt;

    int cm[CSM], ce[CSM];
#pragma unroll
    for (int cs = 0; cs < CSM; cs++) {
      const int c = 4 * cs + lq;
      cm[cs] = (c < NC) ? c % nrbf3 : 0;
      ce[cs] = (c < NC) ? c / nrbf3 : -1;
    }
    // the A operands (force coefficients) of this lane, loaded once; afterwards the
    // forcecoeff LDS may be reused by the pair tile (a.freg)
    TA fa[MEGA_RTM][CSM];
    const bool flane = (a.freg == 2) && mega_has(S0, S1, MS_FC);
    const bool fdbg = (a.freg == 3) && mega_has(S0, S1, MS_FC);    // both, report differences
#pragma unroll
    for (int kt = 0; kt < MEGA_RTM; kt++)
#pragma unroll
      for (int cs = 0; cs < CSM; cs++) {
        const int k = 16 * kt + lr;
        const bool valid = (kt < KT) && (cs < CS) && (k < K3) && (ce[cs] >= 0);
        if (!flane && !fdbg) {
          fa[kt][cs] = valid ? static_cast<TA>(fc_s[ce[cs] + nelements * k + NeK3 * cm[cs]]) : TA(0);
          continue;
        }
        // forcecoeff34_atom() for one element, entry (m, k) only: the 3-body term, then the
        // 4-body terms that update it (the order of the LDS atomics there is arbitrary), from
        // the host-built recipe of monomial k
        T f = T(0);
        if (valid) {
          const int m = cm[cs];
          const T *cb3 = d_s + a.nl2, *cb4 = d_s + a.nl2 + a.nl3;
          const T *um = u_s + K3 * m;
          const int *ri = a.frec_i + MEGA_FRI * k;
          const T *rd = a.frec_d + MEGA_FRD * k;
          const int a3 = ri[0], cnt = ri[1];
          if (a3 >= 0) {
            const T wq = rd[0] * cb3[a3 + a.fc.nabf3_active * m];
            const T t = wq * um[k];
            f = t + t;
          }
          if (m < a.fc.nrbf4) {    // the 4-body terms use the first nrbf4 radial functions
            const int nb4 = a.fc.nabf4;
#pragma unroll
            for (int t = 0; t < MEGA_FT; t++)
              if (t < cnt) {
                const int p4 = ri[2 + 3 * t], iA = ri[3 + 3 * t], iB = ri[4 + 3 * t];
                f += (rd[1 + t] * cb4[p4 + nb4 * m]) * (um[iA] * um[iB]);
              }
          }
        }
        fa[kt][cs] = static_cast<TA>(f);
        if (fdbg) {
          const T ref = valid ? fc_s[ce[cs] + nelements * k + NeK3 * cm[cs]] : T(0);
          if ((blockIdx.x < 4) && !(fabs(f - ref) <= 1e-12 * fabs(ref) + 1e-300))
            printf("mega freg dbg atom %d lane %d kt %d cs %d m %d k %d lane-f %.17g lds-f %.17g\n",
                   (int) blockIdx.x, lane, kt, cs, cm[cs], k, (double) f, (double) ref);
          fa[kt][cs] = static_cast<TA>(ref);
        }
      }
    __syncthreads();
    if (lane < 9) acc_s[lane] = TA(0);
    pod_exponents<T>(ex_s, a.pq_m, a.pq_d, K3);
    typedef typename Mfma<TA>::acc_t acc_t;
#if LMP_POD_MEGA_S6POW
    T xh = T(0);
    int tjh = -1;
    // pair jbase + jj is at jbase + jj - curw in the LDS window
    auto fetch = [&](const int jbase) {
      if (lane < 3 * JT) {
        const int jj = lane % JT, d = lane / JT;
        if (jbase + jj < nj) {
          xh = pg.rhat[jbase - curw + jj + d * pg.N];
          tjh = 0;
        }
      }
    };
    nj = np;
    if constexpr (SPILL && (LMP_POD_MEGA_SPILLS & 64)) ensure(0);
    fetch(0);
    // with more than cap pairs, a tile that does not fit in the LDS window starts a new window
    // (and is fetched again); otherwise the window holds all pairs
    for (int j0 = 0; j0 < nj; j0 += JT) {
      const int nt = min(JT, nj - j0);
      if constexpr (SPILL && (LMP_POD_MEGA_SPILLS & 64)) {
        if (j0 - curw + nt > cap) {
          ensure(j0);
          fetch(j0);
        }
      }
      const int jl0 = __builtin_amdgcn_readfirstlane(j0 - curw);
      __syncthreads();
      if (lane < 3 * JT) {
        const int jj = lane % JT, d = lane / JT;
        T *pd = pd_s + jj * PD + d * PL;
        const bool in = (jj < nt);
        T p = in ? static_cast<T>(1.0) : T(0);
        if (LMP_POD_MEGA_S6POW == 2) p *= xh;
        for (int n = 0; n < PL; n++) {
          pd[n] = p;
          p *= xh;
        }
        if (d == 0) tj_s[jj] = in ? tjh : -1;
      }
      if ((j0 + JT < nj) && (j0 + JT - curw < cap)) fetch(j0 + JT);
      __syncthreads();

      for (int s0 = 0; s0 < nt; s0 += 16) {
        TA SA = TA(0), SU = TA(0), SV = TA(0), SW = TA(0);
        const int jb = s0 + lr;
        const int tjb = tj_s[jb];
        const PairRadial<T> prad(rs, pg, jl0 + min(jb, nt - 1));
        const T *pd = pd_s + jb * PD;
#if LMP_POD_MEGA_S6POW == 2
#define LMP_POD_MEGA_MONO(EX, P, Pu, Pv, Pw) pod_monomial1<T>(pd, EX, PL, P, Pu, Pv, Pw)
#else
#define LMP_POD_MEGA_MONO(EX, P, Pu, Pv, Pw) pod_monomial<T>(pd, EX, L1, P, Pu, Pv, Pw)
#endif
#else
    // no power table: each lane forms the monomials of its pair column from the pair cache
    nj = np;
    if constexpr (SPILL && (LMP_POD_MEGA_SPILLS & 64)) ensure(0);
    __syncthreads();
    for (int j0 = 0; j0 < nj; j0 += JT) {
      const int nt = min(JT, nj - j0);
      if constexpr (SPILL && (LMP_POD_MEGA_SPILLS & 64)) {
        if (j0 - curw + nt > cap) ensure(j0);
      }
      const int jl0 = __builtin_amdgcn_readfirstlane(j0 - curw);

      for (int s0 = 0; s0 < nt; s0 += 16) {
        TA SA = TA(0), SU = TA(0), SV = TA(0), SW = TA(0);
        const int jb = s0 + lr;
        const int tjb = (jb < nt) ? 0 : -1;
        const int jc = jl0 + min(jb, nt - 1);
        const PairRadial<T> prad(rs, pg, jc);
        T ux, vx, wx;
        pg.uvw(jc, ux, vx, wx);
#define LMP_POD_MEGA_MONO(EX, P, Pu, Pv, Pw) pod_monomial_uvw<T>(ux, vx, wx, EX, P, Pu, Pv, Pw)
#endif
        TA br[CSM], bdr[CSM];
#pragma unroll
        for (int cs = 0; cs < CSM; cs++) {
          br[cs] = bdr[cs] = TA(0);
          if ((cs < CS) && (jb < nt) && (ce[cs] == tjb)) {
            T f, df;
            prad(cm[cs], f, df);
            br[cs] = static_cast<TA>(f);
            bdr[cs] = static_cast<TA>(df);
          }
        }
#pragma unroll
        for (int kt = 0; kt < MEGA_RTM; kt++) {
          if (kt >= KT) break;
          acc_t accG = acc_t{0, 0, 0, 0}, accH = acc_t{0, 0, 0, 0};
#pragma unroll
          for (int cs = 0; cs < CSM; cs++)
            if (cs < CS) {
              const TA av = fa[kt][cs];
              accG = Mfma<TA>::mma(av, br[cs], accG);
              accH = Mfma<TA>::mma(av, bdr[cs], accH);
            }
#pragma unroll
          for (int v = 0; v < 4; v++) {
            const int kv = 16 * kt + Mfma<TA>::row(lane, v);
            if (kv < K3) {
              T P, Pu, Pv, Pw;
              LMP_POD_MEGA_MONO(ex_s[kv], P, Pu, Pv, Pw);
              SA += accH[v] * static_cast<TA>(P);
              SU += accG[v] * static_cast<TA>(Pu);
              SV += accG[v] * static_cast<TA>(Pv);
              SW += accG[v] * static_cast<TA>(Pw);
            }
          }
        }
#undef LMP_POD_MEGA_MONO
#pragma unroll
        for (int off = 16; off < 64; off <<= 1) {
          SA += __shfl_xor(SA, off);
          SU += __shfl_xor(SU, off);
          SV += __shfl_xor(SV, off);
          SW += __shfl_xor(SW, off);
        }
        if ((lq == 0) && (jb < nt)) {
          T g[9];
          pod_pair_gradients<T>(g, pg, jl0 + jb);
          pod_pair_finish<T, TA>(ft, pg, nullptr, 0, jl0 + jb, tjb, g, SA, SU, SV, SW, acc_s);
        }
      }
    }
    __syncthreads();
    if constexpr (PERSIST) {
      ForceTally<T, TA> fa = ft;
      fa.vir = nullptr;
      pod_atom_finish<T, TA>(fa, i, 0, np, acc_s);
      if ((lane == 0) && ft.vir)
        for (int d = 0; d < 6; d++) vacc[d] += acc_s[3 + d];
    } else
      pod_atom_finish<T, TA>(ft, i, 0, np, acc_s);
  }

  // ---- live outputs of a range that does not end at stage 6
  if constexpr (S1 < MS_N - 1) {
    if (S1 > a.stop) return;
    if constexpr (mega_has(S0, S1, 0) && !PG) {
      T *gr = a.pr + (size_t) cap * i;
      T *gh = a.prh + (size_t) cap * i;
      int *ga = a.paj + (size_t) cap * i;
      for (int t = lane; t < np; t += 64) {
        gr[t] = r_s[t];
        gh[t] = rh_s[t];
        gh[t + NG] = rh_s[t + cap];
        gh[t + 2 * NG] = rh_s[t + 2 * cap];
        ga[t] = aj_s[t];
      }
    }
    if ((mega_has(S0, S1, 1) || mega_has(S0, S1, 3) || mega_has(S0, S1, 4)) && (S1 <= 5))
      for (int t = lane; t < nDes; t += 64) {
        if (a.sdt) a.sd[i + (size_t) a.Ni * t] = d_s[t];
        else a.sd[(size_t) nDes * i + t] = d_s[t];
      }
    if (mega_has(S0, S1, 0) && a.sdt && (lane == 0)) {
      a.pai[(size_t) cap * i] = gi;
      a.ptype[i] = itype;
    }
    if (mega_has(S0, S1, 2) && (S1 <= 4))
      for (int t = lane; t < nout; t += 64) a.su[(size_t) nout * i + t] = u_s[t];
    if (mega_has(S0, S1, 5) && (S1 == 5))
      for (int t = lane; t < nout; t += 64) a.sf[(size_t) nout * i + t] = f_s[t];
  }
}

// one block (wavefront) per atom
// register budget of the mega kernel (amdgpu_num_vgpr(N) caps VGPR+AGPR at 2N on gfx942)
#ifndef LMP_POD_MEGA_VGPR
#define LMP_POD_MEGA_VGPR LMP_POD_ALLBODY_VGPR
#endif
#if LMP_POD_MEGA_VGPR > 0
#define LMP_POD_MEGA_ATTR __attribute__((amdgpu_num_vgpr(LMP_POD_MEGA_VGPR)))
#else
#define LMP_POD_MEGA_ATTR
#endif

template <typename T, typename TA, int S0, int S1, bool PG>
__global__ void __launch_bounds__(64) LMP_POD_MEGA_ATTR mega_kernel(const MegaArgs<T, TA> a)
{
  extern __shared__ char lds_raw[];
  TA e = TA(0), v[6];
  mega_atom<T, TA, S0, S1, false, PG>(a, lds_raw, blockIdx.x, e, v);
}

// persistent version: each block loops over the atoms blockIdx.x + k*gridDim.x and adds the
// energy and virial of its atoms to acc once.  SB < 6: the stages 0..SB of all atoms, a grid
// barrier (all blocks must be resident), then the stages SB+1..6 (study of a grid-wide
// dependency; the spill area carries the live intermediates across the barrier)
template <typename T, typename TA, int SB, bool PG>
__global__ void __launch_bounds__(64) LMP_POD_ALLBODY_ATTR
    mega_persist_kernel(const MegaArgs<T, TA> a, TA *acc, unsigned *bar, const int eg, const int vg)
{
  extern __shared__ char lds_raw[];
  // energy and virial sums of the block in LDS (registers would stay allocated through all
  // stages), atoms taken from a global ticket (bar[1], bar[2] for the second phase)
  __shared__ TA acc_b[7];
  __shared__ int i_s;
  if (threadIdx.x < 7) acc_b[threadIdx.x] = TA(0);
  TA &e = acc_b[0];
  TA(&v)[6] = *reinterpret_cast<TA(*)[6]>(acc_b + 1);
  auto next = [&](unsigned *ctr) {
    __syncthreads();
    if (threadIdx.x == 0) i_s = (int) atomicAdd(ctr, 1u);
    __syncthreads();
    return i_s;
  };
  if constexpr (SB >= MS_N - 1) {
    for (int i = next(bar + 1); i < a.Ni; i = next(bar + 1))
      mega_atom<T, TA, 0, MS_N - 1, true, PG>(a, lds_raw, i, e, v);
  } else {
    for (int i = next(bar + 1); i < a.Ni; i = next(bar + 1))
      mega_atom<T, TA, 0, SB, true, PG>(a, lds_raw, i, e, v);
    // grid barrier: the spill writes are released before the arrival is counted
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
      atomicAdd(bar, 1u);
      while (__hip_atomic_load(bar, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < gridDim.x)
        __builtin_amdgcn_s_sleep(2);
    }
    __syncthreads();
    __threadfence();
    for (int i = next(bar + 2); i < a.Ni; i = next(bar + 2))
      mega_atom<T, TA, SB + 1, MS_N - 1, true, PG>(a, lds_raw, i, e, v);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (eg) atomicAdd(acc, e);
    if (vg)
      for (int d = 0; d < 6; d++) atomicAdd(acc + 1 + d, v[d]);
  }
}

// query != nullptr: no launch, *query = LDS bytes per block, query[1] = resident blocks per CU
template <typename T, typename TA, int S0, int S1>
bool mega_launch(hipStream_t stream, const MegaArgs<T, TA> &a, int *query = nullptr)
{
  const MegaLayout lo = mega_layout<S0, S1>(a);
  if (lo.total > 65536) return false;
  if (query) {
    query[0] = lo.total;
    query[1] = 0;
    const void *k = a.pglobal ? reinterpret_cast<const void *>(mega_kernel<T, TA, S0, S1, true>)
                              : reinterpret_cast<const void *>(mega_kernel<T, TA, S0, S1, false>);
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(&query[1], k, 64, (size_t) lo.total) ==
        hipSuccess;
  }
  if (a.pglobal)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(mega_kernel<T, TA, S0, S1, true>), dim3(a.Ni), dim3(64),
                       (size_t) lo.total, stream, a);
  else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(mega_kernel<T, TA, S0, S1, false>), dim3(a.Ni), dim3(64),
                       (size_t) lo.total, stream, a);
  return hipGetLastError() == hipSuccess;
}

// persistent launch with nblk blocks (0: as many as are resident on the device); bar: one
// zeroed word for the grid barrier.  returns the number of blocks, or 0 on failure
template <typename T, typename TA, int SB, bool PG>
int mega_persist_launch(hipStream_t stream, const MegaArgs<T, TA> &a, TA *acc, unsigned *bar,
                        int eg, int vg, int nblk)
{
  MegaLayout lo = mega_layout<0, MS_N - 1>(a);
  if (SB < MS_N - 1) {
    const MegaLayout l1 = mega_layout<0, (SB < MS_N - 1) ? SB : 0>(a);
    const MegaLayout l2 = mega_layout<(SB < MS_N - 1) ? SB + 1 : 0, MS_N - 1>(a);
    lo.total = std::max(l1.total, l2.total);
  }
  if (lo.total > 65536) return 0;
  int dev = 0, ncu = 0, per = 0;
  if ((hipGetDevice(&dev) != hipSuccess) ||
      (hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess) ||
      (hipOccupancyMaxActiveBlocksPerMultiprocessor(
           &per, reinterpret_cast<const void *>(mega_persist_kernel<T, TA, SB, PG>), 64,
           (size_t) lo.total) != hipSuccess))
    return 0;
  const int resident = ncu * per;
  if ((nblk <= 0) || (nblk > resident)) nblk = resident;
  if (nblk > a.Ni) nblk = a.Ni;
  if (hipMemsetAsync(bar, 0, 3 * sizeof(unsigned), stream) != hipSuccess) return 0;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(mega_persist_kernel<T, TA, SB, PG>), dim3(nblk), dim3(64),
                     (size_t) lo.total, stream, a, acc, bar, eg, vg);
  return (hipGetLastError() == hipSuccess) ? nblk : 0;
}

template <typename T, typename TA>
int mega_persist(hipStream_t stream, const MegaArgs<T, TA> &a, TA *acc, unsigned *bar, int eg,
                 int vg, int nblk, int SB)
{
  if (a.pglobal) {
    if (SB == 3) return mega_persist_launch<T, TA, 3, true>(stream, a, acc, bar, eg, vg, nblk);
    return mega_persist_launch<T, TA, 6, true>(stream, a, acc, bar, eg, vg, nblk);
  }
  if (SB == 3) return mega_persist_launch<T, TA, 3, false>(stream, a, acc, bar, eg, vg, nblk);
  return mega_persist_launch<T, TA, 6, false>(stream, a, acc, bar, eg, vg, nblk);
}

// runs the stages S0..S1 (one of the instantiated ranges); false if not instantiated
template <typename T, typename TA>
bool mega_range(hipStream_t stream, const MegaArgs<T, TA> &a, int S0, int S1, int *query = nullptr)
{
#define LMP_POD_MEGA_R(A, B) \
  if ((S0 == A) && (S1 == B)) return mega_launch<T, TA, A, B>(stream, a, query)
  LMP_POD_MEGA_R(0, 6);
  LMP_POD_MEGA_R(0, 3);
  LMP_POD_MEGA_R(4, 6);
  LMP_POD_MEGA_R(0, 4);
  LMP_POD_MEGA_R(5, 6);
  LMP_POD_MEGA_R(0, 5);
  LMP_POD_MEGA_R(0, 1);
  LMP_POD_MEGA_R(3, 6);
  LMP_POD_MEGA_R(2, 6);
  LMP_POD_MEGA_R(0, 2);
  LMP_POD_MEGA_R(0, 0);
  LMP_POD_MEGA_R(1, 1);
  LMP_POD_MEGA_R(2, 2);
  LMP_POD_MEGA_R(3, 3);
  LMP_POD_MEGA_R(4, 4);
  LMP_POD_MEGA_R(5, 5);
  LMP_POD_MEGA_R(6, 6);
#undef LMP_POD_MEGA_R
  return false;
}

}    // namespace PODHIP
}    // namespace LAMMPS_NS

#endif    // KOKKOS_ENABLE_HIP
#endif
