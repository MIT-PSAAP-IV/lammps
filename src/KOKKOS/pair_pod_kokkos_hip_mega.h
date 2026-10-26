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

// Mega kernel of pair pod/kk (HIP, opt-in: LAMMPS_POD_MEGA=1, see
// PairPODKokkos::mega_block()): the whole native step of a one-element model
// with Hermite many-body radial functions, B-spline 2-body functions, local
// EA-POD and 3+4-body terms (as the Hf model) in one kernel, one wavefront per
// atom.  The stages are the bodies of the native kernels of
// pair_pod_kokkos_hip.h, with the pairs of the atom and all per-atom
// intermediates kept in LDS:
//
//   0 neighbors      -> pair cache (r, rhat, j) of atom i
//   1 2-body desc    -> descriptors d2
//   2 radial-angular -> sumU (matrix cores)
//   3 3/4-body desc  -> descriptors d3, d4
//   4 local EA-POD   -> energy ei (global), coefficients cb
//   5 force coeffs   -> forcecoeff (formed per lane in the force stage)
//   6 forces         -> f (global atomics), energy/virial of the atom
//
// An atom with more than cap pairs keeps them in a per-step pair pool in
// global memory and walks them in windows of cap pairs; if the pool is full it
// is computed by an overflow kernel after the main one, and the host grows the
// pool for the next step.  mega_kernel<.., HF = true> is an instance with the
// sizes of the Hf model as compile-time constants, used when they match.

#ifndef LMP_PAIR_POD_KOKKOS_HIP_MEGA_H
#define LMP_PAIR_POD_KOKKOS_HIP_MEGA_H

#ifdef KOKKOS_ENABLE_HIP

namespace LAMMPS_NS {
namespace PODHIP {

enum { MS_NEIGH = 0, MS_TWOBODY, MS_RAS, MS_DESC, MS_LOCALENV, MS_FC, MS_FORCE, MS_N };

static constexpr int MEGA_NCNT = 5;    // per-step counters of the mega kernel (MegaArgs::cnt)
static constexpr int MEGA_RTM = 2;    // K3 <= 32 (angular degree <= 3)
static constexpr int MEGA_CSM = 2;    // nelements * nrbf3 <= 8
static constexpr int MEGA_FT = 12;                // 4-body terms per monomial (recipe)
static constexpr int MEGA_FRI = 2 + 3 * MEGA_FT;  // ints per monomial in the recipe
static constexpr int MEGA_FRD = 1 + MEGA_FT;      // doubles per monomial in the recipe

static constexpr int MEGA_KM = 4;       // active clusters of the wave-wide local EA-POD
static constexpr int MEGA_S6PAD = 1;    // padding doubles per row of the force stage power table

// DPP move of a double (CTRL: DPP control, ROWMASK: rows written)
template <int CTRL, int ROWMASK> __device__ inline double mega_dpp(const double v)
{
  const long long b = __builtin_bit_cast(long long, v);
  const int lo = __builtin_amdgcn_update_dpp(0, (int) (b & 0xffffffffll), CTRL, ROWMASK, 0xf, false);
  const int hi = __builtin_amdgcn_update_dpp(0, (int) (b >> 32), CTRL, ROWMASK, 0xf, false);
  return __builtin_bit_cast(double, ((long long) hi << 32) | (unsigned int) lo);
}
__device__ inline double mega_lane63(const double v)
{
  const long long b = __builtin_bit_cast(long long, v);
  const int lo = __builtin_amdgcn_readlane((int) (b & 0xffffffffll), 63);
  const int hi = __builtin_amdgcn_readlane((int) (b >> 32), 63);
  return __builtin_bit_cast(double, ((long long) hi << 32) | (unsigned int) lo);
}
// sum over the 64 lanes, in every lane (as wave_sum_dpp of the multi-kernel path)
__device__ inline double mega_wave_sum(double v)
{
  v += mega_dpp<0xb1, 0xf>(v);     // quad_perm [1,0,3,2]
  v += mega_dpp<0x4e, 0xf>(v);     // quad_perm [2,3,0,1]
  v += mega_dpp<0x141, 0xf>(v);    // row_half_mirror
  v += mega_dpp<0x140, 0xf>(v);    // row_mirror
  v += mega_dpp<0x142, 0xa>(v);    // row_bcast15 to rows 1 and 3
  v += mega_dpp<0x143, 0xc>(v);    // row_bcast31 to rows 2 and 3
  return mega_lane63(v);
}
// v of lane ^ 16 (within each half of the wave)
__device__ inline double mega_xor16(const double v)
{
  const long long b = __builtin_bit_cast(long long, v);
  const int lo = __builtin_amdgcn_ds_swizzle((int) (b & 0xffffffffll), 0x401f);
  const int hi = __builtin_amdgcn_ds_swizzle((int) (b >> 32), 0x401f);
  return __builtin_bit_cast(double, ((long long) hi << 32) | (unsigned int) lo);
}

// pod_pair_finish() split: the 3/4-body force of pair j from the pair sums and gradients ...
template <typename T, typename TA>
__device__ inline void mega_pair_force(const T *g, const TA SA, const TA SU, const TA SV,
                                       const TA SW, TA &fx, TA &fy, TA &fz)
{
  fx = SU * static_cast<TA>(g[0]) + SV * static_cast<TA>(g[1]) + SW * static_cast<TA>(g[2]) +
      SA * static_cast<TA>(g[6]);
  fy = SU * static_cast<TA>(g[1]) + SV * static_cast<TA>(g[3]) + SW * static_cast<TA>(g[4]) +
      SA * static_cast<TA>(g[7]);
  fz = SU * static_cast<TA>(g[2]) + SV * static_cast<TA>(g[4]) + SW * static_cast<TA>(g[5]) +
      SA * static_cast<TA>(g[8]);
}
// ... and its 2-body force (B-spline radial functions) and tally: global force atomics on
// atom j, the force and virial of atom i added to the lane sums o[9]
template <typename T, typename TA>
__device__ inline void mega_pair_tally(const ForceTally<T, TA> &ft, const PairGeom<T> &pg,
                                       const int j, TA fx, TA fy, TA fz, TA (&o)[9])
{
  T u, v, w;
  pg.uvw(j, u, v, w);
  const T r = pg.r(j);
  {
    TA cR = TA(0);
    T Bq[4], dBq[4];
    const int ii = ft.Ni * pod_bspline_pair(ft.bs, pg, j, Bq, dBq);
#pragma unroll
    for (int m = 0; m < 4; ++m) cR += static_cast<TA>(ft.cb2[ii + ft.Ni * m] * dBq[m]);
    cR *= static_cast<TA>(r);    // dBq is R'(r)/r
    fx += cR * static_cast<TA>(u);
    fy += cR * static_cast<TA>(v);
    fz += cR * static_cast<TA>(w);
  }
  TA *fj = ft.f + ft.aj[j] * ft.fs0;
  atomicAdd(fj, -fx);
  atomicAdd(fj + ft.fs1, -fy);
  atomicAdd(fj + 2 * ft.fs1, -fz);
  o[0] += fx;
  o[1] += fy;
  o[2] += fz;
  if (ft.vir) {
    const TA rx = static_cast<TA>(r * u), ry = static_cast<TA>(r * v), rz = static_cast<TA>(r * w);
    o[3] += rx * fx;
    o[4] += ry * fy;
    o[5] += rz * fz;
    o[6] += rx * fy;
    o[7] += rx * fz;
    o[8] += ry * fz;
  }
}

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

// x^n by repeated multiplication from 1, as the power tables (bit-identical values)
template <typename T> __device__ inline T pod_ipow(const T x, const int n)
{
  T p = static_cast<T>(1.0);
  for (int q = 0; q < n; q++) p *= x;
  return p;
}




// pod_monomial() from a power row [1][u^1..u^L][v^1..v^L][w^1..w^L] and the table offsets of
// x^a, x^(a-1) (0, i.e. the 1, for a = 0) of the three directions, packed in mo.x (5 bits
// each: a b c a-1 b-1 c-1) with the exponents in mo.y (4 bits each); same products as pod_monomial
template <typename T>
__device__ inline void pod_monomial_tab(const T *pd, const int2 mo, T &P, T &Pu, T &Pv, T &Pw)
{
  const T pa = pd[mo.x & 31], pb = pd[(mo.x >> 5) & 31], pc = pd[(mo.x >> 10) & 31];
  const T pa1 = pd[(mo.x >> 15) & 31], pb1 = pd[(mo.x >> 20) & 31], pc1 = pd[(mo.x >> 25) & 31];
  const T pbc = pb * pc;
  P = pa * pbc;
  Pu = static_cast<T>(mo.y & 15) * pa1 * pbc;
  Pv = static_cast<T>((mo.y >> 4) & 15) * pa * pb1 * pc;
  Pw = static_cast<T>((mo.y >> 8) & 15) * pa * pb * pc1;
}

template <typename T, typename TA> struct MegaArgs {
  // neighbors
  NeighArgs na;
  const T *x, *rcutsq;
  int gi1, Ni;
  int cap;        // pairs per atom kept in LDS (more: the pair pool)

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

  int lenv;     // 1: local EA-POD with all lanes of the wave, 0: as the multi-kernel path
  int n_pn3;    // entries of pn3
  const int *extab;    // exponents a | b<<8 | c<<16 of the monomials k < K3 (host table)
  const int2 *mono;    // per monomial the 6 power table offsets (5 bits each) and a|b<<4|c<<8
  int hf;       // 1: the sizes are those of the Hf model (mega_hf_sizes), compile-time kernel
  // freg == 2: per monomial k the terms of forcecoeff34_atom() for one element, built on the
  // host: frec_i[MEGA_FRI k] = active 3-body index or -1, number of 4-body terms, then per
  // term (p4, iA, iB); frec_d[MEGA_FRD k] = pc3[k], then pc4 of each term
  const int *frec_i;
  const T *frec_d;
  const int *rowmono;    // Hf kernel: monomial k of A row 0..31 of the force stage, or -1
  int freg;     // 2: force coefficients formed per lane (frec_*), 1: by stage 5 in LDS

  // pairs beyond cap (stages 0-6 in one kernel, pairs in LDS): an atom with more than cap pairs
  // stores all of them in a per-step pool, pool_n pairs (rhat stride pool_n), at an offset
  // claimed from *pool_ctr (zeroed before each launch).  if the pool is full, the windows of
  // cap pairs are found again from the neighbor list instead.  spill == 0: pairs beyond cap
  // are dropped and flag[0] reports them (the run then stops)
  T *gpr, *gprh;
  int *gpaj, *pool_ctr;
  int pool_n, spill;
  int *ovf;     // nullptr, or the atoms whose pairs did not fit in the pool (count in cnt[4])
  // per-step counters (MEGA_NCNT ints): [0] pool counter (pool_ctr), [1] atoms beyond cap,
  // [2] of which pool full, [3] largest pair count beyond cap, [4] overflow atoms; cntz: the
  // counters of the next step, zeroed by the overflow kernel (or nullptr)
  int *cnt, *cntz;
  TA *fdummy;   // force sink of those atoms in mega_kernel (the overflow kernel adds their forces)
  int onetype;    // 1: one atom type mapped to the one element (no type/map/rcutsq gathers)
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
  lo.z = o;    // the atom index of i
  if (lo.needP) o += 16;
  lo.p = o;
  if (lo.needP) o += mega_round(a.cap * (4 * (int) sizeof(T) + (int) sizeof(int)));
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
    const int np2 = 3 * L1;    // power table
    w = mega_max(w, ((np2 + a.nrbf3) * jt + a.nrbf3) * (int) sizeof(T) +
                (jt + a.K3) * (int) sizeof(int));
  }
  if (mega_has(S0, S1, 3))
    w = mega_max(w, mega_round(nout * (int) sizeof(T)) +
                (a.dg.nabf3_active + a.n_pn3 + a.K3 + a.dg.nabf4 + 1 + 4 * a.dg.Q4) * (int) sizeof(int));
  if (mega_has(S0, S1, 4) && !a.lenv)
    w = mega_max(w, mega_round(nout * (int) sizeof(T)) + (4 + 4 * LP_KMAX + 3 * LP_KMAX + 2) * (int) sizeof(TA));
  if (mega_has(S0, S1, 5) && !flane)
    w = mega_max(w, mega_round(nout * (int) sizeof(T)) + (a.K3 + a.fc.Q4) * (int) sizeof(int));
  if (flane) w = mega_max(w, nout * (int) sizeof(T));
  int end6 = 0;
  lo.w6 = lo.w;
  if (mega_has(S0, S1, 6)) {
    // the atom sums, then (generic kernel) the power table of the pair tile: one shared 1, x^1..x^L
    // per direction and a padding entry per pair, the pair elements and the monomial tables; the
    // Hf kernel forms its monomials in registers (sym6)
    const bool sym6 = a.hf;
    const int t6 = sym6 ? mega_round(9 * (int) sizeof(TA)) : mega_round(9 * (int) sizeof(TA)) +
        MFMA_PAIR_TILE * (3 * a.L + 1 + MEGA_S6PAD) * (int) sizeof(T) +
        MFMA_PAIR_TILE * (int) sizeof(int) + a.K3 * 2 * (int) sizeof(int);
    // after the 2-body coefficients cb2, over cb3/cb4 (dead once the force coefficients are in
    // registers) and the stage scratch
    if (lo.needD) {
      lo.w6 = lo.d + mega_round(a.nl2 * (int) sizeof(T));
      end6 = lo.w6 + mega_round(t6);
    } else
      w = mega_max(w, t6);
  }
  o += mega_round(w);
  lo.total = mega_max(o, end6);
  return lo;
}

// stages S0..S1 of atom i (the kernels run S0 = 0, S1 = MS_N - 1).  HF: the sizes of the Hf
// model as constants; RESCAN: the overflow kernel (pair windows found again from the neighbors)
template <typename T, typename TA, int S0, int S1, bool HF = false, bool RESCAN = false>
__device__ inline void mega_atom(const MegaArgs<T, TA> &a, char *lds_raw, const int i)
{
  const MegaLayout lo = mega_layout<S0, S1>(a);
  const int lane = threadIdx.x;
  const int lr = lane & 15, lq = lane >> 4;
  const int cap = a.cap;
  const int nelements = 1;
  const int nrbf3 = a.nrbf3, K3 = a.K3, L1 = a.L + 1;
  const int nout = nrbf3 * K3;
  const int nDes = a.nDes;

  int *gi_s = reinterpret_cast<int *>(lds_raw + lo.z);   // atom index of i
  int *const zt = nullptr;    // element indices of the pairs: one element
  T *d_s = reinterpret_cast<T *>(lds_raw + lo.d);        // descriptors, then cb
  T *f_s = reinterpret_cast<T *>(lds_raw + lo.f);        // forcecoeff [m][k][e]
  char *w_s = lds_raw + lo.w;                            // stage scratch
  T *u_s = reinterpret_cast<T *>(w_s);                   // sumU [m][k][e] at the start of w

  const int gi = a.na.ilist[a.gi1 + i];
  const int itype = a.onetype ? 0 : a.na.map[a.na.type[gi]];

  // pair cache in LDS: r, rhat [d*cap + p], neighbor index
  T *r_s = reinterpret_cast<T *>(lds_raw + lo.p);
  T *rh_s = r_s + cap;
  int *aj_s = reinterpret_cast<int *>(rh_s + 3 * cap);
  const int PN = cap;
  const PairGeom<T> pg{r_s, rh_s, PN};
  if (lo.needP && (lane == 0)) gi_s[0] = gi;
  RadialSrc<T> rs = a.rs;
  // one element, as a constant (pod-hip's helpers test ne > 1 for the element indices)
  rs.ti = zt;
  rs.tj = zt;
  rs.ne = 1;
  rs.rbf = nullptr;    // the Hermite table only (no stored radial functions)
  rs.drbf = nullptr;

  int np = 0;

  // the pairs beyond cap (see MegaArgs::spill): poff = offset of the pairs of atom i in the
  // pool or -1, curw = first pair held in the LDS cache.  the stages that read the pairs walk
  // them in windows of cap pairs; with at most cap pairs there is a single window
  constexpr bool SPILL = (S0 == 0) && (S1 == MS_N - 1);
  int poff = -1, curw = 0;

  // the pairs n of the neighbor list inside the cutoff with lo <= n < hi, to (pr, prh,
  // paj)[n - lo] (rhat stride ps), in the order of the list; returns the number of pairs
  auto scan = [&](const int lo, const int hi, T *pr, T *prh, int *paj, const int ps) -> int {
    const T xi0 = a.x[gi * a.na.xs0];
    const T xi1 = a.x[gi * a.na.xs0 + a.na.xs1];
    const T xi2 = a.x[gi * a.na.xs0 + 2 * a.na.xs1];
    const int jnum = a.na.numneigh[gi];
    const T rc2one = a.onetype ? a.rcutsq[0] : T(0);
    const unsigned long long below = (1ull << lane) - 1ull;
    int cnt = 0;
    for (int j0 = 0; j0 < jnum; j0 += 64) {
      const int jj = j0 + lane;
      bool in = false;
      int gj = 0, jtype = 0;
      T dx = T(0), dy = T(0), dz = T(0);
      if (a.onetype) {
        if (jj < jnum) {
          gj = a.na.neighbors[gi * a.na.ns0 + jj * a.na.ns1] & NEIGHMASK;
          dx = a.x[gj * a.na.xs0] - xi0;
          dy = a.x[gj * a.na.xs0 + a.na.xs1] - xi1;
          dz = a.x[gj * a.na.xs0 + 2 * a.na.xs1] - xi2;
          in = (dx * dx + dy * dy + dz * dz) < rc2one;
        }
      } else if (jj < jnum)
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
        mega_reload<T>(r_s, rh_s, aj_s, gr, gh, ga, nw, cap, pn);
      }
      else if constexpr (RESCAN)
        scan(wb, wb + cap, r_s, rh_s, aj_s, cap);
      __syncthreads();
      curw = __builtin_amdgcn_readfirstlane(wb);
    } else
      (void) wb;
  };

  // the atom's pairs did not fit in the pool: it is listed for the overflow kernel and computed
  // here with its first window only, its forces going to a dummy (wave-uniform)
  bool ovfskip = false;

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
          atomicAdd(a.cnt + 1, 1);                 // atoms with pairs beyond cap
          if (off < 0) atomicAdd(a.cnt + 2, 1);    // of which found again (pool full)
          if (cnt > __hip_atomic_load(a.cnt + 3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))
            atomicMax(a.cnt + 3, cnt);
        }
        poff = __builtin_amdgcn_readfirstlane(__shfl(off, 0));    // wave-uniform: SGPR
        if (poff >= 0) scan(0, cnt, a.gpr + poff, a.gprh + poff, a.gpaj + poff, a.pool_n);
        // pool full: the atom is processed by the overflow kernel (its windows found again from
        // the neighbor list), after this one
        if ((poff < 0) && !RESCAN && a.ovf) {
          if (lane == 0) a.ovf[atomicAdd(a.cnt + 4, 1)] = i;
          ovfskip = true;
        }
        // pool full in the overflow kernel: the windows found again from the neighbor list
        spilled = (poff >= 0) || RESCAN;
        if (spilled) np = cnt;
      }
    }
    if (!spilled) {
      if ((cnt > cap) && (lane == 0) && !ovfskip &&
          (cnt > __hip_atomic_load(a.flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)))
        atomicMax(a.flag, cnt);
      np = min(cnt, cap);
    }
    __syncthreads();
  }

  // ---- stage 1: 2-body descriptors d2(m) = sum_j B_m(r_j), as twobodydesc_bspline_kernel
  if constexpr (mega_has(S0, S1, MS_TWOBODY)) {
    TA *s_s = reinterpret_cast<TA *>(w_s);
    BsplineArgs<T> bs = a.bs;
    bs.ti = zt;
    bs.tj = zt;
    bs.ne = 1;
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
    if constexpr (SPILL) {
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
  constexpr int LPRE_AT = mega_has(S0, S1, MS_LOCALENV) ? ((3 < S0) ? S0 : 3) : -1;
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
    if constexpr (LPRE_AT == MS_RAS) lpre_load();
    constexpr int RTM = MEGA_RTM, CTM = 1;
    const int jtile = a.jtile;
    const int nrbf3 = HF ? 5 : a.nrbf3;
    const int NC = nelements * nrbf3;
    const int RT = (K3 + 15) / 16, CT = (NC + 15) / 16;
    // power table [jtile][3][PL]: x^0..x^L per direction
    const int PL = L1;
    T *pow_s = reinterpret_cast<T *>(w_s);
    const int ldr = jtile + 1;
    T *rbf_s = pow_s + jtile * 3 * PL;    // [nrbf3][ldr]
    int *tj_s = reinterpret_cast<int *>(rbf_s + nrbf3 * ldr);    // [jtile]
    // table offsets of the powers, from the exponents a | b<<8 | c<<16 (host table extab)
    int oa[RTM], ob[RTM], oc[RTM];
#pragma unroll
    for (int rt = 0; rt < RTM; rt++) {
      const int k = rt * 16 + lr;
      const int ex = (k < K3) ? a.extab[k] : 0;
      const int o = PL;
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

    ensure(0);
    auto win2 = [&](const int nj) {
    for (int j0 = 0; j0 < nj; j0 += jtile) {
      const int nt = min(jtile, nj - j0);
      const int nt4 = (nt + 3) & ~3;
      __syncthreads();
      // the tile filled by (pair, radial function) and (pair, direction) lanes if they fit in
      // the wave (with the Hf sizes the test is on the tile size and resolved at compile time)
      if (HF ? (jtile * (nrbf3 + 3) <= 64) : (nt4 * (nrbf3 + 3) <= 64)) {
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
          {
            T u, v, w;
            pg.uvw(j, u, v, w);
            T pu = static_cast<T>(1.0), pv = pu, pz = pu;
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
        const T *pw = pow_s + jj * 3 * L1;
#pragma unroll
        for (int rt = 0; rt < RTM; rt++)
          av[rt] = (rt * 16 + lr < K3) ? static_cast<TA>(pw[oa[rt]] * pw[ob[rt]] * pw[oc[rt]])
                                       : TA(0);
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
    if constexpr (SPILL) {
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
    if constexpr (LPRE_AT == MS_DESC) lpre_load();
    DescArgs<T> g = a.dg;
    g.d3 = d_s + a.nl2;
    g.d4 = d_s + a.nl2 + a.nl3;
    g.Ni = 1;
    {
      // index tables in LDS after sumU: p3_active, pn3, pc3, pa4, pb4, pc4
      int *t_s = reinterpret_cast<int *>(w_s + mega_round(nout * (int) sizeof(T)));
      const int n1 = g.nabf3_active, n2 = a.n_pn3, n3 = K3, n4 = g.nabf4 + 1, n5 = 3 * g.Q4,
                n6 = g.Q4;
      int *s1 = t_s, *s2 = s1 + n1, *s3 = s2 + n2, *s4 = s3 + n3, *s5 = s4 + n4, *s6 = s5 + n5;
      for (int t = lane; t < n1 + n2 + n3 + n4 + n5 + n6; t += 64) {
        int v;
        if (t < n1) v = g.p3_active[t];
        else if (t < n1 + n2) v = g.pn3[t - n1];
        else if (t < n1 + n2 + n3) v = g.pc3[t - n1 - n2];
        else if (t < n1 + n2 + n3 + n4) v = g.pa4[t - n1 - n2 - n3];
        else if (t < n1 + n2 + n3 + n4 + n5) v = g.pb4[t - n1 - n2 - n3 - n4];
        else v = g.pc4[t - n1 - n2 - n3 - n4 - n5];
        t_s[t] = v;
      }
      g.p3_active = s1;
      g.pn3 = s2;
      g.pc3 = s3;
      g.pa4 = s4;
      g.pb4 = s5;
      g.pc4 = s6;
      __syncthreads();
    }
    // as CPU pair pod: no 3-body terms with at most one pair, no 4-body with at most two
    desc34_atom<T, TA>(g, u_s, 0, np);
    __syncthreads();
  }

  // ---- stage 4: local EA-POD (localenv_pca_split, localenv_atom_split, localenv_cb).
  // a.lenv == 1: the wave works on its one atom with all lanes (wave sums for the
  // projections, ballots instead of the binary searches); a.lenv == 0: the same
  // decomposition as the multi-kernel path (4 partial sums, binary searches, lane 0)
  if constexpr (mega_has(S0, S1, MS_LOCALENV)) {
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
      const TA pcai = mega_wave_sum(p);
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
        if (k < kni) sumE[k] = mega_wave_sum(sumE[k]);
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
        a.ei[i] = static_cast<T>(eival);
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
      a.ei[i] = static_cast<T>(eival);
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
    const bool flane = (a.freg == 2) && mega_has(S0, S1, MS_FORCE);
    FcArgs<T> g = a.fc;
    g.cb3 = d_s + a.nl2;
    g.cb4 = d_s + a.nl2 + a.nl3;
    g.sumU = u_s;
    g.Ni = 1;
    int *qa_s = reinterpret_cast<int *>(w_s + mega_round(nout * (int) sizeof(T)));
    if (!flane) forcecoeff34_atom<T>(g, f_s, u_s, qa_s, qa_s + K3, 0, np);
  }

  // ---- stage 6: forces and tally, as allbody_forces_mfma_kernel<CSM = 2> with the 2-body
  // forces (HIP_FTALLY)
  if constexpr (mega_has(S0, S1, MS_FORCE)) {
    constexpr int CSM = MEGA_CSM, JT = MFMA_PAIR_TILE;
    const int nrbf3 = HF ? 5 : a.nrbf3;
    // power table (generic kernel): row of pair jj = [1][x^1..x^L] x 3 directions, padded to an
    // odd count of doubles (no LDS bank conflicts between the 16 pair rows read by a row group)
    const int PL = L1 - 1;    // powers per direction
    const int PO = 1;         // row[0] = 1
    const int PD = 3 * PL + PO + MEGA_S6PAD;
    const int NeK3 = nelements * K3;
    const int NC = nelements * nrbf3;
    const int CS = (NC + 3) / 4;
    const int KT = (K3 + 15) / 16;
    char *w6_s = lds_raw + lo.w6;    // force stage scratch (over cb3/cb4 and U)
    TA *acc_s = reinterpret_cast<TA *>(w6_s);
    T *pd_s = reinterpret_cast<T *>(w6_s + mega_round(9 * (int) sizeof(TA)));    // [JT][PD]
    int *tj_s = reinterpret_cast<int *>(pd_s + JT * PD);                           // [JT]
    int2 *mo_s = reinterpret_cast<int2 *>(tj_s + JT);    // [K3] table offsets and exponents
    const T *fc_s = f_s;
    int nj = 0;

    ForceTally<T, TA> ft = a.ft;
    if (ovfskip) {    // the overflow kernel adds this atom's forces
      ft.f = a.fdummy;
      ft.fs0 = 0;
      ft.fs1 = 0;
    }
    ft.ai = gi_s;    // ai[0] = gi
    ft.aj = aj_s;
    ft.cb2 = d_s;
    ft.Ni = 1;
    ft.bs.ti = zt;
    ft.bs.tj = zt;
    ft.bs.ne = 1;

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
    // SYM (Hf kernel): the angular monomials formed in registers from the pair's (u, v, w),
    // without the power table: lane group g of the MFMA output holds 8 template monomials 1, X,
    // X^2, XY, X^3, X^2 Y, X Y^2, XYZ of (X, Y, Z) = sigma_g(u, v, w) (sigma = identity, two
    // cyclic shifts, swap of u and v), which cover the 20 monomials of degree <= 3 (the A rows
    // of repeated monomials are zero, host table rowmono)
    constexpr bool SYM = HF;
    // monomial of A row q (SYM: host table), or -1
    auto rowk = [&](const int q) -> int { return SYM ? a.rowmono[q] : ((q < K3) ? q : -1); };
    const bool flane = (a.freg == 2) && mega_has(S0, S1, MS_FC);
    // forcecoeff34_atom() for one element, entry (m, k) only (see below)
    auto frecipe = [&](const int m, const int k) -> T {
      const T *cb3 = d_s + a.nl2, *cb4 = d_s + a.nl2 + a.nl3;
      const T *um = u_s + K3 * m;
      const int *ri = a.frec_i + MEGA_FRI * k;
      const T *rd = a.frec_d + MEGA_FRD * k;
      const int a3 = ri[0], cnt = ri[1];
      T f = T(0);
      // as CPU pair pod: 3-body terms with more than one pair, 4-body with more than two
      if ((a3 >= 0) && (np > 1)) {
        const T wq = rd[0] * cb3[a3 + a.fc.nabf3_active * m];
        const T t = wq * um[k];
        f = t + t;
      }
      if ((m < a.fc.nrbf4) && (np > 2)) {
        const int nb4 = a.fc.nabf4;
#pragma unroll
        for (int t = 0; t < MEGA_FT; t++)
          if (t < cnt) {
            const int p4 = ri[2 + 3 * t], iA = ri[3 + 3 * t], iB = ri[4 + 3 * t];
            f += (rd[1 + t] * cb4[p4 + nb4 * m]) * (um[iA] * um[iB]);
          }
      }
      return f;
    };
    if (flane) {
      // formed for the NC*K3 valid (c, k) entries packed over the lanes, entry e = c K3 + k in
      // lane e % 64 of round e / 64, then moved to the operand layout with lane shuffles
      constexpr int RMAX = (MEGA_RTM * 16 * CSM * 4 + 63) / 64;
      const int NE = NC * K3;
      T fr[RMAX];
#pragma unroll
      for (int r = 0; r < RMAX; r++) {
        fr[r] = T(0);
        const int e = lane + 64 * r;
        if ((64 * r < NE) && (e < NE)) {
          const int cc = e / K3, k = e - cc * K3;
          fr[r] = frecipe(cc % nrbf3, k);
        }
      }
#pragma unroll
      for (int kt = 0; kt < MEGA_RTM; kt++)
#pragma unroll
        for (int cs = 0; cs < CSM; cs++) {
          const int k = rowk(16 * kt + lr), cc = 4 * cs + lq;
          const bool valid = (kt < KT) && (cs < CS) && (k >= 0) && (ce[cs] >= 0);
          const int e = valid ? cc * K3 + k : 0;
          T v = T(0);
#pragma unroll
          for (int r = 0; r < RMAX; r++)
            if (64 * r < NE) {
              const T t = __shfl(fr[r], e & 63);
              if ((e >> 6) == r) v = t;
            }
          fa[kt][cs] = valid ? static_cast<TA>(v) : TA(0);
        }
    } else
#pragma unroll
    for (int kt = 0; kt < MEGA_RTM; kt++)
#pragma unroll
      for (int cs = 0; cs < CSM; cs++) {
        const int kr = rowk(16 * kt + lr), k = (kr >= 0) ? kr : 0;
        const bool valid = (kt < KT) && (cs < CS) && (kr >= 0) && (ce[cs] >= 0);
        fa[kt][cs] = valid ? static_cast<TA>(fc_s[ce[cs] + nelements * k + NeK3 * cm[cs]]) : TA(0);
      }
    __syncthreads();
    if (lane < 9) acc_s[lane] = TA(0);
    if (!SYM && (lane < K3)) mo_s[lane] = a.mono[lane];
    typedef typename Mfma<TA>::acc_t acc_t;
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
    if constexpr (SPILL) ensure(0);
    if (!SYM) fetch(0);
    // the tally of the pair forces (2-body force, global atomics) is deferred over up to 4 pair
    // tiles, tile t kept by lane group t % 4, so that it runs with up to 64 lanes instead of 16:
    // the deferred pair of this lane (window index qj, -1: none) and its sums
    int qj = -1, tc = 0;
    TA q0 = TA(0), q1 = TA(0), q2 = TA(0), q3 = TA(0);
    auto flush = [&]() {
      TA o[9];
#pragma unroll
      for (int d = 0; d < 9; d++) o[d] = TA(0);
      if (qj >= 0) {
        TA fx, fy, fz;
        T g[9];
        pod_pair_gradients<T>(g, pg, qj);
        mega_pair_force<T, TA>(g, q0, q1, q2, q3, fx, fy, fz);
        mega_pair_tally<T, TA>(ft, pg, qj, fx, fy, fz, o);
      }
      qj = -1;
      // the force and virial of atom i summed over the lanes
      {
        const int nd = ft.vir ? 9 : 3;
        for (int d = 0; d < nd; d++) {
          const TA s = mega_wave_sum(o[d]);
          if (lane == 0) acc_s[d] += s;
        }
      }
    };
    // with more than cap pairs, a tile that does not fit in the LDS window starts a new window
    // (and is fetched again); otherwise the window holds all pairs
    for (int j0 = 0; j0 < nj; j0 += JT) {
      const int nt = min(JT, nj - j0);
      if constexpr (SPILL) {
        if (j0 - curw + nt > cap) {
          ensure(j0);
          fetch(j0);
        }
      }
      const int jl0 = __builtin_amdgcn_readfirstlane(j0 - curw);
      if (!SYM) __syncthreads();
      if (!SYM && (lane < 3 * JT)) {
        const int jj = lane % JT, d = lane / JT;
        T *pd = pd_s + jj * PD + PO + d * PL;
        const bool in = (jj < nt);
        T p = in ? xh : T(0);
        if (PO && (d == 0)) pd_s[jj * PD] = static_cast<T>(1.0);
        for (int n = 0; n < PL; n++) {
          pd[n] = p;
          p *= xh;
        }
        if (d == 0) tj_s[jj] = in ? tjh : -1;
      }
      if (!SYM) {
        if ((j0 + JT < nj) && (j0 + JT - curw < cap)) fetch(j0 + JT);
        __syncthreads();
      }

      for (int s0 = 0; s0 < nt; s0 += 16) {
        TA SA = TA(0), SU = TA(0), SV = TA(0), SW = TA(0);
        const int jb = s0 + lr;
        const int tjb = SYM ? ((jb < nt) ? 0 : -1) : tj_s[jb];
        const PairRadial<T> prad(rs, pg, jl0 + min(jb, nt - 1));
        const T *pd = pd_s + jb * PD;
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
        if constexpr (SYM) {
          // (X, Y, Z) = sigma_g(u, v, w) for lane group g = lq; rows lq + 4 v of tile 0 are the
          // template slots v, rows 16 + lq + 4 v of tile 1 the slots 4 + v
          T ux, vx, wx;
          pg.uvw(jl0 + min(jb, nt - 1), ux, vx, wx);
          const T X = (lq == 0) ? ux : ((lq == 2) ? wx : vx);
          const T Y = (lq == 0) ? vx : ((lq == 1) ? wx : ux);
          const T Z = (lq == 1) ? ux : ((lq == 2) ? vx : wx);
          const T X2 = X * X, XY = X * Y;
          acc_t accG0 = acc_t{0, 0, 0, 0}, accH0 = acc_t{0, 0, 0, 0};
          acc_t accG1 = acc_t{0, 0, 0, 0}, accH1 = acc_t{0, 0, 0, 0};
#pragma unroll
          for (int cs = 0; cs < CSM; cs++)
            if (cs < CS) {
              accG0 = Mfma<TA>::mma(fa[0][cs], br[cs], accG0);
              accH0 = Mfma<TA>::mma(fa[0][cs], bdr[cs], accH0);
              accG1 = Mfma<TA>::mma(fa[1][cs], br[cs], accG1);
              accH1 = Mfma<TA>::mma(fa[1][cs], bdr[cs], accH1);
            }
          TA SX = TA(0), SY = TA(0), SZ = TA(0);
          // slots 0-3: 1, X, X^2, XY
          SA += accH0[0];
          SA += accH0[1] * X;
          SX += accG0[1];
          SA += accH0[2] * X2;
          SX += accG0[2] * (X + X);
          SA += accH0[3] * XY;
          SX += accG0[3] * Y;
          SY += accG0[3] * X;
          // slots 4-7: X^3, X^2 Y, X Y^2, XYZ
          SA += accH1[0] * (X2 * X);
          SX += accG1[0] * (T(3) * X2);
          SA += accH1[1] * (X2 * Y);
          SX += accG1[1] * (XY + XY);
          SY += accG1[1] * X2;
          SA += accH1[2] * (XY * Y);
          SX += accG1[2] * (Y * Y);
          SY += accG1[2] * (XY + XY);
          SA += accH1[3] * (XY * Z);
          SX += accG1[3] * (Y * Z);
          SY += accG1[3] * (X * Z);
          SZ += accG1[3] * XY;
          // back to (u, v, w): d/du is d/dX (g 0), d/dZ (g 1), d/dY (g 2, 3), ...
          SU = (lq == 0) ? SX : ((lq == 1) ? SZ : SY);
          SV = (lq == 0) ? SY : ((lq == 2) ? SZ : SX);
          SW = (lq == 1) ? SY : ((lq == 2) ? SX : SZ);
        } else
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
              pod_monomial_tab<T>(pd, mo_s[kv], P, Pu, Pv, Pw);
              SA += accH[v] * static_cast<TA>(P);
              SU += accG[v] * static_cast<TA>(Pu);
              SV += accG[v] * static_cast<TA>(Pv);
              SW += accG[v] * static_cast<TA>(Pw);
            }
          }
        }
        SA += mega_xor16(SA);
        SU += mega_xor16(SU);
        SV += mega_xor16(SV);
        SW += mega_xor16(SW);
        SA += __shfl_xor(SA, 32);
        SU += __shfl_xor(SU, 32);
        SV += __shfl_xor(SV, 32);
        SW += __shfl_xor(SW, 32);
        static_assert(JT == 16, "deferred tally: one 16-pair chunk per tile");
        if ((lq == (tc & 3)) && (jb < nt)) {
          q0 = SA;
          q1 = SU;
          q2 = SV;
          q3 = SW;
          qj = jl0 + jb;
        }
      }
      // flush every 4 tiles, after the last, and before a tile that reloads the window
      tc++;
      {
        const int jn = j0 + JT;
        bool fl = ((tc & 3) == 0) || (jn >= nj);
        if constexpr (SPILL) fl = fl || (jn - curw + min(JT, nj - jn) > cap);
        if (fl) flush();
      }
    }
    __syncthreads();
    pod_atom_finish<T, TA>(ft, i, 0, np, acc_s);
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

// the sizes of the Hf model as constants, so that loops and index arithmetic of the specialized
// kernel are resolved at compile time; the host checks that they hold (mega_is_hf).  (the
// many-body radial functions nrbf3 = 5 are constants in the stages that use them)
template <typename T, typename TA> __host__ __device__ inline void mega_hf_sizes(MegaArgs<T, TA> &b)
{
  b.K3 = 20;
  b.L = 3;
  b.nrbf2 = 40;
  b.ft.nrbf2 = 40;
  b.ft.nm2 = 4;
  {
    b.nDes = 66;
    b.nl2 = 40;
    b.nl3 = 20;
    b.nl4 = 6;
    b.nCls = 50;
    b.nActMax = 3;
    b.nCoeff = 3301;
    b.n_pn3 = 5;
  }
  b.jtile = 8;
  {
    b.dg.nelements = b.fc.nelements = 1;
    b.dg.nrbf3 = b.fc.nrbf3 = 5;
    b.dg.nabf3_active = b.fc.nabf3_active = 4;
    b.dg.nrbf4 = b.fc.nrbf4 = 3;
    b.dg.nabf4 = b.fc.nabf4 = 2;
    b.dg.K3 = b.fc.K3 = 20;
    b.dg.Q4 = b.fc.Q4 = 4;
    b.dg.do4 = b.fc.do4 = 1;
  }
}

template <typename T, typename TA> inline bool mega_is_hf(const MegaArgs<T, TA> &a)
{
  MegaArgs<T, TA> b = a;
  mega_hf_sizes(b);
  return (a.K3 == b.K3) && (a.L == b.L) && (a.nrbf3 == b.nrbf3) && (a.nrbf2 == b.nrbf2) &&
      (a.nDes == b.nDes) && (a.nl2 == b.nl2) && (a.nl3 == b.nl3) && (a.nl4 == b.nl4) &&
      (a.nCls == b.nCls) && (a.nActMax == b.nActMax) && (a.nCoeff == b.nCoeff) &&
      (a.jtile == b.jtile) && (a.n_pn3 == b.n_pn3) && (a.dg.nelements == 1) &&
      (a.dg.nrbf3 == 5) && (a.dg.nabf3_active == 4) && (a.dg.nrbf4 == 3) && (a.dg.nabf4 == 2) &&
      (a.dg.K3 == 20) && (a.dg.Q4 == 4) && (a.dg.do4 == 1) && (a.fc.nrbf4 == 3) &&
      (a.fc.nabf4 == 2) && (a.fc.Q4 == 4) && (a.fc.nabf3_active == 4) && (a.ft.nrbf2 == 40) &&
      (a.ft.nm2 == 4);
}

template <typename T, typename TA, int S0, int S1, bool HF = false>
__global__ void __launch_bounds__(64) LMP_POD_MEGA_ATTR mega_kernel(const MegaArgs<T, TA> a)
{
  extern __shared__ char lds_raw[];
  if constexpr (HF) {
    MegaArgs<T, TA> b = a;
    mega_hf_sizes(b);
    mega_atom<T, TA, S0, S1, true>(b, lds_raw, blockIdx.x);
  } else
    mega_atom<T, TA, S0, S1>(a, lds_raw, blockIdx.x);
}

// overflow kernel: the atoms left by mega_kernel because the pair pool was full (a.ovf, count
// in flag[8]); their pair windows are found again from the neighbor list in each stage.  Rare
// (the host grows the pool from the pairs claimed), so a grid-stride loop of a few blocks
template <typename T, typename TA, bool HF>
__global__ void __launch_bounds__(64) mega_ovf_kernel(const MegaArgs<T, TA> a)
{
  extern __shared__ char lds_raw[];
  const int n = __hip_atomic_load(a.cnt + 4, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  // the counters of the next step (the other parity) are zeroed here instead of by a memset
  if ((blockIdx.x == 0) && a.cntz && (threadIdx.x < MEGA_NCNT)) a.cntz[threadIdx.x] = 0;
  MegaArgs<T, TA> b = a;
  if constexpr (HF) mega_hf_sizes(b);
  b.ovf = nullptr;
  b.pool_n = 0;
  for (int k = blockIdx.x; k < n; k += gridDim.x) {
    __syncthreads();
    mega_atom<T, TA, 0, MS_N - 1, HF, true>(b, lds_raw, a.ovf[k]);
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
    const void *k = reinterpret_cast<const void *>(mega_kernel<T, TA, S0, S1>);
    if constexpr ((S0 == 0) && (S1 == MS_N - 1))
      if (a.hf)
        k = reinterpret_cast<const void *>(mega_kernel<T, TA, S0, S1, true>);
    return hipOccupancyMaxActiveBlocksPerMultiprocessor(&query[1], k, 64, (size_t) lo.total) ==
        hipSuccess;
  }
  {
    bool done = false;
    if constexpr ((S0 == 0) && (S1 == MS_N - 1)) {
      if (a.hf) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(mega_kernel<T, TA, S0, S1, true>), dim3(a.Ni),
                           dim3(64), (size_t) lo.total, stream, a);
        done = true;
      }
    }
    if (!done)
      hipLaunchKernelGGL(HIP_KERNEL_NAME(mega_kernel<T, TA, S0, S1>), dim3(a.Ni), dim3(64),
                         (size_t) lo.total, stream, a);
  }
  return hipGetLastError() == hipSuccess;
}

// the overflow kernel of a full-range launch (after mega_launch<0, MS_N - 1>)
template <typename T, typename TA>
bool mega_ovf_launch(hipStream_t stream, const MegaArgs<T, TA> &a, const int nblk)
{
  if (!a.ovf) return true;
  const MegaLayout lo = mega_layout<0, MS_N - 1>(a);
  const int g = std::max(1, std::min(nblk, a.Ni));
  bool done = false;
  {
    if (a.hf) {
      hipLaunchKernelGGL(HIP_KERNEL_NAME(mega_ovf_kernel<T, TA, true>), dim3(g), dim3(64),
                         (size_t) lo.total, stream, a);
      done = true;
    }
  }
  if (!done)
    hipLaunchKernelGGL(HIP_KERNEL_NAME(mega_ovf_kernel<T, TA, false>), dim3(g), dim3(64),
                       (size_t) lo.total, stream, a);
  return hipGetLastError() == hipSuccess;
}

// row-major copy of the neighbor list: out[g*mn + jj] = neighbor jj of atom g (rows 0..nrow-1)
__global__ void __launch_bounds__(256)
    mega_transpose_kernel(int *__restrict__ out, const int *__restrict__ nb, const int *__restrict__ numneigh,
                          const int ns0, const int ns1, const int mn, const int nrow)
{
  const long long n = (long long) nrow * mn;
  for (long long idx = (long long) blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += (long long) gridDim.x * blockDim.x) {
    const int g = (int) (idx / mn), jj = (int) (idx - (long long) g * mn);
    out[idx] = (jj < numneigh[g]) ? nb[(long long) g * ns0 + (long long) jj * ns1] : 0;
  }
}
inline bool mega_transpose(hipStream_t stream, int *out, const int *nb, const int *numneigh, int ns0,
                           int ns1, int mn, int nrow)
{
  const long long n = (long long) nrow * mn;
  const int nb2 = (int) std::min<long long>((n + 255) / 256, 65536);
  if (nb2 <= 0) return true;
  hipLaunchKernelGGL(mega_transpose_kernel, dim3(nb2), dim3(256), 0, stream, out, nb, numneigh, ns0, ns1,
                     mn, nrow);
  return hipGetLastError() == hipSuccess;
}

// runs the mega kernel on all atoms (query != nullptr: no launch, see mega_launch)
template <typename T, typename TA>
bool mega_run(hipStream_t stream, const MegaArgs<T, TA> &a, int *query = nullptr)
{
  return mega_launch<T, TA, 0, MS_N - 1>(stream, a, query);
}

}    // namespace PODHIP
}    // namespace LAMMPS_NS

#endif    // KOKKOS_ENABLE_HIP
#endif
