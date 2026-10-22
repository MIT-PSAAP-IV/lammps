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

// Native HIP versions of the most expensive pair pod/kk kernels.
//
// The kernels work directly on the data of the existing Kokkos views and are
// launched on the stream of the Kokkos HIP execution space instance, so they
// are ordered with the surrounding Kokkos kernels without extra fences.  Each
// launcher returns false when a problem size is outside what the kernel
// supports; the caller then runs the portable Kokkos version instead.

#ifndef LMP_PAIR_POD_KOKKOS_HIP_H
#define LMP_PAIR_POD_KOKKOS_HIP_H

#include "lmptype.h"

#include <Kokkos_Core.hpp>

#ifdef KOKKOS_ENABLE_HIP

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstring>

namespace LAMMPS_NS {
namespace PODHIP {

static constexpr int MAX_ELEMENTS = 6;      // same limit as the Kokkos kernels
static constexpr int MAX_RADIAL = 16;       // radial functions kept in registers
static constexpr int MAX_THREADS = 1024;    // HIP block size limit
static constexpr int LDS_BUDGET = 32768;    // bytes of LDS used for neighbor tiles

// launch helper: returns false if the launch failed; nothing to launch is success
template <typename... P, typename... A>
inline bool launch(void (*kernel)(P...), int nblocks, int nthreads, size_t lds, hipStream_t stream,
                   A... args)
{
  if (nblocks <= 0) return true;
  hipLaunchKernelGGL(kernel, dim3(nblocks), dim3(nthreads), lds, stream, static_cast<P>(args)...);
  return hipGetLastError() == hipSuccess;
}

inline int nblocks_for(long n, int nthreads)
{
  return static_cast<int>((n + nthreads - 1) / nthreads);
}

// geometry of the pairs (i, j) of a block: distance r and unit vector (u, v, w) = (rj - ri)/r,
// stored as rhat[j + N*d], d = 0, 1, 2
// the pairs of atom i are start .. start+nj-1: with ps == 0 numij holds the prefix sums of
// the pair counts (start = numij[i]); with ps > 0 the pairs of atom i start at i*ps and
// numij[i + 1] is their number (fixed-stride list of neighbor_fixed_kernel)
template <typename T> struct PairGeom {
  const T *rr, *rhat;
  int N;
  int ps;
  __device__ T r(const int j) const { return rr[j]; }
  __device__ void range(const int *numij, const int i, int &start, int &nj) const
  {
    if (ps > 0) {
      start = i * ps;
      nj = numij[i + 1];
    } else {
      start = numij[i];
      nj = numij[i + 1] - start;
    }
  }
  __device__ void uvw(const int j, T &u, T &v, T &w) const
  {
    u = rhat[j];
    v = rhat[j + N];
    w = rhat[j + 2 * N];
  }
};

/* ----------------------------------------------------------------------
   three- and four-body force coefficients, fused:

   fb(e, q, m, i) = sum over the 3-body and 4-body descriptors of
                    cb(descriptor, i) * d descriptor / d sumU(e, q, m, i)

   this replaces zeroing forcecoeff followed by threebody_forcecoeff() and
   fourbody_forcecoeff().  one block per atom i: sumU and the result of atom i
   live in LDS.  the 3-body pass assigns one (m, q) item per thread; items
   write disjoint entries, so no atomics are needed.  the 4-body pass assigns
   one (m, q) item per thread as well, but its three updates go to arbitrary
   entries of the m-row and use LDS atomics.  forcecoeff34_atom() is also
   called by the matrix core force kernel, which then needs no forcecoeff array.
------------------------------------------------------------------------- */

template <typename T> struct FcArgs {
  const T *cb3, *cb4, *sumU;    // sumU == nullptr: nothing to compute
  const int *pc3, *pn3, *p3_active, *pa4, *pb4, *pc4;
  int nelements, nrbf3, nabf3_active, nrbf4, nabf4, K3, Q4, Ni, do4;
  int ca, cm;    // coefficient k of atom i: cb3[i*ca + cm*k] (likewise cb4)
  const int *qa, *qp;    // or nullptr: tables of the active index a of each 3-body term q
                         // (or -1) and of the 4-body channel p of each 4-body term q
};

// force coefficients of atom i into f_s, with u_s (nelements * K3 * nrbf3 values), qa_s (K3)
// and qp_s (Q4) as scratch, all in LDS; called by all threads of the block, ends synchronized
template <typename T>
__device__ inline void forcecoeff34_atom(const FcArgs<T> &g, T *f_s, T *u_s, int *qa_s,
                                         int *qp_s, const int i)
{
  const int nelements = g.nelements, nrbf3 = g.nrbf3, nabf3_active = g.nabf3_active;
  const int K3 = g.K3, Q4 = g.Q4, ca = g.ca, cm = g.cm;
  const int S = nelements * K3 * nrbf3;
  const int base = S * i;
  const int NeK3 = nelements * K3;

  for (int t = threadIdx.x; t < S; t += blockDim.x) {
    u_s[t] = g.sumU[base + t];
    f_s[t] = T(0);
  }
  // the index tables: precomputed (g.qa), or built here
  const int *qa = g.qa ? g.qa : qa_s;
  const int *qp = g.qp ? g.qp : qp_s;
  if (!g.qa) {
    for (int q = threadIdx.x; q < K3; q += blockDim.x) qa_s[q] = -1;
    __syncthreads();
    for (int a = threadIdx.x; a < nabf3_active; a += blockDim.x) {
      const int p = g.p3_active[a];
      for (int q = g.pn3[p]; q < g.pn3[p + 1]; q++) qa_s[q] = a;
    }
    if (g.do4)
      for (int p = threadIdx.x; p < g.nabf4; p += blockDim.x)
        for (int q = g.pa4[p]; q < g.pa4[p + 1]; q++) qp_s[q] = p;
  }
  __syncthreads();

  // three-body: cb3 index i*ca + cm*(a + nabf3_active*m) + N3*(element pair)

  const int N3 = cm * nabf3_active * nrbf3;
  for (int item = threadIdx.x; item < nrbf3 * K3; item += blockDim.x) {
    const int q = item % K3;
    const int m = item / K3;
    const int a = qa[q];
    if (a < 0) continue;
    const T pk = static_cast<T>(g.pc3[q]);
    const int iu = NeK3 * m + nelements * q;
    int em = i * ca + cm * (a + nabf3_active * m);
    for (int i1 = 0; i1 < nelements; i1++) {
      const T u1 = u_s[iu + i1];
      for (int i2 = i1; i2 < nelements; i2++, em += N3) {
        const T w = pk * g.cb3[em];
        f_s[iu + i2] += w * u1;
        f_s[iu + i1] += w * u_s[iu + i2];
      }
    }
  }
  __syncthreads();

  // four-body: cb4 index i*ca + cm*(p + nabf4*m) + N4*(element triple)

  if (g.do4) {
    const int N4 = cm * g.nabf4 * g.nrbf4;
    for (int item = threadIdx.x; item < g.nrbf4 * Q4; item += blockDim.x) {
      const int q = item % Q4;
      const int m = item / Q4;
      const int p = qp[q];
      const T c = static_cast<T>(g.pc4[q]);
      const int iu = NeK3 * m;
      const int idx1 = iu + nelements * g.pb4[q];
      const int idx2 = iu + nelements * g.pb4[q + Q4];
      const int idx3 = iu + nelements * g.pb4[q + 2 * Q4];
      int k = i * ca + cm * (p + g.nabf4 * m);
      for (int i1 = 0; i1 < nelements; i1++) {
        const T c1 = u_s[idx1 + i1];
        for (int i2 = i1; i2 < nelements; i2++) {
          const T c2 = u_s[idx2 + i2];
          for (int i3 = i2; i3 < nelements; i3++, k += N4) {
            const T c3 = u_s[idx3 + i3];
            const T c4 = c * g.cb4[k];
            atomicAdd(&f_s[idx3 + i3], c4 * (c1 * c2));
            atomicAdd(&f_s[idx2 + i2], c4 * (c1 * c3));
            atomicAdd(&f_s[idx1 + i1], c4 * (c2 * c3));
          }
        }
      }
    }
    __syncthreads();
  }
}

// LDS of forcecoeff34_atom() beyond the result: u_s, qa_s and qp_s
template <typename T> inline size_t forcecoeff34_scratch(const FcArgs<T> &g)
{
  return (size_t) g.nelements * g.K3 * g.nrbf3 * sizeof(T) +
      ((size_t) g.K3 + (g.do4 ? g.Q4 : 0)) * sizeof(int);
}

template <typename T> __global__ void forcecoeff34_kernel(T *__restrict__ fb, const FcArgs<T> g)
{
  extern __shared__ char lds_raw[];
  const int S = g.nelements * g.K3 * g.nrbf3;
  T *u_s = reinterpret_cast<T *>(lds_raw);    // sumU of atom i: [m][q][e]
  T *f_s = u_s + S;                           // result of atom i, same layout
  int *qa_s = reinterpret_cast<int *>(f_s + S);    // active angular index a of q, or -1
  int *qp_s = qa_s + g.K3;                         // 4-body angular index p of each term q

  const int i = blockIdx.x;
  forcecoeff34_atom<T>(g, f_s, u_s, qa_s, qp_s, i);
  for (int t = threadIdx.x; t < S; t += blockDim.x) fb[S * i + t] = f_s[t];
}

template <typename T> bool forcecoeff34(hipStream_t stream, T *fb, const FcArgs<T> &g)
{
  if ((g.Ni <= 0) || (g.nelements > MAX_ELEMENTS)) return false;
  const size_t lds = (size_t) g.nelements * g.K3 * g.nrbf3 * sizeof(T) + forcecoeff34_scratch(g);
  if (lds > 65536) return false;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(forcecoeff34_kernel<T>), dim3(g.Ni), dim3(128), lds, stream,
                     fb, g);
  return hipGetLastError() == hipSuccess;
}


/* ----------------------------------------------------------------------
   kernels that compute the angular basis on the fly instead of reading it
   from memory.  the angular functions of pair j are the monomials u^a v^b w^c
   (a+b+c <= L) of its unit vector (u,v,w).
------------------------------------------------------------------------- */

// matrix core (MFMA) support: one 16x16x4 product per instruction, wave64.
// A (16x4) lane l holds A[l%16][l/16], B (4x16) lane l holds B[l/16][l%16],
// and the 16x16 result in lane l, element v is at row Mfma<TA>::row(l, v),
// column l%16.  the layouts were verified on gfx942.

#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || \
     defined(__gfx942__) || defined(__gfx950__))
#define LMP_POD_HIP_MFMA 1
#else
#define LMP_POD_HIP_MFMA 0
#endif

template <typename TA> struct Mfma;

template <> struct Mfma<double> {
  typedef double acc_t __attribute__((ext_vector_type(4)));
  static __device__ inline acc_t mma(double a, double b, acc_t c)
  {
#if LMP_POD_HIP_MFMA
    return __builtin_amdgcn_mfma_f64_16x16x4f64(a, b, c, 0, 0, 0);
#else
    (void) a;
    (void) b;
    return c;
#endif
  }
  static __device__ inline int row(int lane, int v) { return (lane >> 4) + 4 * v; }
};

template <> struct Mfma<float> {
  typedef float acc_t __attribute__((ext_vector_type(4)));
  static __device__ inline acc_t mma(float a, float b, acc_t c)
  {
#if LMP_POD_HIP_MFMA
    return __builtin_amdgcn_mfma_f32_16x16x4f32(a, b, c, 0, 0, 0);
#else
    (void) a;
    (void) b;
    return c;
#endif
  }
  static __device__ inline int row(int lane, int v) { return 4 * (lane >> 4) + v; }
};

// true if the current device has matrix cores usable by the kernels above
inline bool device_has_mfma()
{
  static int has = -1;
  if (has < 0) {
    has = 0;
    int dev = 0;
    hipDeviceProp_t prop;
    if ((hipGetDevice(&dev) == hipSuccess) && (hipGetDeviceProperties(&prop, dev) == hipSuccess)) {
      const char *a = prop.gcnArchName;
      if ((strncmp(a, "gfx90a", 6) == 0) || (strncmp(a, "gfx94", 5) == 0) ||
          (strncmp(a, "gfx95", 5) == 0))
        has = 1;
    }
  }
  return has == 1;
}

/* ----------------------------------------------------------------------
   radial basis followed by the projection with Phi, fused:

   rbf(j, n) = sum_s rbft(j, s) Phi(s, n)   (and the same for drbf)

   one thread per pair; the ns raw radial functions are consumed as they are
   generated instead of being written to and read back from memory.  the
   arithmetic and the summation order are the same as in radialbasis() and
   radialPhi().  at most MAX_RADIAL outputs are kept in registers: with more,
   blockIdx.y selects a group of at most MAX_RADIAL outputs and each group
   regenerates the raw functions.
------------------------------------------------------------------------- */

template <typename T, typename TA>
__global__ void __launch_bounds__(256) radialbasis_phi_kernel(T *__restrict__ rbf, T *__restrict__ drbf,
                                       const PairGeom<T> pg, const T *__restrict__ rin,
                                       const T *__restrict__ invrdiff, const int rs0,
                                       const int rs1, const int ds0, const int ds1,
                                       const T *__restrict__ neg_alpha,
                                       const T *__restrict__ pi_inv_t1,
                                       const T *__restrict__ dx_factor, const T *__restrict__ Phi,
                                       const int *__restrict__ ti, const int *__restrict__ tj,
                                       const int nelements, const int besseldegree,
                                       const int inversedegree, const int nbesselpars,
                                       const int ns, const int nrbfmax, const int Nij)
{
  extern __shared__ char lds_raw[];
  T *phi_s = reinterpret_cast<T *>(lds_raw);    // Phi(s, q0 + q) at s + ns*q
  const int nq = (nrbfmax + gridDim.y - 1) / gridDim.y;
  const int q0 = blockIdx.y * nq;
  const int nr = min(nq, nrbfmax - q0);
  for (int t = threadIdx.x; t < ns * nr; t += blockDim.x) phi_s[t] = Phi[t + ns * q0];
  __syncthreads();

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= Nij) return;

  const T E_V = static_cast<T>(2.718281828459045235360287);
  const int itype = ti[n];
  const int jtype = tj[n];
  const int bc_base = (itype * nelements + jtype) * nbesselpars;

  const T dij = pg.r(n);
  const T invdij = static_cast<T>(1.0) / dij;

  const T invrmax = invrdiff[itype * ds0 + jtype * ds1];
  const T r = dij - rin[itype * rs0 + jtype * rs1];
  const T invr = static_cast<T>(1.0) / r;
  const T y = r * invrmax;

  // cutoff function, as in PairPODKokkos::cutoff_exp()
  const T y2 = y * y;
  const T y3 = y * y2 - static_cast<T>(1.0);
  const T y4 = y3 * y3 + static_cast<T>(1.0e-6);
  const T y5 = sqrt(y4);
  const T fcut = E_V * exp(static_cast<T>(-1.0) / y5);
  const T dfcut = fcut * (static_cast<T>(3.0) * invrmax * y2 * y3) / (y4 * y5);

  const T f1 = fcut * invr;
  const T g1 = (dfcut - f1) * invr;
  const T bfac = sqrt(static_cast<T>(2.0) * invrmax);
  const T bf1 = bfac * f1;
  const T bg1 = bfac * g1;

  TA sum1[MAX_RADIAL], sum2[MAX_RADIAL];
#pragma unroll
  for (int q = 0; q < MAX_RADIAL; q++) sum1[q] = sum2[q] = TA(0);

  int s = 0;
  auto add = [&](const T rv, const T dv) {
#pragma unroll
    for (int q = 0; q < MAX_RADIAL; q++)
      if (q < nr) {
        const T phi = phi_s[s + ns * q];
        sum1[q] += static_cast<TA>(rv * phi);
        sum2[q] += static_cast<TA>(dv * phi);
      }
    s++;
  };

  // Bessel radial basis
  for (int j = 0; j < nbesselpars; ++j) {
    const int bcj = bc_base + j;
    const T mt2 = expm1(neg_alpha[bcj] * y);
    const T xpi = mt2 * pi_inv_t1[bcj];
    const T pi_dx = dx_factor[bcj] * (static_cast<T>(1.0) + mt2);
    const T Kf1dx = bf1 * pi_dx;

    TA ix = static_cast<TA>(xpi);
    for (int i = 1; i <= besseldegree; ++i) {
      const T cosax = static_cast<T>(cos(ix));
      const T inv_i = static_cast<T>(1.0) / static_cast<T>(i);
      const T isinax = inv_i * static_cast<T>(sin(ix));
      const T drbftdr = bg1 * isinax + Kf1dx * cosax;
      add(bf1 * isinax, drbftdr * invdij);
      ix += static_cast<TA>(xpi);
    }
  }

  // inverse polynomial radial basis
  const T fcut_invd = fcut * invdij;
  T dterm = dfcut;
  T inva = static_cast<T>(1.0);
  for (int i = 1; i <= inversedegree; ++i) {
    dterm -= fcut_invd;
    inva *= invdij;
    add(fcut * inva, dterm * inva * invdij);
  }

#pragma unroll
  for (int q = 0; q < MAX_RADIAL; q++)
    if (q < nr) {
      rbf[n + Nij * (q0 + q)] = static_cast<T>(sum1[q]);
      drbf[n + Nij * (q0 + q)] = static_cast<T>(sum2[q]);
    }
}

template <typename T, typename TA>
bool radialbasis_phi(hipStream_t stream, T *rbf, T *drbf, const PairGeom<T> &pg, const T *rin,
                     const T *invrdiff, int rs0, int rs1, int ds0, int ds1, const T *neg_alpha,
                     const T *pi_inv_t1, const T *dx_factor, const T *Phi, const int *ti,
                     const int *tj, int nelements, int besseldegree, int inversedegree,
                     int nbesselpars, int ns, int nrbfmax, int Nij)
{
  if ((Nij <= 0) || (nrbfmax <= 0)) return false;
  if (ns != nbesselpars * besseldegree + inversedegree) return false;
  // groups of at most MAX_RADIAL outputs, of equal size
  const int ng = (nrbfmax + MAX_RADIAL - 1) / MAX_RADIAL;
  const int nq = (nrbfmax + ng - 1) / ng;
  const size_t lds = (size_t) ns * nq * sizeof(T);
  if (lds > 65536) return false;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(radialbasis_phi_kernel<T, TA>),
                     dim3(nblocks_for(Nij, 256), ng), dim3(256), lds, stream, rbf, drbf, pg, rin, invrdiff, rs0, rs1, ds0, ds1,
                     neg_alpha, pi_inv_t1, dx_factor, Phi, ti, tj, nelements, besseldegree,
                     inversedegree, nbesselpars, ns, nrbfmax, Nij);
  return hipGetLastError() == hipSuccess;
}

/* ----------------------------------------------------------------------
   sumU(e, k, m, i) = sum_j [tj(j) == e] rbf(j, m) * abf(j, k)  with abf
   generated on the fly.

   one block per atom i.  a tile of neighbors is staged in LDS: one thread
   per pair computes its angular functions with the same recursion as
   angularbasis() (so abf is bit-identical), stored [k][j]; then one thread
   per (k, m) accumulates the sums.
------------------------------------------------------------------------- */

static constexpr int MFMA_MAX_RT = 4;    // K3 <= 64 (angular degree <= 5)

// sizes of the Hf benchmark model (one element), compile-time constants in the HF instances of
// the matrix core and local EA-POD kernels; the host checks that the model has them.  the
// number of many-body radial functions stays a run-time value (as a constant it costs
// registers)
struct PodHf {
  static constexpr int K3 = 20, L = 3, nrbf2 = 40, nDes = 66, nl2 = 40, nl3 = 20, nl4 = 6;
  static constexpr int nCls = 50, nActMax = 3, nCoeff = 3301, nabf3_active = 4, nrbf4 = 3;
  static constexpr int nabf4 = 2, Q4 = 4, jtile = 16;
};
static constexpr int MFMA_MAX_CT = 2;    // nelements * nrbf3 <= 32

/* ----------------------------------------------------------------------
   source of the many-body radial functions of a pair: the rbf/drbf arrays,
   or, with coef != nullptr, the radial Hermite spline table evaluated on the
   fly with the same operations as radialbasis_hermite_kernel, so that the
   consumers need not read rbf/drbf from memory
------------------------------------------------------------------------- */

template <typename T> struct RadialSrc {
  const T *rbf, *drbf;             // [j + Nij*m], used when coef == nullptr
  const T *coef, *r0v, *invdrv;    // Hermite table [pair][bin][nr][4], bin origin and 1/width
  const int *ti, *tj;
  int ne, nb, nr, Nij;
};

// calls emit(m, rbf(j, m), drbf(j, m)) for m < nrbf3; MAXR > 0 unrolls the
// loop to MAXR (nrbf3 <= MAXR) so that emit may index register arrays
// as pair_radial() below, with the distance r of pair j given (used with the Hermite table)
template <int MAXR, typename T, typename F>
__device__ inline void pair_radial_r(const RadialSrc<T> &s, const T r, const int j,
                                     const int nrbf3, F &&emit)
{
  constexpr int MR = (MAXR > 0) ? MAXR : 1;
  if (s.coef == nullptr) {
    if (MAXR > 0) {
#pragma unroll
      for (int m = 0; m < MR; m++)
        if (m < nrbf3) emit(m, s.rbf[j + s.Nij * m], s.drbf ? s.drbf[j + s.Nij * m] : T(0));
    } else {
      for (int m = 0; m < nrbf3; m++)
        emit(m, s.rbf[j + s.Nij * m], s.drbf ? s.drbf[j + s.Nij * m] : T(0));
    }
    return;
  }
  const int pair = (s.ne > 1) ? s.ti[j] * s.ne + s.tj[j] : 0;
  const T r0 = s.r0v[pair];
  const T invdr = s.invdrv[pair];
  const T invdrr = invdr / r;
  const T tg = (r - r0) * invdr;
  int b = (int) tg;
  if (b < 0) b = 0;
  else if (b > s.nb - 1) b = s.nb - 1;
  const T t = tg - (T) b;
  const T *cb = s.coef + ((size_t) pair * (size_t) s.nb + (size_t) b) * (size_t) s.nr * 4ull;
  auto one = [&](const int m) {
    const T c0 = cb[4 * m + 0], c1 = cb[4 * m + 1], c2 = cb[4 * m + 2], c3 = cb[4 * m + 3];
    const T f = c0 + t * (c1 + t * (c2 + t * c3));
    const T dfdt = c1 + t * (static_cast<T>(2.0) * c2 + static_cast<T>(3.0) * c3 * t);
    emit(m, f, dfdt * invdrr);
  };
  if (MAXR > 0) {
#pragma unroll
    for (int m = 0; m < MR; m++)
      if (m < nrbf3) one(m);
  } else {
    for (int m = 0; m < nrbf3; m++) one(m);
  }
}

template <int MAXR, typename T, typename F>
__device__ inline void pair_radial(const RadialSrc<T> &s, const PairGeom<T> &pg, const int j,
                                   const int nrbf3, F &&emit)
{
  pair_radial_r<MAXR>(s, s.coef ? pg.r(j) : T(0), j, nrbf3, emit);
}

// the radial functions of one pair, evaluated one m at a time: the Hermite bin of the
// pair is found once, or the stored rbf/drbf are read
template <typename T> struct PairRadial {
  const T *c;    // Hermite coefficients of the bin, [m][4]; nullptr: stored rbf/drbf
  T t, s;        // position in the bin and d(t)/dr / r
  const T *rbf, *drbf;
  int j, Nij;

  __device__ PairRadial(const RadialSrc<T> &rs, const PairGeom<T> &pg, const int jj) :
      c(nullptr), t(T(0)), s(T(0)), rbf(rs.rbf), drbf(rs.drbf), j(jj), Nij(rs.Nij)
  {
    if (rs.coef == nullptr) return;
    const int pair = (rs.ne > 1) ? rs.ti[j] * rs.ne + rs.tj[j] : 0;
    const T r = pg.r(j);
    const T invdr = rs.invdrv[pair];
    const T tg = (r - rs.r0v[pair]) * invdr;
    int b = (int) tg;
    if (b < 0) b = 0;
    else if (b > rs.nb - 1) b = rs.nb - 1;
    t = tg - (T) b;
    s = invdr / r;
    c = rs.coef + ((size_t) pair * (size_t) rs.nb + (size_t) b) * (size_t) rs.nr * 4ull;
  }

  __device__ void operator()(const int m, T &f, T &df) const
  {
    if (c == nullptr) {
      f = rbf[j + Nij * m];
      df = drbf ? drbf[j + Nij * m] : T(0);
      return;
    }
    const T *cm = c + 4 * m;
    f = cm[0] + t * (cm[1] + t * (cm[2] + t * cm[3]));
    df = (cm[1] + t * (static_cast<T>(2.0) * cm[2] + static_cast<T>(3.0) * cm[3] * t)) * s;
  }
};

template <typename T, typename TA>
__global__ void radialangularsum_otf_kernel(T *__restrict__ sumU, const RadialSrc<T> rs,
                                            const PairGeom<T> pg,
                                            const int *__restrict__ tj,
                                            const int *__restrict__ numij,
                                            const int *__restrict__ pq_m,
                                            const int *__restrict__ pq_d, const int nelements,
                                            const int nrbf3, const int K3, const int Nij,
                                            const int jtile)
{
  extern __shared__ char lds_raw[];
  const int ld = jtile + 1;                     // padded row length, avoids bank conflicts
  T *abf_s = reinterpret_cast<T *>(lds_raw);    // [K3][ld]
  T *rbf_s = abf_s + K3 * ld;                   // [nrbf3][ld]
  int *tj_s = reinterpret_cast<int *>(rbf_s + nrbf3 * ld);    // [jtile]
  int *pqm_s = tj_s + jtile;                                  // [K3]
  int *pqd_s = pqm_s + K3;                                    // [K3]

  const int i = blockIdx.x;
  int start, nj;
  pg.range(numij, i, start, nj);
  const int nout = nrbf3 * K3;

  for (int t = threadIdx.x; t < K3; t += blockDim.x) {
    pqm_s[t] = pq_m[t];
    pqd_s[t] = pq_d[t];
  }

  TA acc[MAX_ELEMENTS];
  for (int e = 0; e < MAX_ELEMENTS; e++) acc[e] = TA(0);
  const int o = threadIdx.x;    // output index o = k + K3*m
  const int ko = o % K3;
  const int mo = o / K3;

  for (int j0 = 0; j0 < nj; j0 += jtile) {
    const int nt = min(jtile, nj - j0);
    __syncthreads();

    for (int jj = threadIdx.x; jj < nt; jj += blockDim.x) {
      const int j = start + j0 + jj;
      T u, v, w;
      pg.uvw(j, u, v, w);
      abf_s[jj] = static_cast<T>(1.0);
      for (int k = 1; k < K3; k++) {
        const int d = pqd_s[k];
        const T c = (d == 1) ? u : ((d == 2) ? v : w);
        abf_s[k * ld + jj] = abf_s[pqm_s[k] * ld + jj] * c;
      }
      pair_radial<0>(rs, pg, j, nrbf3, [&](int m, T f, T) { rbf_s[m * ld + jj] = f; });
      tj_s[jj] = (nelements > 1) ? tj[j] : 0;
    }
    __syncthreads();

    if (o < nout) {
      const T *ak = abf_s + ko * ld;
      const T *rm = rbf_s + mo * ld;
      if (nelements == 1) {
        for (int jj = 0; jj < nt; jj++) acc[0] += static_cast<TA>(rm[jj] * ak[jj]);
      } else {
        for (int jj = 0; jj < nt; jj++) acc[tj_s[jj]] += static_cast<TA>(rm[jj] * ak[jj]);
      }
    }
  }

  if (o < nout) {
    const int base = nelements * (o + nout * i);
    for (int e = 0; e < nelements; e++) sumU[base + e] = static_cast<T>(acc[e]);
  }
}

/* ----------------------------------------------------------------------
   3- and 4-body descriptors of one atom, computed by the calling block
------------------------------------------------------------------------- */

template <typename T> struct DescArgs {
  T *d3, *d4;    // d3 == nullptr: no descriptors to compute
  const int *pc3, *pn3, *p3_active, *pa4, *pb4, *pc4;
  const int2 *qrange3;    // (pn3[p3_active[a]], pn3[p3_active[a] + 1]) for each a
  int nelements, nrbf3, nabf3_active, nrbf4, nabf4, K3, Q4, Ni, do4;
  int split;    // 1: a descriptor may be summed by up to 8 threads (desc34_atom)
};

// threads per descriptor in desc34_atom(): the power of 2 (<= 8) that spreads n items over the
// block, or 1
__device__ inline int desc34_lpi(const int split, const int n)
{
  if (!split || (n <= 0)) return 1;
  int l = 1;
  while ((l < 8) && (2 * l * n <= (int) blockDim.x)) l *= 2;
  return l;
}

// 3- and 4-body descriptors of atom i from its sumU block u_s (in LDS),
// computed by the threads of the calling block
template <typename T, typename TA>
__device__ inline void desc34_atom(const DescArgs<T> &g, const T *u_s, const int i)
{
  const int nelements = g.nelements, Ni = g.Ni;
  const int NeK3 = nelements * g.K3;

  // three-body: d3 index i + Ni*(a + nabf3_active*m) + N3*(element pair).  lpi threads per
  // descriptor sum every lpi-th term and combine their partial sums
  const int N3 = Ni * g.nabf3_active * g.nrbf3;
  const int lpi3 = desc34_lpi(g.split, g.nrbf3 * g.nabf3_active);
  const int sub3 = threadIdx.x % lpi3;
  for (int item = threadIdx.x / lpi3; item < g.nrbf3 * g.nabf3_active;
       item += blockDim.x / lpi3) {
    const int a = item % g.nabf3_active;
    const int m = item / g.nabf3_active;
    const int2 qr = g.qrange3[a];    // = pn3[p3_active[a]], pn3[p3_active[a] + 1]
    const int n1 = qr.x, n2 = qr.y;
    const T *u = u_s + NeK3 * m;
    int k = i + Ni * (a + g.nabf3_active * m);
    for (int i1 = 0; i1 < nelements; i1++)
      for (int i2 = i1; i2 < nelements; i2++, k += N3) {
        TA tmp = TA(0);
        for (int q = n1 + sub3; q < n2; q += lpi3)
          tmp += static_cast<TA>(static_cast<T>(g.pc3[q]) * u[i1 + nelements * q] *
                                 u[i2 + nelements * q]);
        for (int o = 1; o < lpi3; o <<= 1) tmp += __shfl_xor(tmp, o);
        if (sub3 == 0) g.d3[k] = static_cast<T>(tmp);
      }
  }

  // four-body: d4 index i + Ni*(p + nabf4*m) + N4*(element triple)
  if (g.do4) {
    const int N4 = Ni * g.nabf4 * g.nrbf4;
    const int lpi4 = desc34_lpi(g.split, g.nrbf4 * g.nabf4);
    const int sub4 = threadIdx.x % lpi4;
    for (int item = threadIdx.x / lpi4; item < g.nrbf4 * g.nabf4; item += blockDim.x / lpi4) {
      const int p = item % g.nabf4;
      const int m = item / g.nabf4;
      const int n1 = g.pa4[p], n2 = g.pa4[p + 1];
      const T *u = u_s + NeK3 * m;
      int k = i + Ni * (p + g.nabf4 * m);
      for (int i1 = 0; i1 < nelements; i1++)
        for (int i2 = i1; i2 < nelements; i2++)
          for (int i3 = i2; i3 < nelements; i3++, k += N4) {
            TA tmp = TA(0);
            for (int q = n1 + sub4; q < n2; q += lpi4) {
              const int c = g.pc4[q];
              const int j1 = g.pb4[q], j2 = g.pb4[q + g.Q4], j3 = g.pb4[q + 2 * g.Q4];
              tmp += static_cast<TA>(static_cast<T>(c) * u[i1 + nelements * j1] *
                                     u[i2 + nelements * j2] * u[i3 + nelements * j3]);
            }
            for (int o = 1; o < lpi4; o <<= 1) tmp += __shfl_xor(tmp, o);
            if (sub4 == 0) g.d4[k] = static_cast<T>(tmp);
          }
    }
  }
}

/* ----------------------------------------------------------------------
   matrix core version of radialangularsum_otf_kernel: one wavefront per
   atom computes the sums as a GEMM on the matrix cores,

     U (K3 x Ne*nrbf3) = A (K3 x nj) * B (nj x Ne*nrbf3),

   with A(k, j) = abf(j, k) and B(j, m + nrbf3*e) = [tj(j) == e] rbf(j, m),
   in 16x16 output tiles padded with zeros.  LDS holds only the powers u^n,
   v^n, w^n (n <= L) and the radial functions of a tile of neighbors; each
   lane builds its own element of A, abf(j, k) = u^a v^b w^c, from the
   powers.
------------------------------------------------------------------------- */

// packed exponents a | b << 8 | c << 16 of the monomials k < K3 into ex_s, from the pq recursion
template <typename T>
__device__ inline void pod_exponents(int *ex_s, const int *pq_m, const int *pq_d, const int K3)
{
  if (threadIdx.x == 0) {
    ex_s[0] = 0;
    for (int k = 1; k < K3; k++) ex_s[k] = ex_s[pq_m[k]] + (1 << (8 * (pq_d[k] - 1)));
  }
}

// the same, from the table pq_ex built on the host, copied by all threads
__device__ inline void pod_exponents_table(int *ex_s, const int *pq_ex, const int K3)
{
  for (int k = threadIdx.x; k < K3; k += blockDim.x) ex_s[k] = pq_ex[k];
}

// RTM x CTM output tiles (RTM >= ceil(K3/16), CTM >= ceil(Ne*nrbf3/16)) are kept in registers
// PF 1: the geometry of the next tile (one of u, v, w or r of a pair per lane, jtile <= 16) is
// loaded while the current tile is on the matrix cores, and all lanes fill the tile.
// PF 2: the tile is filled by all lanes, one (pair, radial function) or (pair, direction) each
// HF: the sizes of the Hf model (PodHf) as constants
template <typename T, typename TA, int RTM, int CTM, int PF, bool HF>
__global__ void __launch_bounds__(64)
    radialangularsum_mfma_kernel(T *__restrict__ sumU, const RadialSrc<T> rs_in,
                                 const PairGeom<T> pg, const int *__restrict__ tj,
                                 const int *__restrict__ numij, const int *__restrict__ pq_m,
                                 const int *__restrict__ pq_d, const int nelements_in,
                                 const int nrbf3, const int K3_in, const int L_in, const int Nij,
                                 const int jtile_in, const int *__restrict__ pq_ex)
{
  RadialSrc<T> rs = rs_in;
  if constexpr (HF) rs.ne = 1;
  const int nelements = HF ? 1 : nelements_in;
  const int K3 = HF ? PodHf::K3 : K3_in;
  const int L = HF ? PodHf::L : L_in;
  const int jtile = HF ? PodHf::jtile : jtile_in;
  extern __shared__ char lds_raw[];
  const int L1 = L + 1;
  T *pow_s = reinterpret_cast<T *>(lds_raw);      // [jtile][3][L1]
  const int ldr = jtile + 1;                      // padded row: functions m of a pair in different banks
  T *rbf_s = pow_s + jtile * 3 * L1;              // [nrbf3][ldr]
  int *tj_s = reinterpret_cast<int *>(rbf_s + nrbf3 * ldr);    // [jtile]
  int *ex_s = tj_s + jtile;                                      // [K3] packed a | b<<8 | c<<16

  const int i = blockIdx.x;
  int start, nj;
  pg.range(numij, i, start, nj);
  const int nout = nrbf3 * K3;
  const int NC = nelements * nrbf3;
  const int lane = threadIdx.x;
  const int lr = lane & 15, lq = lane >> 4;
  const int RT = (K3 + 15) / 16, CT = (NC + 15) / 16;

  pod_exponents_table(ex_s, pq_ex, K3);
  __syncthreads();

  // exponents of the A rows owned by this lane (offsets into pow_s), and the
  // (m, e) of the B columns owned by this lane
  int oa[RTM], ob[RTM], oc[RTM];
#pragma unroll
  for (int rt = 0; rt < RTM; rt++) {
    const int k = rt * 16 + lr;
    const int ex = (k < K3) ? ex_s[k] : 0;
    oa[rt] = ex & 255;
    ob[rt] = L1 + ((ex >> 8) & 255);
    oc[rt] = 2 * L1 + ((ex >> 16) & 255);
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

  T xh = T(0);
  int tjh = 0;
  const int pjj = lane % jtile, pd = lane / jtile;    // PF: pair and component of this lane
  auto fetch = [&](const int jbase) {
    if ((pd < 4) && (jbase + pjj < nj)) {
      const int j = start + jbase + pjj;
      if (pd < 3) xh = pg.rhat[j + pd * pg.N];
      else {
        xh = pg.r(j);
        tjh = (nelements > 1) ? tj[j] : 0;
      }
    }
  };
  if constexpr (PF == 1) fetch(0);

  for (int j0 = 0; j0 < nj; j0 += jtile) {
    const int nt = min(jtile, nj - j0);
    const int nt4 = (nt + 3) & ~3;
    __syncthreads();
    if constexpr (PF == 2) {
      const int nr = nt4 * nrbf3;
      for (int t = lane; t < nt4 * (nrbf3 + 3); t += 64) {
        if (t < nr) {
          const int jj = t / nrbf3, m = t - jj * nrbf3;
          T f = T(0), df;
          if (jj < nt) {
            const PairRadial<T> prad(rs, pg, start + j0 + jj);
            prad(m, f, df);
          }
          rbf_s[m * ldr + jj] = f;
          if (m == 0) tj_s[jj] = (jj < nt) ? ((nelements > 1) ? tj[start + j0 + jj] : 0) : -1;
        } else {
          const int q = t - nr, jj = q / 3, d = q - 3 * jj;
          T *pw = pow_s + jj * 3 * L1 + d * L1;
          T x = T(0), p = T(0);
          if (jj < nt) {
            x = pg.rhat[start + j0 + jj + d * pg.N];
            p = static_cast<T>(1.0);
          }
          for (int n = 0; n < L1; n++) {
            pw[n] = p;
            p *= x;
          }
        }
      }
    } else if constexpr (PF == 1) {
      if ((pd < 4) && (pjj < nt4)) {
        T *pw = pow_s + pjj * 3 * L1;
        if (pd < 3) {
          T p = (pjj < nt) ? static_cast<T>(1.0) : T(0);
          for (int n = 0; n < L1; n++) {
            pw[pd * L1 + n] = p;
            p *= xh;
          }
        } else if (pjj < nt) {
          pair_radial_r<0>(rs, xh, start + j0 + pjj, nrbf3,
                           [&](int m, T f, T) { rbf_s[m * ldr + pjj] = f; });
          tj_s[pjj] = tjh;
        } else {
          for (int m = 0; m < nrbf3; m++) rbf_s[m * ldr + pjj] = T(0);
          tj_s[pjj] = -1;
        }
      }
      if (j0 + jtile < nj) fetch(j0 + jtile);
    } else
    for (int jj = lane; jj < nt4; jj += 64) {
      T *pw = pow_s + jj * 3 * L1;
      if (jj < nt) {
        const int j = start + j0 + jj;
        T u, v, w;
        pg.uvw(j, u, v, w);
        T pu = static_cast<T>(1.0), pv = pu, pz = pu;
        for (int n = 0; n < L1; n++) {
          pw[n] = pu;
          pw[L1 + n] = pv;
          pw[2 * L1 + n] = pz;
          pu *= u;
          pv *= v;
          pz *= w;
        }
        pair_radial<0>(rs, pg, j, nrbf3, [&](int m, T f, T) { rbf_s[m * ldr + jj] = f; });
        tj_s[jj] = (nelements > 1) ? tj[j] : 0;
      } else {
        for (int n = 0; n < 3 * L1; n++) pw[n] = T(0);
        for (int m = 0; m < nrbf3; m++) rbf_s[m * ldr + jj] = T(0);
        tj_s[jj] = -1;
      }
    }
    __syncthreads();

    for (int q = 0; q < nt4; q += 4) {
      const int jj = q + lq;
      const T *pw = pow_s + jj * 3 * L1;
      const int tjj = tj_s[jj];
      TA a[RTM], b[CTM];
#pragma unroll
      for (int rt = 0; rt < RTM; rt++)
        a[rt] = (rt * 16 + lr < K3) ? static_cast<TA>(pw[oa[rt]] * pw[ob[rt]] * pw[oc[rt]])
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
            if (ct < CT) acc[rt][ct] = Mfma<TA>::mma(a[rt], b[ct], acc[rt][ct]);
    }
  }

#pragma unroll
  for (int rt = 0; rt < RTM; rt++)
#pragma unroll
    for (int ct = 0; ct < CTM; ct++)
#pragma unroll
      for (int v = 0; v < 4; v++) {
        const int k = rt * 16 + Mfma<TA>::row(lane, v);
        if ((rt < RT) && (ct < CT) && (k < K3) && (bm[ct] >= 0))
          sumU[be[ct] + nelements * (k + K3 * bm[ct] + nout * i)] = static_cast<T>(acc[rt][ct][v]);
      }
}

template <typename T, typename TA>
bool radialangularsum_otf(hipStream_t stream, T *sumU, const RadialSrc<T> &rs, const PairGeom<T> &pg,
                          const int *tj,
                          const int *numij, const int *pq_m, const int *pq_d, int nelements,
                          int nrbf3, int K3, int Ni, int Nij, bool use_mfma, int jtile,
                          bool prefetch = false, bool spread = false, const int *pq_ex = nullptr,
                          bool hf = false)
{
  // pq_ex: packed exponents of the monomials (pod_exponents()), needed by the matrix cores
  if (!pq_ex) use_mfma = false;
  const int nout = nrbf3 * K3;
  if ((Ni <= 0) || (nelements > MAX_ELEMENTS) || (K3 < 1)) return false;
  if (use_mfma) {
    if (!device_has_mfma() || (K3 > 16 * MFMA_MAX_RT) || (nelements * nrbf3 > 16 * MFMA_MAX_CT))
      use_mfma = false;
  }
  if (!use_mfma && (nout > MAX_THREADS)) return false;

  const int threads = use_mfma ? 64 : ((nout + 63) / 64) * 64;
  if (jtile > threads) jtile = threads;
  jtile &= ~3;
  if (jtile < 4) return false;
  const size_t lds = (size_t) (K3 + nrbf3) * (jtile + 1) * sizeof(T) +
      ((size_t) jtile + 2 * K3) * sizeof(int);
  if (lds > 65536) return false;

  if (use_mfma) {
    int L = 0;
    while ((L + 1) * (L + 2) * (L + 3) / 6 < K3) L++;
    if ((L + 1) * (L + 2) * (L + 3) / 6 != K3) return false;
    const size_t lds2 = ((size_t) (3 * (L + 1) + nrbf3) * jtile + nrbf3) * sizeof(T) +
        ((size_t) jtile + K3) * sizeof(int);
    const int RT = (K3 + 15) / 16, CT = (nelements * nrbf3 + 15) / 16;
    const int pf = spread ? 2 : ((prefetch && (4 * jtile <= 64)) ? 1 : 0);
#define LMP_POD_RAS_MFMA1(R, C, P, H)                                                            \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(radialangularsum_mfma_kernel<T, TA, R, C, P, H>), dim3(Ni), \
                     dim3(threads), lds2, stream, sumU, rs, pg, tj, numij, pq_m, pq_d,         \
                     nelements, nrbf3, K3, L, Nij, jtile, pq_ex)
    // the Hf instance: one element, K3 20, the spread fill with a tile of 16 pairs
    const bool hfi = hf && (nelements == 1) && (K3 == PodHf::K3) && (L == PodHf::L) &&
        (jtile == PodHf::jtile) && (pf == 2);
#define LMP_POD_RAS_MFMA(R, C)                                                                   \
  do {                                                                                           \
    if (pf == 2) LMP_POD_RAS_MFMA1(R, C, 2, false);                                              \
    else if (pf == 1) LMP_POD_RAS_MFMA1(R, C, 1, false);                                         \
    else LMP_POD_RAS_MFMA1(R, C, 0, false);                                                      \
  } while (0)
    if (hfi) LMP_POD_RAS_MFMA1(2, 1, 2, true);
    else if (CT == 1) {
      if (RT == 1) LMP_POD_RAS_MFMA(1, 1);
      else if (RT == 2) LMP_POD_RAS_MFMA(2, 1);
      else if (RT == 3) LMP_POD_RAS_MFMA(3, 1);
      else LMP_POD_RAS_MFMA(4, 1);
    } else {
      if (RT == 1) LMP_POD_RAS_MFMA(1, 2);
      else if (RT == 2) LMP_POD_RAS_MFMA(2, 2);
      else if (RT == 3) LMP_POD_RAS_MFMA(3, 2);
      else LMP_POD_RAS_MFMA(4, 2);
    }
#undef LMP_POD_RAS_MFMA
#undef LMP_POD_RAS_MFMA1
  } else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(radialangularsum_otf_kernel<T, TA>), dim3(Ni),
                       dim3(threads), lds, stream, sumU, rs, pg, tj, numij, pq_m, pq_d,
                       nelements, nrbf3, K3, Nij, jtile);
  return hipGetLastError() == hipSuccess;
}

/* ----------------------------------------------------------------------
   allbody_forces() with the angular functions and their derivatives
   generated on the fly.

   with P_k = u^a v^b w^c the angular function k, the force on pair j is

     f = SU grad(u) + SV grad(v) + SW grad(w) + SA rij,

     SU = sum_k G_k dP_k/du  (SV, SW likewise),   SA = sum_k H_k P_k,
     G_k = sum_m fc(e_j, k, m) rbf(j, m),  H_k = sum_m fc(e_j, k, m) drbf(j, m)

   which equals the sum over k of G_k * grad(P_k) + H_k * P_k * rij used by
   the Kokkos kernel.  the powers u^n, v^n, w^n (n <= L) of each pair are
   kept in LDS and P_k and its derivatives are products of three of them.
------------------------------------------------------------------------- */

// powers u^n, v^n, w^n (n = 0..L) of pair j, stored [u^n][v^n][w^n]
template <typename T>
__device__ inline void pod_pair_powers(T *pd, const PairGeom<T> &pg, const int j, const int L1)
{
  T u, v, w;
  pg.uvw(j, u, v, w);
  T pu = static_cast<T>(1.0), pv = pu, pz = pu;
  for (int n = 0; n < L1; n++) {
    pd[n] = pu;
    pd[L1 + n] = pv;
    pd[2 * L1 + n] = pz;
    pu *= u;
    pv *= v;
    pz *= w;
  }
}

// du/dx, du/dy, du/dz, dv/dy, dv/dz, dw/dz, x, y, z of pair j, with grad(u, v, w) =
// (I - rhat rhat^T) / r
template <typename T>
__device__ inline void pod_pair_gradients(T *g, const PairGeom<T> &pg, const int j)
{
  T u, v, w;
  pg.uvw(j, u, v, w);
  const T r = pg.r(j);
  const T invr = static_cast<T>(1.0) / r;
  g[0] = (v * v + w * w) * invr;
  g[1] = -(u * v) * invr;
  g[2] = -(u * w) * invr;
  g[3] = (u * u + w * w) * invr;
  g[4] = -(v * w) * invr;
  g[5] = (u * u + v * v) * invr;
  g[6] = r * u;
  g[7] = r * v;
  g[8] = r * w;
}

// P_k and its derivatives for the packed exponents ex = a | b<<8 | c<<16
template <typename T>
__device__ inline void pod_monomial(const T *pd, const int ex, const int L1, T &P, T &Pu, T &Pv,
                                    T &Pw)
{
  const int a = ex & 255, b = (ex >> 8) & 255, c = (ex >> 16) & 255;
  const T pa = pd[a], pb = pd[L1 + b], pc = pd[2 * L1 + c];
  const T pbc = pb * pc;
  P = pa * pbc;
  Pu = static_cast<T>(a) * pd[a > 0 ? a - 1 : 0] * pbc;
  Pv = static_cast<T>(b) * pa * pd[L1 + (b > 0 ? b - 1 : 0)] * pc;
  Pw = static_cast<T>(c) * pa * pb * pd[2 * L1 + (c > 0 ? c - 1 : 0)];
}

// as pod_monomial(), with the powers of pd at stride S: pd[S*n] (layout [direction][power][pair])
template <typename T, int S>
__device__ inline void pod_monomial_s(const T *pd, const int ex, const int L1, T &P, T &Pu, T &Pv,
                                      T &Pw)
{
  const int a = ex & 255, b = (ex >> 8) & 255, c = (ex >> 16) & 255;
  const T pa = pd[S * a], pb = pd[S * (L1 + b)], pc = pd[S * (2 * L1 + c)];
  const T pbc = pb * pc;
  P = pa * pbc;
  Pu = static_cast<T>(a) * pd[S * (a > 0 ? a - 1 : 0)] * pbc;
  Pv = static_cast<T>(b) * pa * pd[S * (L1 + (b > 0 ? b - 1 : 0))] * pc;
  Pw = static_cast<T>(c) * pa * pb * pd[S * (2 * L1 + (c > 0 ? c - 1 : 0))];
}

template <typename T> __device__ inline T pod_sel4(int i, T v0, T v1, T v2, T v3)
{
  T r = T(0);
  r = (i == 0) ? v0 : r;
  r = (i == 1) ? v1 : r;
  r = (i == 2) ? v2 : r;
  r = (i == 3) ? v3 : r;
  return r;
}

// B-spline radial basis of the 2-body terms; r0v == nullptr: use the stored rbf2/drbf2/base
template <typename T> struct BsplineArgs {
  const T *r0v, *invhv;    // bin origin and 1/width per element pair
  const int *ti, *tj;
  int ne, nbm1, gmax, ish;
};

// the 4 nonzero B-splines of pair n and their derivatives (times 1/r), in the slots
// of the window that starts at the returned base index, as in PairPODKokkos::bspline4()
template <typename T>
__device__ inline int pod_bspline_eval(const BsplineArgs<T> &g, const int pair, const T r,
                                       T (&Bq)[4], T (&dBq)[4])
{
  const T invh = g.invhv[pair];
  T tg = (r - g.r0v[pair]) * invh;
  const bool inside = (tg >= T(0));
  const T dscale = inside ? invh / r : T(0);
  if (!inside) tg = T(0);
  int b = (int) tg;
  if (b > g.nbm1) b = g.nbm1;

  // uniform cubic B-spline values and derivatives, as in PairPODKokkos::bspline4()
  const T t = tg - (T) b;
  constexpr T S = (T) (1.0 / 6.0);
  const T t2 = t * t, a = (T) 1.0 - t, a2 = a * a, t3 = (T) 3.0 * t, hd = (T) 0.5 * dscale;
  const T B0 = S * a2 * a;
  const T B1 = (T) 4.0 * S + (T) 0.5 * t2 * (t - (T) 2.0);
  const T B2 = S + (T) 0.5 * t * ((T) 1.0 + t * a);
  const T B3 = S * t2 * t;
  const T dB0 = -a2 * hd;
  const T dB1 = t * (t3 - (T) 4.0) * hd;
  const T dB2 = ((T) 1.0 + t * ((T) 2.0 - t3)) * hd;
  const T dB3 = t2 * hd;

  const int g0 = b - g.ish;
  int gb = g0;
  if (gb < 0) gb = 0;
  else if (gb > g.gmax) gb = g.gmax;
  const int sh = gb - g0;
#pragma unroll
  for (int q = 0; q < 4; q++) {
    Bq[q] = pod_sel4(sh + q, B0, B1, B2, B3);
    dBq[q] = pod_sel4(sh + q, dB0, dB1, dB2, dB3);
  }
  return gb;
}

template <typename T>
__device__ inline int pod_bspline_pair(const BsplineArgs<T> &g, const PairGeom<T> &pg,
                                       const int n, T (&Bq)[4], T (&dBq)[4])
{
  return pod_bspline_eval(g, (g.ne > 1) ? g.ti[n] * g.ne + g.tj[n] : 0, pg.r(n), Bq, dBq);
}


/* ----------------------------------------------------------------------
   the 2-body force of a pair and the tally of the pair forces into the
   forces and the virial of the atoms, done inside the many-body force
   kernels (one wavefront per atom i) so that the pair forces need not be
   stored: f_j -= f_ij per pair, while f_i and the virial of atom i are
   summed in LDS and added once per atom.  the per-atom virial sums go to
   vir[6*i + d], summed over the atoms by virial_reduce_kernel, instead of
   one global atomic per atom on the same six addresses.
------------------------------------------------------------------------- */

template <typename T, typename TA> struct ForceTally {
  TA *f;                       // atom forces, f[a*fs0 + d*fs1]; nullptr: write fij instead
  int fs0, fs1;
  TA *vir;                     // [6*Ni] virial of the atoms of the block, or nullptr
  TA *vatom;                   // per-atom virial vatom[a*vs0 + d*vs1], or nullptr
  int vs0, vs1;
  const int *ai, *aj;          // atom indices of the pairs
  const int *gia;              // atom index of atom i of the block, or nullptr: ai[first pair]
  const T *cb2;                // 2-body force coefficients [i + Ni*m]; nullptr: no 2-body part
  const T *drbf2;              // stored 2-body basis derivatives (used when bs.r0v == nullptr)
  const int *base2;            // stored B-spline base index, or nullptr
  BsplineArgs<T> bs;           // bs.r0v != nullptr: B-splines evaluated here
  int nrbf2, nm2, Ni;
  int ca, cm;                  // cb2 of atom i, coefficient m: cb2[i*ca + cm*m]
};

// adds the force (SA, SU, SV, SW contracted with the gradients g) of pair j of atom i,
// whose neighbor has element e, plus its 2-body force, to the atoms
template <typename T, typename TA>
__device__ inline void pod_pair_finish(const ForceTally<T, TA> &ft, const PairGeom<T> &pg,
                                       T *fij, const int i, const int j, const int e, const T *g,
                                       const TA SA, const TA SU, const TA SV, const TA SW,
                                       TA *acc_s)
{
  TA fx = SU * static_cast<TA>(g[0]) + SV * static_cast<TA>(g[1]) + SW * static_cast<TA>(g[2]) +
      SA * static_cast<TA>(g[6]);
  TA fy = SU * static_cast<TA>(g[1]) + SV * static_cast<TA>(g[3]) + SW * static_cast<TA>(g[4]) +
      SA * static_cast<TA>(g[7]);
  TA fz = SU * static_cast<TA>(g[2]) + SV * static_cast<TA>(g[4]) + SW * static_cast<TA>(g[5]) +
      SA * static_cast<TA>(g[8]);
  if (ft.f == nullptr) {
    fij[3 * j + 0] += static_cast<T>(fx);
    fij[3 * j + 1] += static_cast<T>(fy);
    fij[3 * j + 2] += static_cast<T>(fz);
    return;
  }
  T u, v, w;
  pg.uvw(j, u, v, w);
  const T r = pg.r(j);
  if (ft.cb2) {
    TA cR = TA(0);
    if (ft.bs.r0v) {
      T Bq[4], dBq[4];
      const int ii =
          i * ft.ca + ft.cm * (pod_bspline_pair(ft.bs, pg, j, Bq, dBq) + ft.nrbf2 * e);
#pragma unroll
      for (int m = 0; m < 4; ++m) cR += static_cast<TA>(ft.cb2[ii + ft.cm * m] * dBq[m]);
    } else {
      const int ii = i * ft.ca + ft.cm * ((ft.base2 ? ft.base2[j] : 0) + ft.nrbf2 * e);
      for (int m = 0; m < ft.nm2; ++m)
        cR += static_cast<TA>(ft.cb2[ii + ft.cm * m] * ft.drbf2[j + pg.N * m]);
    }
    cR *= static_cast<TA>(r);
    fx += cR * static_cast<TA>(u);
    fy += cR * static_cast<TA>(v);
    fz += cR * static_cast<TA>(w);
  }
  TA *fj = ft.f + ft.aj[j] * ft.fs0;
  atomicAdd(fj, -fx);
  atomicAdd(fj + ft.fs1, -fy);
  atomicAdd(fj + 2 * ft.fs1, -fz);
  atomicAdd(acc_s + 0, fx);
  atomicAdd(acc_s + 1, fy);
  atomicAdd(acc_s + 2, fz);
  if (ft.vir || ft.vatom) {
    const TA rx = static_cast<TA>(r * u), ry = static_cast<TA>(r * v), rz = static_cast<TA>(r * w);
    const TA vl[6] = {rx * fx, ry * fy, rz * fz, rx * fy, rx * fz, ry * fz};
#pragma unroll
    for (int d = 0; d < 6; ++d) atomicAdd(acc_s + 3 + d, vl[d]);
    if (ft.vatom) {
      TA *vj = ft.vatom + ft.aj[j] * ft.vs0;
#pragma unroll
      for (int d = 0; d < 6; ++d) atomicAdd(vj + d * ft.vs1, static_cast<TA>(-0.5) * vl[d]);
    }
  }
}

// component d (0, 1, 2) of the force of pair j of atom i as pod_pair_finish() (atom forces
// tallied, ft.f != nullptr), with the distance r and the direction (u, v, w) of the pair given
// (N: the number of pairs): subtracts it from f_j and returns it in fd, with the virial terms
// vd = r_d f_d and vo (xy, xz, yz for d = 0, 1, 2) of the pair
template <typename T, typename TA>
__device__ inline void pod_pair_force_dg(const ForceTally<T, TA> &ft, const int N, const int i,
                                         const int j, const int e, const TA SA, const TA SU,
                                         const TA SV, const TA SW, const int d, const T r,
                                         const T u, const T v, const T w, TA &fd, TA &vd,
                                         TA &vo)
{
  const T invr = static_cast<T>(1.0) / r;
  const T g0 = (v * v + w * w) * invr, g1 = -(u * v) * invr, g2 = -(u * w) * invr;
  const T g3 = (u * u + w * w) * invr, g4 = -(v * w) * invr, g5 = (u * u + v * v) * invr;
  TA fx = SU * static_cast<TA>(g0) + SV * static_cast<TA>(g1) + SW * static_cast<TA>(g2) +
      SA * static_cast<TA>(r * u);
  TA fy = SU * static_cast<TA>(g1) + SV * static_cast<TA>(g3) + SW * static_cast<TA>(g4) +
      SA * static_cast<TA>(r * v);
  TA fz = SU * static_cast<TA>(g2) + SV * static_cast<TA>(g4) + SW * static_cast<TA>(g5) +
      SA * static_cast<TA>(r * w);
  if (ft.cb2) {
    TA cR = TA(0);
    if (ft.bs.r0v) {
      T Bq[4], dBq[4];
      const int pair = (ft.bs.ne > 1) ? ft.bs.ti[j] * ft.bs.ne + ft.bs.tj[j] : 0;
      const int ii = i * ft.ca + ft.cm * (pod_bspline_eval(ft.bs, pair, r, Bq, dBq) + ft.nrbf2 * e);
#pragma unroll
      for (int m = 0; m < 4; ++m) cR += static_cast<TA>(ft.cb2[ii + ft.cm * m] * dBq[m]);
    } else {
      const int ii = i * ft.ca + ft.cm * ((ft.base2 ? ft.base2[j] : 0) + ft.nrbf2 * e);
      for (int m = 0; m < ft.nm2; ++m)
        cR += static_cast<TA>(ft.cb2[ii + ft.cm * m] * ft.drbf2[j + N * m]);
    }
    cR *= static_cast<TA>(r);
    fx += cR * static_cast<TA>(u);
    fy += cR * static_cast<TA>(v);
    fz += cR * static_cast<TA>(w);
  }
  fd = (d == 0) ? fx : ((d == 1) ? fy : fz);
  atomicAdd(ft.f + ft.aj[j] * ft.fs0 + d * ft.fs1, -fd);
  vd = vo = TA(0);
  if (ft.vir || ft.vatom) {
    const TA rx = static_cast<TA>(r * u), ry = static_cast<TA>(r * v), rz = static_cast<TA>(r * w);
    vd = ((d == 0) ? rx : ((d == 1) ? ry : rz)) * fd;
    vo = (d == 0) ? rx * fy : ((d == 1) ? rx * fz : ry * fz);
    if (ft.vatom) {
      TA *vj = ft.vatom + ft.aj[j] * ft.vs0;
      atomicAdd(vj + d * ft.vs1, static_cast<TA>(-0.5) * vd);
      atomicAdd(vj + (3 + d) * ft.vs1, static_cast<TA>(-0.5) * vo);
    }
  }
}

// as pod_pair_force_dg(), with the geometry of pair j read from pg
template <typename T, typename TA>
__device__ inline void pod_pair_force_d(const ForceTally<T, TA> &ft, const PairGeom<T> &pg,
                                        const int i, const int j, const int e, const TA SA,
                                        const TA SU, const TA SV, const TA SW, const int d,
                                        TA &fd, TA &vd, TA &vo)
{
  T u, v, w;
  pg.uvw(j, u, v, w);
  pod_pair_force_dg<T, TA>(ft, pg.N, i, j, e, SA, SU, SV, SW, d, pg.r(j), u, v, w, fd, vd, vo);
}

// adds the sums of atom i (pairs start .. start+nj-1) to its force and virial; called by
// all threads of the block after the last pair
template <typename T, typename TA>
__device__ inline void pod_atom_finish(const ForceTally<T, TA> &ft, const int i, const int start,
                                       const int nj, const TA *acc_s)
{
  if ((ft.f == nullptr) || (threadIdx.x != 0)) return;
  if (nj > 0) {
    const int ga = ft.gia ? ft.gia[i] : ft.ai[start];
    TA *fi = ft.f + ga * ft.fs0;
    atomicAdd(fi, acc_s[0]);
    atomicAdd(fi + ft.fs1, acc_s[1]);
    atomicAdd(fi + 2 * ft.fs1, acc_s[2]);
    if (ft.vatom) {
      TA *vi = ft.vatom + ga * ft.vs0;
      for (int d = 0; d < 6; ++d) atomicAdd(vi + d * ft.vs1, static_cast<TA>(-0.5) * acc_s[3 + d]);
    }
  }
  if (ft.vir)
    for (int d = 0; d < 6; ++d) ft.vir[6 * i + d] = acc_s[3 + d];
}

// scalar version: one block per atom i, one thread per neighbor j

template <typename T, typename TA>
__global__ void __launch_bounds__(64)
    allbody_forces_otf_kernel(T *__restrict__ fij, const ForceTally<T, TA> ft,
                              const T *__restrict__ forcecoeff,
                              const RadialSrc<T> rs, const PairGeom<T> pg, const int *__restrict__ tj,
                              const int *__restrict__ numij, const int *__restrict__ pq_m,
                              const int *__restrict__ pq_d, const int nelements, const int nrbf3,
                              const int K3, const int L, const int Nij)
{
  const int L1 = L + 1;
  const int PD = 3 * L1;
  const int NeK3 = nelements * K3;
  const int nfc = NeK3 * nrbf3;
  extern __shared__ char lds_raw[];
  T *fc_s = reinterpret_cast<T *>(lds_raw);    // forcecoeff of atom i: [m][k][e]
  T *pd_s = fc_s + nfc;                        // [blockDim.x][PD]
  int *ex_s = reinterpret_cast<int *>(pd_s + blockDim.x * PD);    // [K3]

  __shared__ TA acc_s[9];    // force and virial sums of atom i

  const int i = blockIdx.x;
  int start, nj;
  pg.range(numij, i, start, nj);

  for (int t = threadIdx.x; t < nfc; t += blockDim.x) fc_s[t] = forcecoeff[t + nfc * i];
  if (threadIdx.x < 9) acc_s[threadIdx.x] = TA(0);
  pod_exponents<T>(ex_s, pq_m, pq_d, K3);
  __syncthreads();

  T *pd = pd_s + threadIdx.x * PD;
  for (int jj = threadIdx.x; jj < nj; jj += blockDim.x) {
    const int j = start + jj;
    const int e = (nelements > 1) ? tj[j] : 0;
    pod_pair_powers<T>(pd, pg, j, L1);

    T r[MAX_RADIAL], dr[MAX_RADIAL];
    pair_radial<MAX_RADIAL>(rs, pg, j, nrbf3, [&](int m, T f, T df) {
      r[m] = f;
      dr[m] = df;
    });

    TA SA = TA(0), SU = TA(0), SV = TA(0), SW = TA(0);
    for (int k = 0; k < K3; k++) {
      const T *fck = fc_s + e + nelements * k;
      TA G = TA(0), H = TA(0);
#pragma unroll
      for (int m = 0; m < MAX_RADIAL; m++)
        if (m < nrbf3) {
          const T fc = fck[NeK3 * m];
          G += static_cast<TA>(fc * r[m]);
          H += static_cast<TA>(fc * dr[m]);
        }
      T P, Pu, Pv, Pw;
      pod_monomial<T>(pd, ex_s[k], L1, P, Pu, Pv, Pw);
      SA += H * static_cast<TA>(P);
      SU += G * static_cast<TA>(Pu);
      SV += G * static_cast<TA>(Pv);
      SW += G * static_cast<TA>(Pw);
    }
    T g[9];
    pod_pair_gradients<T>(g, pg, j);
    pod_pair_finish<T, TA>(ft, pg, fij, i, j, e, g, SA, SU, SV, SW, acc_s);
  }
  __syncthreads();
  pod_atom_finish<T, TA>(ft, i, start, nj, acc_s);
}

/* ----------------------------------------------------------------------
   matrix core version: one wavefront per atom i.  for a tile of 16 pairs,

     G^T (K3 x 16) = F^T (K3 x Ne*nrbf3) * R^T (Ne*nrbf3 x 16)

   with R(j, m + nrbf3*e) = [tj(j) == e] rbf(j, m) and F the forcecoeff
   block of atom i (H likewise with drbf), in 16x16 output tiles.  each lane
   holds G and H for one pair (its column) and 4 angular indices k, so it
   accumulates SA, SU, SV, SW of that pair over its own k, and only the 4
   lanes that share the column are reduced.  the lanes evaluate the radial
   functions of their own entries of R, so only the powers of u, v, w go
   through LDS.
------------------------------------------------------------------------- */

static constexpr int MFMA_MAX_CS = 8;    // nelements * nrbf3 <= 32
static constexpr int MFMA_PAIR_TILE = 16;

#ifndef LMP_POD_HF_SPREAD
#define LMP_POD_HF_SPREAD 1
#endif
// LMP_POD_HF_EREG (HF spread epilogue): the direction of the pair is taken from the powers of
// the tile in LDS instead of global memory
#ifndef LMP_POD_HF_EREG
#define LMP_POD_HF_EREG 1
#endif
// LMP_POD_HF_PDT (HF instance): the power table of the tile is stored [direction][power][pair]
// instead of [pair][direction][power], so that the 16 lanes of a row group (16 pairs, same
// monomial) read consecutive entries (no LDS bank conflicts)
#ifndef LMP_POD_HF_PDT
#define LMP_POD_HF_PDT 1
#endif
// LMP_POD_HF_REMAT (HF instance, bits): the lane-dependent indices of a region of the tile loop
// are derived from a copy of the lane index the compiler cannot see through, so that it
// recomputes them there instead of keeping them in registers over the whole loop (1 force
// epilogue, 2 tile setup, 4 matrix core part)
#ifndef LMP_POD_HF_REMAT
#define LMP_POD_HF_REMAT 7
#endif
__device__ inline int pod_opaque(int x)
{
  asm volatile("" : "+v"(x));
  return x;
}

// register budget of the matrix core force kernel: without it the compiler keeps the
// accumulators in separate AGPRs, which costs occupancy
#ifndef LMP_POD_ALLBODY_VGPR
#define LMP_POD_ALLBODY_VGPR 128
#endif
#if LMP_POD_ALLBODY_VGPR > 0
#define LMP_POD_ALLBODY_ATTR __attribute__((amdgpu_num_vgpr(LMP_POD_ALLBODY_VGPR)))
#else
#define LMP_POD_ALLBODY_ATTR
#endif

// HF: the sizes of the Hf model (PodHf) as constants (descriptors stored per atom)
template <typename T, typename TA, int CSM, int JT, bool GROUP, bool HF>
__global__ void __launch_bounds__(64) LMP_POD_ALLBODY_ATTR
    allbody_forces_mfma_kernel(T *__restrict__ fij, const ForceTally<T, TA> ft_in,
                               const T *__restrict__ forcecoeff, const FcArgs<T> fca_in,
                               const RadialSrc<T> rs_in, const PairGeom<T> pg,
                               const int *__restrict__ tj, const int *__restrict__ numij,
                               const int *__restrict__ pq_m, const int *__restrict__ pq_d,
                               const int nelements_in, const int nrbf3, const int K3_in,
                               const int L_in, const int Nij, const int *__restrict__ pq_ex)
{
  ForceTally<T, TA> ft = ft_in;
  FcArgs<T> fca = fca_in;
  RadialSrc<T> rs = rs_in;
  if constexpr (HF) {
    ft.nrbf2 = PodHf::nrbf2;
    ft.nm2 = 4;
    ft.ca = PodHf::nDes;
    ft.cm = 1;
    ft.bs.ne = 1;
    rs.ne = 1;
    fca.nelements = 1;
    fca.nabf3_active = PodHf::nabf3_active;
    fca.nrbf4 = PodHf::nrbf4;
    fca.nabf4 = PodHf::nabf4;
    fca.K3 = PodHf::K3;
    fca.Q4 = PodHf::Q4;
    fca.do4 = 1;
    fca.ca = PodHf::nDes;
    fca.cm = 1;
  }
  const int nelements = HF ? 1 : nelements_in;
  const int K3 = HF ? PodHf::K3 : K3_in;
  const int L = HF ? PodHf::L : L_in;
  const int L1 = L + 1;
  const int PD = 3 * L1;    // only the powers; the gradients are recomputed per pair
  const int NeK3 = nelements * K3;
  const int nfc = NeK3 * nrbf3;
  const int NC = nelements * nrbf3;
  const int CS = (NC + 3) / 4;
  const int KT = (K3 + 15) / 16;
  extern __shared__ char lds_raw[];
  T *fc_s = reinterpret_cast<T *>(lds_raw);    // [m][k][e]
  T *pd_s = fc_s + nfc;    // [JT][PD]; before the pair loop, scratch of forcecoeff34_atom()
  int *tj_s = reinterpret_cast<int *>(pd_s + JT * PD);       // [JT]
  int *ex_s = tj_s + JT;                                     // [K3]

  const int i = blockIdx.x;
  int start, nj;
  pg.range(numij, i, start, nj);
  const int lane = threadIdx.x;
  const int lr = lane & 15, lq = lane >> 4;

  __shared__ TA acc_s[9];    // force and virial sums of atom i
  if (fca.sumU) {
    int *qa_s = reinterpret_cast<int *>(pd_s + nfc);
    forcecoeff34_atom<T>(fca, fc_s, pd_s, qa_s, qa_s + K3, i);
  } else {
    for (int t = lane; t < nfc; t += 64) fc_s[t] = forcecoeff[t + nfc * i];
  }
  if (lane < 9) acc_s[lane] = TA(0);
  // the HF instance copies the table built on the host (the generic one would need 2 more
  // VGPRs for it, one occupancy step)
  if constexpr (HF)
    pod_exponents_table(ex_s, pq_ex, K3);
  else
    pod_exponents<T>(ex_s, pq_m, pq_d, K3);

  // (m, e) of the inner index c = 4*cs + lq owned by this lane.  with GROUP the slices are
  // processed in groups of CG = 4 (see below), which recompute (m, e)
  constexpr int CG = (CSM < 4) ? CSM : 4;
  constexpr int CN = GROUP ? CG : CSM;
  int cm[CN], ce[CN];
#pragma unroll
  for (int cs = 0; cs < CN; cs++) {
    const int c = 4 * cs + lq;
    cm[cs] = (c < NC) ? c % nrbf3 : 0;
    ce[cs] = (c < NC) ? c / nrbf3 : -1;
  }
  typedef typename Mfma<TA>::acc_t acc_t;

  // LMP_POD_HF_SPREAD (HF instance): the force of a pair is finished by three lane groups, one
  // component each, summed per lane (af, avd, avo) and reduced once per atom, instead of one
  // lane group and per-pair LDS atomics
  constexpr bool SPREAD = HF && LMP_POD_HF_SPREAD;
  constexpr bool EREG = SPREAD && LMP_POD_HF_EREG;
  constexpr bool PDT = HF && (JT == 16) && (CSM == 2) && !GROUP && LMP_POD_HF_PDT;
  constexpr int PS = PDT ? JT : 1;    // stride of the powers of a pair in pd_s
  TA af = TA(0), avd = TA(0), avo = TA(0);

  // the direction (u, v or w) and element of the pairs of the next tile are loaded while the
  // current tile is computed.  only where the 2 registers this takes do not lower the
  // occupancy (2 slices, and 5 or 6 slices in groups)
  constexpr bool PREFETCH = (CSM == 2) || GROUP;
  T xh = T(0);
  int tjh = -1;
  auto fetch = [&](const int jbase) {
    if (lane < 3 * JT) {
      const int jj = lane % JT, d = lane / JT;
      if (jbase + jj < nj) {
        const int j = start + jbase + jj;
        xh = pg.rhat[j + d * pg.N];
        tjh = (nelements > 1) ? tj[j] : 0;
      }
    }
  };
  if constexpr (PREFETCH) fetch(0);

  for (int j0 = 0; j0 < nj; j0 += JT) {
    const int nt = min(JT, nj - j0);
    __syncthreads();
    // powers of u, v and w of the pairs of the tile, one direction per lane
    if constexpr (PREFETCH) {
      const int ls = (HF && (LMP_POD_HF_REMAT & 2)) ? pod_opaque(lane) : lane;
      if (ls < 3 * JT) {
        const int jj = ls % JT, d = ls / JT;
        T *pd = PDT ? pd_s + d * L1 * JT + jj : pd_s + jj * PD + d * L1;
        const bool in = (jj < nt);
        T p = in ? static_cast<T>(1.0) : T(0);
        for (int n = 0; n < L1; n++) {
          pd[PS * n] = p;
          p *= xh;
        }
        if (d == 0) tj_s[jj] = in ? tjh : -1;
      }
      if (j0 + JT < nj) fetch(j0 + JT);
    } else
    for (int t = lane; t < 3 * JT; t += 64) {
      const int jj = t % JT, d = t / JT;
      T *pd = pd_s + jj * PD + d * L1;
      if (jj < nt) {
        const int j = start + j0 + jj;
        T u, v, w;
        pg.uvw(j, u, v, w);
        const T x = (d == 0) ? u : ((d == 1) ? v : w);
        T p = static_cast<T>(1.0);
        for (int n = 0; n < L1; n++) {
          pd[n] = p;
          p *= x;
        }
        if (d == 0) tj_s[jj] = (nelements > 1) ? tj[j] : 0;
      } else {
        for (int n = 0; n < L1; n++) pd[n] = T(0);
        if (d == 0) tj_s[jj] = -1;
      }
    }
    __syncthreads();

    // C^T (k x pairs) = F^T R^T on the matrix cores: lane column lr is pair s0 + lr and
    // its four accumulators are rows k = 16 kt + row(lane, v), so each lane sums over its own
    // k and only the four row groups of a pair have to be reduced
    for (int s0 = 0; s0 < nt; s0 += 16) {
      TA SA = TA(0), SU = TA(0), SV = TA(0), SW = TA(0);

      // B operands of this pair tile: column s0 + lr, inner index 4*cs + lq; each lane
      // evaluates the radial functions of its own entries
      const int lb = (HF && (LMP_POD_HF_REMAT & 4)) ? pod_opaque(lane) : lane;
      const int lrb = lb & 15;
      const int jb = s0 + lrb;
      const int tjb = tj_s[jb];
      const PairRadial<T> prad(rs, pg, start + j0 + min(jb, nt - 1));
      const T *pd = PDT ? pd_s + jb : pd_s + jb * PD;
      if constexpr (!GROUP) {
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

        for (int kt = 0; kt < KT; kt++) {
          const int k = 16 * kt + lrb;
          acc_t accG = acc_t{0, 0, 0, 0}, accH = acc_t{0, 0, 0, 0};
#pragma unroll
          for (int cs = 0; cs < CSM; cs++)
            if (cs < CS) {
              const TA a = ((k < K3) && (ce[cs] >= 0))
                  ? static_cast<TA>(fc_s[ce[cs] + nelements * k + NeK3 * cm[cs]])
                  : TA(0);
              accG = Mfma<TA>::mma(a, br[cs], accG);
              accH = Mfma<TA>::mma(a, bdr[cs], accH);
            }
#pragma unroll
          for (int v = 0; v < 4; v++) {
            const int kv = 16 * kt + Mfma<TA>::row(lb, v);
            if (kv < K3) {
              T P, Pu, Pv, Pw;
              pod_monomial_s<T, PS>(pd, ex_s[kv], L1, P, Pu, Pv, Pw);
              SA += accH[v] * static_cast<TA>(P);
              SU += accG[v] * static_cast<TA>(Pu);
              SV += accG[v] * static_cast<TA>(Pv);
              SW += accG[v] * static_cast<TA>(Pw);
            }
          }
        }

      } else {
        // only the B operands of one group of 4 slices are live (all 8 need about 200
        // registers, 2 waves/SIMD), and each group adds its own products with the
        // monomials
#pragma unroll 1
        for (int g0 = 0; g0 < CS; g0 += CG) {
          int gm[CG], ge[CG];
          TA br[CG], bdr[CG];
#pragma unroll
          for (int q = 0; q < CG; q++) {
            const int c = 4 * (g0 + q) + lq;
            gm[q] = (c < NC) ? c % nrbf3 : 0;
            ge[q] = (c < NC) ? c / nrbf3 : -1;
            br[q] = bdr[q] = TA(0);
            if ((g0 + q < CS) && (jb < nt) && (ge[q] == tjb)) {
              T f, df;
              prad(gm[q], f, df);
              br[q] = static_cast<TA>(f);
              bdr[q] = static_cast<TA>(df);
            }
          }

          for (int kt = 0; kt < KT; kt++) {
            const int k = 16 * kt + lr;
            acc_t accG = acc_t{0, 0, 0, 0}, accH = acc_t{0, 0, 0, 0};
#pragma unroll
            for (int q = 0; q < CG; q++)
              if (g0 + q < CS) {
                const TA a = ((k < K3) && (ge[q] >= 0))
                    ? static_cast<TA>(fc_s[ge[q] + nelements * k + NeK3 * gm[q]])
                    : TA(0);
                accG = Mfma<TA>::mma(a, br[q], accG);
                accH = Mfma<TA>::mma(a, bdr[q], accH);
              }
#pragma unroll
            for (int v = 0; v < 4; v++) {
              const int kv = 16 * kt + Mfma<TA>::row(lane, v);
              if (kv < K3) {
                T P, Pu, Pv, Pw;
                pod_monomial_s<T, PS>(pd, ex_s[kv], L1, P, Pu, Pv, Pw);
                SA += accH[v] * static_cast<TA>(P);
                SU += accG[v] * static_cast<TA>(Pu);
                SV += accG[v] * static_cast<TA>(Pv);
                SW += accG[v] * static_cast<TA>(Pw);
              }
            }
          }
        }
      }

      // reduce over the four row groups (lanes lr, lr + 16, lr + 32, lr + 48)
#pragma unroll
      for (int off = 16; off < 64; off <<= 1) {
        SA += __shfl_xor(SA, off);
        SU += __shfl_xor(SU, off);
        SV += __shfl_xor(SV, off);
        SW += __shfl_xor(SW, off);
      }
      if constexpr (SPREAD) {
        const int le = (HF && (LMP_POD_HF_REMAT & 1)) ? pod_opaque(lane) : lane;
        const int lqe = le >> 4, jbe = s0 + (le & 15);
        if ((lqe < 3) && (jbe < nt)) {
          TA fd, vd, vo;
          const T *pde = PDT ? pd_s + jbe : pd_s + jbe * PD;
          if constexpr (EREG)
            pod_pair_force_dg<T, TA>(
                ft, pg.N, i, start + j0 + jbe, tj_s[jbe], SA, SU, SV, SW, lqe,
                pg.r(start + j0 + jbe), pde[PS], pde[PS * (L1 + 1)], pde[PS * (2 * L1 + 1)], fd,
                vd, vo);
          else
            pod_pair_force_d<T, TA>(ft, pg, i, start + j0 + jb, tjb, SA, SU, SV, SW, lq, fd, vd,
                                    vo);
          af += fd;
          avd += vd;
          avo += vo;
        }
      } else if ((lq == 0) && (jb < nt)) {
        T g[9];
        pod_pair_gradients<T>(g, pg, start + j0 + jb);
        pod_pair_finish<T, TA>(ft, pg, fij, i, start + j0 + jb, tjb, g, SA, SU, SV, SW, acc_s);
      }
    }
  }
  if constexpr (SPREAD) {
#pragma unroll
    for (int off = 1; off < 16; off <<= 1) {
      af += __shfl_xor(af, off);
      avd += __shfl_xor(avd, off);
      avo += __shfl_xor(avo, off);
    }
    __syncthreads();
    if ((lr == 0) && (lq < 3)) {
      acc_s[lq] = af;
      acc_s[3 + lq] = avd;
      acc_s[6 + lq] = avo;
    }
  }
  __syncthreads();
  pod_atom_finish<T, TA>(ft, i, start, nj, acc_s);
}

// fca != nullptr: the force coefficients are not in forcecoeff yet.  the matrix core kernel
// computes them itself; otherwise forcecoeff34() writes them first
template <typename T, typename TA>
bool allbody_forces_otf(hipStream_t stream, T *fij, const ForceTally<T, TA> &ft,
                        T *forcecoeff, const FcArgs<T> *fca, const RadialSrc<T> &rs,
                        const PairGeom<T> &pg, const int *tj, const int *numij,
                        const int *pq_m, const int *pq_d, int nelements, int nrbf3, int K3,
                        int Ni, int Nij, bool use_mfma, bool hf = false,
                        const int *pq_ex = nullptr)
{
  if ((Ni <= 0) || (nrbf3 > MAX_RADIAL) || (nelements > MAX_ELEMENTS)) return false;
  int L = 0;
  while ((L + 1) * (L + 2) * (L + 3) / 6 < K3) L++;
  if ((L + 1) * (L + 2) * (L + 3) / 6 != K3) return false;
  const int PD = 3 * (L + 1);
  const size_t nfc = (size_t) nelements * K3 * nrbf3;
  const int CS = (nelements * nrbf3 + 3) / 4;
  bool fc_pending = (fca != nullptr);

  if (use_mfma && device_has_mfma() && (CS <= MFMA_MAX_CS)) {
    constexpr int JT = MFMA_PAIR_TILE;
    // the coefficients are only computed in the kernel if their scratch fits in the space of
    // the pair tile: a larger block costs occupancy, which is slower than the separate kernel
    const size_t tile = (size_t) JT * PD * sizeof(T) + ((size_t) JT + K3) * sizeof(int);
    const bool fuse = fc_pending && (forcecoeff34_scratch(*fca) <= tile);
    if (fc_pending && !fuse) {
      if (!forcecoeff34(stream, forcecoeff, *fca)) return false;
      fc_pending = false;
    }
    const size_t lds = nfc * sizeof(T) + tile;
    const FcArgs<T> g = fuse ? *fca : FcArgs<T>{};
    if (lds <= 65536) {
#define LMP_POD_ALLBODY_MFMA1(CSM, GROUP, H)                                                       \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(allbody_forces_mfma_kernel<T, TA, CSM, JT, GROUP, H>),        \
                     dim3(Ni), dim3(64), lds, stream, fij, ft, forcecoeff, g, rs, pg, tj, numij,   \
                     pq_m, pq_d, nelements, nrbf3, K3, L, Nij, pq_ex)
#define LMP_POD_ALLBODY_MFMA(CSM, GROUP) LMP_POD_ALLBODY_MFMA1(CSM, GROUP, false)
      // 5 or 6 slices: two groups (90 instead of 208 registers) is faster; with 7 or 8 the
      // second pass over the monomials costs more than the occupancy gains
      if (hf && pq_ex && (CS <= 2))
        LMP_POD_ALLBODY_MFMA1(2, false, true);
      else if (CS <= 2)
        LMP_POD_ALLBODY_MFMA(2, false);
      else if (CS <= 4)
        LMP_POD_ALLBODY_MFMA(4, false);
      else if (CS <= 6)
        LMP_POD_ALLBODY_MFMA(8, true);
      else
        LMP_POD_ALLBODY_MFMA(8, false);
#undef LMP_POD_ALLBODY_MFMA
#undef LMP_POD_ALLBODY_MFMA1
      return hipGetLastError() == hipSuccess;
    }
  }

  if (fc_pending && !forcecoeff34(stream, forcecoeff, *fca)) return false;
  const size_t lds = (nfc + (size_t) 64 * PD) * sizeof(T) + (size_t) K3 * sizeof(int);
  if (lds > 65536) return false;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(allbody_forces_otf_kernel<T, TA>), dim3(Ni), dim3(64), lds,
                     stream, fij, ft, forcecoeff, rs, pg, tj, numij, pq_m, pq_d, nelements,
                     nrbf3, K3, L, Nij);
  return hipGetLastError() == hipSuccess;
}


/* ----------------------------------------------------------------------
   three- and four-body descriptors, fused (threebodydesc() followed by
   fourbodydesc()).  one block per atom i with sumU of atom i in LDS; one
   thread per (m, angular index) item loops over the element pairs or
   triples.  same arithmetic and summation order as the Kokkos kernels.
------------------------------------------------------------------------- */

template <typename T, typename TA>
__global__ void desc34_kernel(const DescArgs<T> g, const T *__restrict__ sumU)
{
  extern __shared__ char lds_raw[];
  const int S = g.nelements * g.K3 * g.nrbf3;
  T *u_s = reinterpret_cast<T *>(lds_raw);    // sumU of atom i: [m][q][e]
  const int i = blockIdx.x;
  for (int t = threadIdx.x; t < S; t += blockDim.x) u_s[t] = sumU[S * i + t];
  __syncthreads();
  desc34_atom<T, TA>(g, u_s, i);
}

// three-body descriptors only: one thread per (a, m, i), with the same
// decomposition, arithmetic and summation order as threebodydesc()
template <typename T, typename TA>
__global__ void __launch_bounds__(1024)
    desc3_kernel(T *__restrict__ d3, const T *__restrict__ sumU, const int *__restrict__ pc3,
                 const int2 *__restrict__ qrange3, const int nelements, const int nrbf3,
                 const int nabf3_active, const int K3, const int Ni)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Ni * nrbf3 * nabf3_active) return;
  const int a = idx % nabf3_active;
  const int m = (idx / nabf3_active) % nrbf3;
  const int i = idx / (nabf3_active * nrbf3);
  const int2 qr = qrange3[a];    // = pn3[p3_active[a]], pn3[p3_active[a] + 1]
  const int n1 = qr.x, n2 = qr.y;
  const T *u = sumU + nelements * K3 * (m + nrbf3 * i);
  int k = i + Ni * (a + nabf3_active * m);
  if (nelements == 1) {
    TA tmp = TA(0);
    for (int q = n1; q < n2; q++) {
      const T su = u[q];
      tmp += static_cast<TA>(static_cast<T>(pc3[q]) * su * su);
    }
    d3[k] = static_cast<T>(tmp);
    return;
  }
  const int N3 = Ni * nabf3_active * nrbf3;
  for (int i1 = 0; i1 < nelements; i1++)
    for (int i2 = i1; i2 < nelements; i2++, k += N3) {
      TA tmp = TA(0);
      for (int q = n1; q < n2; q++)
        tmp += static_cast<TA>(static_cast<T>(pc3[q]) * u[i1 + nelements * q] *
                               u[i2 + nelements * q]);
      d3[k] = static_cast<T>(tmp);
    }
}

// zero an array, with 16-byte stores when it is 16-byte aligned (all-zero bits are 0.0)
template <typename T>
__global__ void __launch_bounds__(256) zero_kernel(T *__restrict__ a, const long n)
{
  constexpr int V = 16 / sizeof(T);
  const long stride = (long) gridDim.x * blockDim.x;
  const long t = (long) blockIdx.x * blockDim.x + threadIdx.x;
  const long nv = (reinterpret_cast<size_t>(a) % 16 == 0) ? n / V : 0;
  int4 *a4 = reinterpret_cast<int4 *>(a);
  for (long q = t; q < nv; q += stride) a4[q] = make_int4(0, 0, 0, 0);
  for (long k = V * nv + t; k < n; k += stride) a[k] = T(0);
}

template <typename T, typename TA> bool desc34(hipStream_t stream, const DescArgs<T> &g, const T *sumU)
{
  if (g.Ni <= 0) return false;
  if (!g.do4)
    return launch(desc3_kernel<T, TA>, nblocks_for(g.Ni * g.nrbf3 * g.nabf3_active, 1024), 1024, 0,
                  stream, g.d3, sumU, g.pc3, g.qrange3, g.nelements, g.nrbf3, g.nabf3_active, g.K3,
                  g.Ni);
  const size_t lds = (size_t) g.nelements * g.K3 * g.nrbf3 * sizeof(T);
  if (lds > 65536) return false;
  return launch(desc34_kernel<T, TA>, g.Ni, 64, lds, stream, g, sumU);
}

/* ----------------------------------------------------------------------
   force tally: f(ai) += fij, f(aj) -= fij.  one block per atom i of the
   block; the pairs of atom i are contiguous and share ai, so their sum is
   reduced in the block and added with one atomic per component instead of
   one per pair.  fs0, fs1 are the strides of the per-atom force view.
------------------------------------------------------------------------- */

template <typename T, typename TA>
__global__ void __launch_bounds__(64)
    tallyforce_kernel(TA *__restrict__ f, const int fs0, const int fs1, const T *__restrict__ fij,
                      const int *__restrict__ ai, const int *__restrict__ aj,
                      const int *__restrict__ numij)
{
  const int i = blockIdx.x;
  const int start = numij[i];
  const int end = numij[i + 1];
  TA sx = TA(0), sy = TA(0), sz = TA(0);
  for (int n = start + threadIdx.x; n < end; n += 64) {
    const TA fx = static_cast<TA>(fij[3 * n + 0]);
    const TA fy = static_cast<TA>(fij[3 * n + 1]);
    const TA fz = static_cast<TA>(fij[3 * n + 2]);
    TA *fj = f + aj[n] * fs0;
    atomicAdd(fj, -fx);
    atomicAdd(fj + fs1, -fy);
    atomicAdd(fj + 2 * fs1, -fz);
    sx += fx;
    sy += fy;
    sz += fz;
  }
  for (int off = 32; off > 0; off >>= 1) {
    sx += __shfl_xor(sx, off);
    sy += __shfl_xor(sy, off);
    sz += __shfl_xor(sz, off);
  }
  if ((threadIdx.x == 0) && (end > start)) {
    TA *fi = f + ai[start] * fs0;
    atomicAdd(fi, sx);
    atomicAdd(fi + fs1, sy);
    atomicAdd(fi + 2 * fs1, sz);
  }
}

template <typename T, typename TA>
bool tallyforce(hipStream_t stream, TA *f, int fs0, int fs1, const T *fij, const int *ai,
                const int *aj, const int *numij, int Ni)
{
  if (Ni <= 0) return false;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(tallyforce_kernel<T, TA>), dim3(Ni), dim3(64), 0, stream, f,
                     fs0, fs1, fij, ai, aj, numij);
  return hipGetLastError() == hipSuccess;
}

/* ----------------------------------------------------------------------
   two-body descriptors d2(i, m, e) = sum_j [tj(j) == e] rbf2(j, m).  one
   block per atom i; the sums are formed with LDS atomics and written once,
   instead of one global atomic per pair and radial function.
------------------------------------------------------------------------- */

template <typename T, typename TA>
__global__ void __launch_bounds__(64)
    twobodydesc_kernel(T *__restrict__ d2, const T *__restrict__ rbf2, const int *__restrict__ tj,
                       const int *__restrict__ numij, const int nelements, const int nrbf2,
                       const int Ni, const int Nij, const int ps, const int da, const int dm)
{
  extern __shared__ char lds_raw[];
  TA *s_s = reinterpret_cast<TA *>(lds_raw);    // [e][m]
  const int i = blockIdx.x;
  const int start = (ps > 0) ? i * ps : numij[i];
  const int end = (ps > 0) ? start + numij[i + 1] : numij[i + 1];
  const int nd = nelements * nrbf2;
  for (int t = threadIdx.x; t < nd; t += blockDim.x) s_s[t] = TA(0);
  __syncthreads();
  for (int n = start + threadIdx.x; n < end; n += blockDim.x) {
    TA *se = s_s + nrbf2 * ((nelements > 1) ? tj[n] : 0);
    for (int m = 0; m < nrbf2; m++) atomicAdd(se + m, static_cast<TA>(rbf2[n + Nij * m]));
  }
  __syncthreads();
  for (int t = threadIdx.x; t < nd; t += blockDim.x)
    d2[(long) i * da + (long) dm * t] = static_cast<T>(s_s[t]);
}

template <typename T, typename TA>
bool twobodydesc(hipStream_t stream, T *d2, const T *rbf2, const int *tj, const int *numij,
                 int nelements, int nrbf2, int Ni, int Nij, int ps, int da, int dm)
{
  if (Ni <= 0) return false;
  const size_t lds = (size_t) nelements * nrbf2 * sizeof(TA);
  if (lds > 65536) return false;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(twobodydesc_kernel<T, TA>), dim3(Ni), dim3(64), lds, stream,
                     d2, rbf2, tj, numij, nelements, nrbf2, Ni, Nij, ps, da, dm);
  return hipGetLastError() == hipSuccess;
}

/* ======================================================================
   native versions of the remaining pair pod/kk kernels.  unless noted,
   they use the same decomposition and arithmetic as the Kokkos kernels
   they replace; kernels that share an index space are fused.
====================================================================== */

// wavefront and block sums; blockDim.x must be a multiple of 64 and at most 1024
// sum over the 64 lanes with DPP row operations (no LDS permutes), the result in every lane
// (read from lane 63).  CDNA (gfx9) DPP controls: quad_perm, row_half_mirror, row_mirror,
// row_bcast15, row_bcast31
template <int CTRL, int ROWMASK> __device__ inline double dpp_move(const double v)
{
  const long long b = __builtin_bit_cast(long long, v);
  const int lo = __builtin_amdgcn_update_dpp(0, (int) (b & 0xffffffffll), CTRL, ROWMASK, 0xf, false);
  const int hi = __builtin_amdgcn_update_dpp(0, (int) (b >> 32), CTRL, ROWMASK, 0xf, false);
  return __builtin_bit_cast(double, ((long long) hi << 32) | (unsigned int) lo);
}
template <int CTRL, int ROWMASK> __device__ inline float dpp_move(const float v)
{
  return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, v),
                                                                CTRL, ROWMASK, 0xf, false));
}
__device__ inline double lane63(const double v)
{
  const long long b = __builtin_bit_cast(long long, v);
  const int lo = __builtin_amdgcn_readlane((int) (b & 0xffffffffll), 63);
  const int hi = __builtin_amdgcn_readlane((int) (b >> 32), 63);
  return __builtin_bit_cast(double, ((long long) hi << 32) | (unsigned int) lo);
}
__device__ inline float lane63(const float v)
{
  return __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), 63));
}
template <typename V> __device__ inline V wave_sum_dpp(V v)
{
  v += dpp_move<0xb1, 0xf>(v);     // quad_perm [1,0,3,2]
  v += dpp_move<0x4e, 0xf>(v);     // quad_perm [2,3,0,1]
  v += dpp_move<0x141, 0xf>(v);    // row_half_mirror
  v += dpp_move<0x140, 0xf>(v);    // row_mirror
  v += dpp_move<0x142, 0xa>(v);    // row_bcast15 to rows 1 and 3
  v += dpp_move<0x143, 0xc>(v);    // row_bcast31 to rows 2 and 3
  return lane63(v);
}

template <typename V> __device__ inline V wave_sum(V v)
{
  for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off);
  return v;
}

// sum over the block, valid in thread 0; sh needs blockDim.x / 64 entries
template <typename V> __device__ inline V block_sum(V v, V *sh)
{
  v = wave_sum(v);
  if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = v;
  __syncthreads();
  V r = V(0);
  if (threadIdx.x == 0)
    for (int w = 0; w < (int) (blockDim.x >> 6); w++) r += sh[w];
  __syncthreads();
  return r;
}

/* ----------------------------------------------------------------------
   neighbors.  one wavefront per atom i; the neighbors of i are tested 64 at
   a time and the pairs inside the cutoff are compacted with a ballot, so
   they are stored in the same order as by the Kokkos team scan.
------------------------------------------------------------------------- */

struct NeighArgs {
  const int *ilist, *numneigh, *neighbors, *type, *map;
  int ns0, ns1;               // strides of the neighbor view
  int xs0, xs1;               // strides of the coordinate view
  int cs0, cs1;               // strides of rcutsq
};

template <typename T>
__device__ inline bool pod_neigh_test(const NeighArgs &a, const T *x, const T *rcutsq, int gi,
                                      int itype, T xi0, T xi1, T xi2, int jj, int &gj, int &jtype,
                                      T &dx, T &dy, T &dz)
{
  gj = a.neighbors[gi * a.ns0 + jj * a.ns1] & NEIGHMASK;
  jtype = a.map[a.type[gj]];
  dx = x[gj * a.xs0] - xi0;
  dy = x[gj * a.xs0 + a.xs1] - xi1;
  dz = x[gj * a.xs0 + 2 * a.xs1] - xi2;
  const T rsq = dx * dx + dy * dy + dz * dz;
  return rsq < rcutsq[itype * a.cs0 + jtype * a.cs1];
}

template <typename T>
__global__ void __launch_bounds__(256)
    neighbor_count_kernel(int *__restrict__ numij, const T *__restrict__ x,
                          const T *__restrict__ rcutsq, const NeighArgs a, const int gi1,
                          const int Ni)
{
  const int i = blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
  const int lane = threadIdx.x & 63;
  if ((blockIdx.x == 0) && (threadIdx.x == 0)) numij[0] = 0;
  if (i >= Ni) return;
  const int gi = a.ilist[gi1 + i];
  const int itype = a.map[a.type[gi]];
  const T xi0 = x[gi * a.xs0], xi1 = x[gi * a.xs0 + a.xs1], xi2 = x[gi * a.xs0 + 2 * a.xs1];
  const int jnum = a.numneigh[gi];
  int count = 0;
  for (int j0 = 0; j0 < jnum; j0 += 64) {
    const int jj = j0 + lane;
    bool in = false;
    if (jj < jnum) {
      int gj, jtype;
      T dx, dy, dz;
      in = pod_neigh_test<T>(a, x, rcutsq, gi, itype, xi0, xi1, xi2, jj, gj, jtype, dx, dy, dz);
    }
    count += __popcll(__ballot(in));
  }
  if (lane == 0) numij[i + 1] = count;
}

template <typename T>
__global__ void __launch_bounds__(256)
    neighbor_list_kernel(T *__restrict__ rr, T *__restrict__ rhat, int *__restrict__ typeai,
                         int *__restrict__ idxi,
                         int *__restrict__ ai, int *__restrict__ aj, int *__restrict__ ti,
                         int *__restrict__ tj, const int *__restrict__ numij,
                         const T *__restrict__ x, const T *__restrict__ rcutsq,
                         const NeighArgs a, const int gi1, const int Ni, const int Nij)
{
  const int i = blockIdx.x * (blockDim.x >> 6) + (threadIdx.x >> 6);
  const int lane = threadIdx.x & 63;
  if (i >= Ni) return;
  const int gi = a.ilist[gi1 + i];
  const int itype = a.map[a.type[gi]];
  if (lane == 0) typeai[i] = itype;
  const T xi0 = x[gi * a.xs0], xi1 = x[gi * a.xs0 + a.xs1], xi2 = x[gi * a.xs0 + 2 * a.xs1];
  const int jnum = a.numneigh[gi];
  const unsigned long long below = (1ull << lane) - 1ull;
  int offset = numij[i];
  for (int j0 = 0; j0 < jnum; j0 += 64) {
    const int jj = j0 + lane;
    bool in = false;
    int gj = 0, jtype = 0;
    T dx = T(0), dy = T(0), dz = T(0);
    if (jj < jnum)
      in = pod_neigh_test<T>(a, x, rcutsq, gi, itype, xi0, xi1, xi2, jj, gj, jtype, dx, dy, dz);
    const unsigned long long mask = __ballot(in);
    if (in) {
      const int n = offset + __popcll(mask & below);
      // as in the Kokkos version of PairPODKokkos::NeighborList()
      const T r = sqrt(dx * dx + dy * dy + dz * dz);
      const T sinv = static_cast<T>(1.0) / r;
      rr[n] = r;
      rhat[n] = dx * sinv;
      rhat[n + Nij] = dy * sinv;
      rhat[n + 2 * Nij] = dz * sinv;
      idxi[n] = i;
      ai[n] = gi;
      aj[n] = gj;
      ti[n] = itype;
      tj[n] = jtype;
    }
    offset += __popcll(mask);
  }
}

// inclusive scan of numij[1..n] in place, in three passes; aux needs
// ceil(n / 1024) + 1 ints

static constexpr int SCAN_BLOCK = 1024;

static __device__ inline int block_inclusive_scan(int v, int *wsum)
{
  const int lane = threadIdx.x & 63, wid = threadIdx.x >> 6;
  for (int off = 1; off < 64; off <<= 1) {
    const int t = __shfl_up(v, off);
    if (lane >= off) v += t;
  }
  if (lane == 63) wsum[wid] = v;
  __syncthreads();
  if (wid == 0) {
    int w = (lane < (int) (blockDim.x >> 6)) ? wsum[lane] : 0;
    for (int off = 1; off < 64; off <<= 1) {
      const int t = __shfl_up(w, off);
      if (lane >= off) w += t;
    }
    if (lane < (int) (blockDim.x >> 6)) wsum[lane] = w;
  }
  __syncthreads();
  if (wid > 0) v += wsum[wid - 1];
  return v;
}

static __global__ void __launch_bounds__(SCAN_BLOCK) scan_pass1(int *__restrict__ a, int *__restrict__ aux,
                                                         const int n)
{
  __shared__ int wsum[SCAN_BLOCK / 64];
  const int k = blockIdx.x * SCAN_BLOCK + threadIdx.x;
  int v = (k < n) ? a[k + 1] : 0;
  v = block_inclusive_scan(v, wsum);
  if (k < n) a[k + 1] = v;
  if (threadIdx.x == SCAN_BLOCK - 1) aux[blockIdx.x] = v;
}

static __global__ void __launch_bounds__(SCAN_BLOCK) scan_pass2(int *__restrict__ aux, const int nb)
{
  __shared__ int wsum[SCAN_BLOCK / 64];
  __shared__ int total;
  int carry = 0;
  for (int b0 = 0; b0 < nb; b0 += SCAN_BLOCK) {
    const int k = b0 + threadIdx.x;
    int v = (k < nb) ? aux[k] : 0;
    v = block_inclusive_scan(v, wsum) + carry;
    if (k < nb) aux[k] = v;
    if (threadIdx.x == SCAN_BLOCK - 1) total = v;
    __syncthreads();
    carry = total;
    __syncthreads();
  }
}

static __global__ void __launch_bounds__(SCAN_BLOCK) scan_pass3(int *__restrict__ a,
                                                         const int *__restrict__ aux, const int n)
{
  const int k = blockIdx.x * SCAN_BLOCK + threadIdx.x;
  if ((blockIdx.x > 0) && (k < n)) a[k + 1] += aux[blockIdx.x - 1];
}

// returns the number of pairs, or -1 if a launch failed
template <typename T>
int neighbor_count(hipStream_t stream, int *numij, int *aux, const T *x, const T *rcutsq,
                   const NeighArgs &a, int gi1, int Ni)
{
  bool ok = launch(neighbor_count_kernel<T>, nblocks_for(Ni, 4), 256, 0, stream, numij, x, rcutsq,
                   a, gi1, Ni);
  const int nb = nblocks_for(Ni, SCAN_BLOCK);
  ok = ok && launch(scan_pass1, nb, SCAN_BLOCK, 0, stream, numij, aux, Ni);
  if (nb > 1) {
    ok = ok && launch(scan_pass2, 1, SCAN_BLOCK, 0, stream, aux, nb);
    ok = ok && launch(scan_pass3, nb, SCAN_BLOCK, 0, stream, numij, aux, Ni);
  }
  if (!ok) return -1;
  int total = 0;
  if (hipMemcpyAsync(&total, numij + Ni, sizeof(int), hipMemcpyDeviceToHost, stream) !=
      hipSuccess)
    return -1;
  if (hipStreamSynchronize(stream) != hipSuccess) return -1;
  return total;
}

template <typename T>
bool neighbor_list(hipStream_t stream, T *rr, T *rhat, int *typeai, int *idxi, int *ai, int *aj,
                   int *ti, int *tj, const int *numij, const T *x, const T *rcutsq,
                   const NeighArgs &a, int gi1, int Ni, int Nij)
{
  return launch(neighbor_list_kernel<T>, nblocks_for(Ni, 4), 256, 0, stream, rr, rhat, typeai,
                idxi, ai, aj, ti, tj, numij, x, rcutsq, a, gi1, Ni, Nij);
}

/* ----------------------------------------------------------------------
   neighbor_count() and neighbor_list() in one pass, without reading the
   number of pairs back to the host.  one wavefront per atom, four atoms per
   block: each wavefront counts the pairs of its atom inside the cutoff,
   the block finds its offset with a single-pass scan over the blocks
   (decoupled look-back, blocks numbered in the order they start), and the
   wavefronts write their pairs.  the result is the same as that of the two
   kernels and the scan: numij[i] and the pairs of each atom in the same order.
   the pairs of the first two 64-entry chunks of the neighbor list are kept
   in registers between the two passes; longer lists test the rest again.
   rhat is stored with the stride N, an upper bound of the number of pairs.
   status[] (one word per block) and *ticket must be zero at launch.
------------------------------------------------------------------------- */

static constexpr unsigned long long NB_AGG = 1ull << 32;     // status: aggregate of the block
static constexpr unsigned long long NB_PRE = 2ull << 32;     // status: inclusive prefix
// the status words carry only counts and publish no other data, so relaxed atomics suffice
// (acquire loads would invalidate the L2 cache on every poll)

template <typename T>
__global__ void __launch_bounds__(256)
    neighbor_fused_kernel(T *__restrict__ rr, T *__restrict__ rhat, int *__restrict__ typeai,
                          int *__restrict__ idxi, int *__restrict__ ai, int *__restrict__ aj,
                          int *__restrict__ ti, int *__restrict__ tj, int *__restrict__ numij,
                          unsigned long long *__restrict__ status, int *__restrict__ ticket,
                          const T *__restrict__ x, const T *__restrict__ rcutsq, const NeighArgs a,
                          const int gi1, const int Ni, const int N)
{
  __shared__ int bid_s, cnt_s[4], base_s;
  if (threadIdx.x == 0) bid_s = atomicAdd(ticket, 1);
  __syncthreads();
  const int bid = bid_s;
  const int w = threadIdx.x >> 6, lane = threadIdx.x & 63;
  const int i = 4 * bid + w;

  int gi = 0, itype = 0, jnum = 0;
  T xi0 = T(0), xi1 = T(0), xi2 = T(0);
  if (i < Ni) {
    gi = a.ilist[gi1 + i];
    itype = a.map[a.type[gi]];
    xi0 = x[gi * a.xs0];
    xi1 = x[gi * a.xs0 + a.xs1];
    xi2 = x[gi * a.xs0 + 2 * a.xs1];
    jnum = a.numneigh[gi];
  }

  // pass 1: count, keeping the first two chunks
  bool in0 = false, in1 = false;
  int gj0 = 0, jt0 = 0, gj1 = 0, jt1 = 0;
  T dx0 = T(0), dy0 = T(0), dz0 = T(0), dx1 = T(0), dy1 = T(0), dz1 = T(0);
  int count = 0;
  for (int j0 = 0; j0 < jnum; j0 += 64) {
    const int jj = j0 + lane;
    bool in = false;
    int gj = 0, jtype = 0;
    T dx = T(0), dy = T(0), dz = T(0);
    if (jj < jnum)
      in = pod_neigh_test<T>(a, x, rcutsq, gi, itype, xi0, xi1, xi2, jj, gj, jtype, dx, dy, dz);
    count += __popcll(__ballot(in));
    if (j0 == 0) {
      in0 = in;
      gj0 = gj;
      jt0 = jtype;
      dx0 = dx;
      dy0 = dy;
      dz0 = dz;
    } else if (j0 == 64) {
      in1 = in;
      gj1 = gj;
      jt1 = jtype;
      dx1 = dx;
      dy1 = dy;
      dz1 = dz;
    }
  }
  if (lane == 0) cnt_s[w] = count;
  __syncthreads();

  // offset of the block: look back over the blocks that started before it
  if (w == 0) {
    const int agg = cnt_s[0] + cnt_s[1] + cnt_s[2] + cnt_s[3];
    if (lane == 0)
      __hip_atomic_store(&status[bid], (bid == 0 ? NB_PRE : NB_AGG) | (unsigned) agg,
                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    int excl = 0;
    for (int p = bid - 1; p >= 0; p -= 64) {
      unsigned long long st;
      unsigned long long invalid, pre;
      int first;
      do {
        const int q = p - lane;
        st = (q >= 0) ? __hip_atomic_load(&status[q], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)
                      : NB_PRE;
        pre = __ballot((st >> 32) == 2);
        invalid = __ballot((st >> 32) == 0);
        first = pre ? (__ffsll((long long) pre) - 1) : 63;
        // wait until the blocks up to the nearest one with a prefix have published
        if (!(invalid & ((first >= 63) ? ~0ull : ((2ull << first) - 1ull)))) break;
        __builtin_amdgcn_s_sleep(1);
      } while (true);
      int v = (lane <= first) ? (int) (st & 0xffffffffull) : 0;
      for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off);
      excl += v;
      if (pre) break;
    }
    if (lane == 0) {
      if (bid > 0)
        __hip_atomic_store(&status[bid], NB_PRE | (unsigned) (excl + agg), __ATOMIC_RELAXED,
                           __HIP_MEMORY_SCOPE_AGENT);
      base_s = excl;
    }
  }
  __syncthreads();
  if (i >= Ni) return;
  int offset = base_s;
  for (int q = 0; q < w; q++) offset += cnt_s[q];
  if (lane == 0) {
    numij[i + 1] = offset + count;
    typeai[i] = itype;
    if (i == 0) numij[0] = 0;
  }

  // pass 2: write the pairs
  const unsigned long long below = (1ull << lane) - 1ull;
  for (int j0 = 0; j0 < jnum; j0 += 64) {
    const int jj = j0 + lane;
    bool in;
    int gj, jtype;
    T dx, dy, dz;
    if (j0 == 0) {
      in = in0;
      gj = gj0;
      jtype = jt0;
      dx = dx0;
      dy = dy0;
      dz = dz0;
    } else if (j0 == 64) {
      in = in1;
      gj = gj1;
      jtype = jt1;
      dx = dx1;
      dy = dy1;
      dz = dz1;
    } else {
      in = false;
      gj = jtype = 0;
      dx = dy = dz = T(0);
      if (jj < jnum)
        in = pod_neigh_test<T>(a, x, rcutsq, gi, itype, xi0, xi1, xi2, jj, gj, jtype, dx, dy, dz);
    }
    const unsigned long long mask = __ballot(in);
    if (in) {
      const int n = offset + __popcll(mask & below);
      // as in neighbor_list_kernel
      const T r = sqrt(dx * dx + dy * dy + dz * dz);
      const T sinv = static_cast<T>(1.0) / r;
      rr[n] = r;
      rhat[n] = dx * sinv;
      rhat[n + N] = dy * sinv;
      rhat[n + 2 * N] = dz * sinv;
      idxi[n] = i;
      ai[n] = gi;
      aj[n] = gj;
      ti[n] = itype;
      tj[n] = jtype;
    }
    offset += __popcll(mask);
  }
}

// scratch: lookback needs nblocks + 1 words of 8 bytes (the last holds the ticket)
inline size_t neighbor_fused_scratch(int Ni)
{
  return ((size_t) (Ni + 3) / 4 + 1) * sizeof(unsigned long long);
}

template <typename T>
bool neighbor_fused(hipStream_t stream, T *rr, T *rhat, int *typeai, int *idxi, int *ai, int *aj,
                    int *ti, int *tj, int *numij, void *lookback, const T *x, const T *rcutsq,
                    const NeighArgs &a, int gi1, int Ni, int N)
{
  if (Ni <= 0) return false;
  const int nb = (Ni + 3) / 4;
  auto *status = reinterpret_cast<unsigned long long *>(lookback);
  if (hipMemsetAsync(lookback, 0, neighbor_fused_scratch(Ni), stream) != hipSuccess) return false;
  return launch(neighbor_fused_kernel<T>, nb, 256, 0, stream, rr, rhat, typeai, idxi, ai, aj, ti,
                tj, numij, status, reinterpret_cast<int *>(status + nb), x, rcutsq, a, gi1, Ni, N);
}

/* ----------------------------------------------------------------------
   the pairs of the block with a fixed stride: the pairs of atom i inside the
   cutoff, in the order of its neighbor list, go to i*ps .. i*ps + numij[i+1]-1
   (ps >= the largest neighbor count), so that every wavefront writes its
   pairs without waiting for the counts of the other atoms.  one wavefront per
   atom, four atoms per block.  idxi, ai and ti are not written (the consumers
   know the atom of a pair); tj is written only with more than one element.
   with d2 != nullptr the B-spline 2-body descriptors of the atom are also
   summed (as twobodydesc_bspline_kernel) and written to d2[i*da + dm*t].
   HALF: two atoms per wavefront, 32 lanes each (most lists are shorter than 64).
------------------------------------------------------------------------- */

// HF: the sizes of the Hf model (one element, all atom types element 0, 40 B-splines summed
// into the per-atom rows of 66 descriptors) as constants
template <typename T, typename TA, bool HALF, bool HF>
__global__ void __launch_bounds__(256)
    neighbor_fixed_kernel(T *__restrict__ rr, T *__restrict__ rhat, int *__restrict__ typeai,
                          int *__restrict__ aj, int *__restrict__ ti, int *__restrict__ tj,
                          int *__restrict__ numij, const T *__restrict__ x,
                          const T *__restrict__ rcutsq, const NeighArgs a, const int gi1,
                          const int Ni, const int N, const int ps, const int ne_in,
                          T *__restrict__ d2, const BsplineArgs<T> bs_in, const int nrbf2_in,
                          const int da_in, const int dm_in, const int one_in)
{
  BsplineArgs<T> bs = bs_in;
  if constexpr (HF) {
    bs.ne = 1;
    bs.gmax = PodHf::nrbf2 - 4;
  }
  const int ne = HF ? 1 : ne_in, nrbf2 = HF ? PodHf::nrbf2 : nrbf2_in;
  const int da = HF ? PodHf::nDes : da_in, dm = HF ? 1 : dm_in, one = HF ? 1 : one_in;
  extern __shared__ char lds_raw[];
  constexpr int LPA = HALF ? 32 : 64;    // lanes per atom
  const int slot = threadIdx.x / LPA;    // atom of the block
  const int i = blockIdx.x * (blockDim.x / LPA) + slot;
  const int lane = threadIdx.x & (LPA - 1);
  const int hsel = (threadIdx.x & 63) / LPA;    // half of the wavefront
  const int nd = ne * nrbf2;
  TA *s_s = reinterpret_cast<TA *>(lds_raw) + slot * nd;    // [e][m] of atom i
  if (d2) {
    for (int t = lane; t < nd; t += LPA) s_s[t] = TA(0);
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
  }
  if (i >= Ni) return;
  const int gi = a.ilist[gi1 + i];
  // one != 0: every atom type is element 0, the types need not be read
  const int itype = one ? 0 : a.map[a.type[gi]];
  const T rc0 = one ? rcutsq[0] : T(0);
  const T xi0 = x[gi * a.xs0];
  const T xi1 = x[gi * a.xs0 + a.xs1];
  const T xi2 = x[gi * a.xs0 + 2 * a.xs1];
  const int jnum = a.numneigh[gi];
  const unsigned long long below = (1ull << lane) - 1ull;
  int offset = i * ps;
  for (int j0 = 0; j0 < jnum; j0 += LPA) {
    const int jj = j0 + lane;
    bool in = false;
    int gj = 0, jtype = 0;
    T dx = T(0), dy = T(0), dz = T(0);
    if (jj < jnum) {
      if (one) {
        gj = a.neighbors[gi * a.ns0 + jj * a.ns1] & NEIGHMASK;
        dx = x[gj * a.xs0] - xi0;
        dy = x[gj * a.xs0 + a.xs1] - xi1;
        dz = x[gj * a.xs0 + 2 * a.xs1] - xi2;
        in = dx * dx + dy * dy + dz * dz < rc0;
      } else
        in = pod_neigh_test<T>(a, x, rcutsq, gi, itype, xi0, xi1, xi2, jj, gj, jtype, dx, dy, dz);
    }
    unsigned long long mask = __ballot(in);
    if (HALF) mask = (mask >> (32 * hsel)) & 0xffffffffull;
    if (in) {
      const int n = offset + __popcll(mask & below);
      // as in neighbor_list_kernel
      const T r = sqrt(dx * dx + dy * dy + dz * dz);
      const T sinv = static_cast<T>(1.0) / r;
      rr[n] = r;
      rhat[n] = dx * sinv;
      rhat[n + N] = dy * sinv;
      rhat[n + 2 * N] = dz * sinv;
      aj[n] = gj;
      if (ne > 1) {
        ti[n] = itype;
        tj[n] = jtype;
      }
      if (d2) {
        T Bq[4], dBq[4];
        TA *se = s_s + pod_bspline_eval(bs, (ne > 1) ? itype * ne + jtype : 0, r, Bq, dBq) +
            nrbf2 * ((ne > 1) ? jtype : 0);
#pragma unroll
        for (int q = 0; q < 4; q++) atomicAdd(se + q, static_cast<TA>(Bq[q]));
      }
    }
    offset += __popcll(mask);
  }
  if (lane == 0) {
    numij[i + 1] = offset - i * ps;
    typeai[i] = itype;
  }
  if (d2) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup");
    for (int t = lane; t < nd; t += LPA)
      d2[(long) i * da + (long) dm * t] = static_cast<T>(s_s[t]);
  }
}

template <typename T, typename TA>
bool neighbor_fixed(hipStream_t stream, T *rr, T *rhat, int *typeai, int *aj, int *ti, int *tj,
                    int *numij, const T *x, const T *rcutsq, const NeighArgs &a, int gi1, int Ni,
                    int N, int ps, int ne, T *d2 = nullptr, const BsplineArgs<T> &bs = {},
                    int nrbf2 = 0, int da = 0, int dm = 0, int one = 0, bool half = false,
                    bool hf = false)
{
  if (Ni <= 0) return false;
  const int apb = half ? 8 : 4;    // atoms per block
  const size_t lds = d2 ? (size_t) apb * ne * nrbf2 * sizeof(TA) : 0;
  if (lds > 32768) return false;
  // the Hf instance: one element and type, B-spline descriptors in the per-atom rows
  if (hf && half && d2 && one && (ne == 1) && (nrbf2 == PodHf::nrbf2) && (bs.gmax == nrbf2 - 4) &&
      (da == PodHf::nDes) && (dm == 1))
    return launch(neighbor_fixed_kernel<T, TA, true, true>, nblocks_for(Ni, apb), 256, lds, stream,
                  rr, rhat, typeai, aj, ti, tj, numij, x, rcutsq, a, gi1, Ni, N, ps, ne, d2, bs,
                  nrbf2, da, dm, one);
  if (half)
    return launch(neighbor_fixed_kernel<T, TA, true, false>, nblocks_for(Ni, apb), 256, lds,
                  stream, rr, rhat, typeai, aj, ti, tj, numij, x, rcutsq, a, gi1, Ni, N, ps, ne,
                  d2, bs, nrbf2, da, dm, one);
  return launch(neighbor_fixed_kernel<T, TA, false, false>, nblocks_for(Ni, apb), 256, lds,
                stream, rr, rhat, typeai, aj, ti, tj, numij, x, rcutsq, a, gi1, Ni, N, ps, ne, d2,
                bs, nrbf2, da, dm, one);
}

// maximum number of neighbors of the atoms in ilist, or -1 on failure

static __global__ void __launch_bounds__(256) max_neighs_kernel(int *__restrict__ out,
                                                                const int *__restrict__ ilist,
                                                                const int *__restrict__ numneigh,
                                                                const int inum)
{
  __shared__ int sh[4];
  int v = 0;
  for (int ii = blockIdx.x * blockDim.x + threadIdx.x; ii < inum; ii += gridDim.x * blockDim.x)
    v = max(v, numneigh[ilist[ii]]);
  for (int off = 32; off > 0; off >>= 1) v = max(v, __shfl_xor(v, off));
  if ((threadIdx.x & 63) == 0) sh[threadIdx.x >> 6] = v;
  __syncthreads();
  if (threadIdx.x == 0) atomicMax(out, max(max(sh[0], sh[1]), max(sh[2], sh[3])));
}

inline int max_neighs(hipStream_t stream, int *dscratch, const int *ilist, const int *numneigh,
                      int inum)
{
  if (hipMemsetAsync(dscratch, 0, sizeof(int), stream) != hipSuccess) return -1;
  if (!launch(max_neighs_kernel, min(nblocks_for(inum, 256), 1024), 256, 0, stream, dscratch,
              ilist, numneigh, inum))
    return -1;
  int m = 0;
  if (hipMemcpyAsync(&m, dscratch, sizeof(int), hipMemcpyDeviceToHost, stream) != hipSuccess)
    return -1;
  if (hipStreamSynchronize(stream) != hipSuccess) return -1;
  return m;
}

/* ----------------------------------------------------------------------
   Hermite-spline and B-spline radial bases, one thread per pair
------------------------------------------------------------------------- */

template <typename T, typename TA, bool PERATOM>
__global__ void __launch_bounds__(256)
    radialbasis_hermite_kernel(T *__restrict__ rbf, T *__restrict__ drbf,
                               const PairGeom<T> pg, const int *__restrict__ ti,
                               const int *__restrict__ tj, const T *__restrict__ r0v,
                               const T *__restrict__ invdrv, const T *__restrict__ coef,
                               const int ne, const int nb, const int nr, const int N,
                               const int *__restrict__ numij, T *__restrict__ d2,
                               const int nrbf2, const int Ni, const int da, const int dm)
{
  __shared__ TA d2_s[PERATOM ? 256 : 1];    // PERATOM: 2-body descriptors of atom i, [e][m]
  if (PERATOM) {
    for (int t = threadIdx.x; t < ne * nrbf2; t += blockDim.x) d2_s[t] = TA(0);
    __syncthreads();
  }
  auto pair = [&](const int n) {
  const int pair = (ne > 1) ? ti[n] * ne + tj[n] : 0;
  const T r = pg.r(n);
  const T r0 = r0v[pair];
  const T invdr = invdrv[pair];
  const T invdrr = invdr / r;
  const T tg = (r - r0) * invdr;
  int b = (int) tg;
  if (b < 0) b = 0;
  else if (b > nb - 1) b = nb - 1;
  const T t = tg - (T) b;
  const size_t cbase = ((size_t) pair * (size_t) nb + (size_t) b) * (size_t) nr * 4ull;
  // the 4 coefficients of a function are contiguous and 4*sizeof(T) aligned
  typedef T vec2 __attribute__((ext_vector_type(2)));
  const bool aligned = (reinterpret_cast<size_t>(coef) % (2 * sizeof(T)) == 0);
  for (int k = 0; k < nr; ++k) {
    const size_t off = cbase + (size_t) 4 * k;
    T c0, c1, c2, c3;
    if (aligned) {
      const vec2 lo = *reinterpret_cast<const vec2 *>(coef + off);
      const vec2 hi = *reinterpret_cast<const vec2 *>(coef + off + 2);
      c0 = lo.x;
      c1 = lo.y;
      c2 = hi.x;
      c3 = hi.y;
    } else {
      c0 = coef[off + 0];
      c1 = coef[off + 1];
      c2 = coef[off + 2];
      c3 = coef[off + 3];
    }
    const T f = c0 + t * (c1 + t * (c2 + t * c3));
    const T dfdt = c1 + t * (static_cast<T>(2.0) * c2 + static_cast<T>(3.0) * c3 * t);
    rbf[n + N * k] = f;
    drbf[n + N * k] = dfdt * invdrr;
    if (PERATOM && (k < nrbf2))
      atomicAdd(d2_s + ((ne > 1) ? tj[n] : 0) * nrbf2 + k, static_cast<TA>(f));
  }
  };

  if (PERATOM) {
    const int i = blockIdx.x;
    int start, nj;
    pg.range(numij, i, start, nj);
    for (int n = start + threadIdx.x; n < start + nj; n += blockDim.x) pair(n);
    __syncthreads();
    for (int t = threadIdx.x; t < ne * nrbf2; t += blockDim.x)
      d2[(long) i * da + (long) dm * t] = static_cast<T>(d2_s[t]);
  } else {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) pair(n);
  }
}


template <typename T>
__global__ void __launch_bounds__(256)
    radialbasis_bspline_kernel(T *__restrict__ rbf2, T *__restrict__ drbf2,
                               int *__restrict__ base, const PairGeom<T> pg,
                               const BsplineArgs<T> g, const int N)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  T Bq[4], dBq[4];
  base[n] = pod_bspline_pair(g, pg, n, Bq, dBq);
#pragma unroll
  for (int q = 0; q < 4; q++) {
    rbf2[n + q * N] = Bq[q];
    drbf2[n + q * N] = dBq[q];
  }
}

/* ----------------------------------------------------------------------
   two-body forces (plain and B-spline), one thread per pair, and the
   B-spline two-body descriptors with one block per atom and LDS sums
------------------------------------------------------------------------- */

template <typename T, typename TA>
__global__ void __launch_bounds__(256)
    twobody_forces_kernel(T *__restrict__ fij, const T *__restrict__ cb2,
                          const T *__restrict__ drbf2, const PairGeom<T> pg,
                          const int *__restrict__ base, const int *__restrict__ idxi,
                          const int *__restrict__ tj, const int nrbf2, const int nm,
                          const int Ni, const int Nij, const BsplineArgs<T> bs)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= Nij) return;
  TA cR = TA(0);
  if (bs.r0v) {
    // B-splines evaluated here instead of read from drbf2/base
    T Bq[4], dBq[4];
    const int ii = idxi[n] + Ni * (pod_bspline_pair(bs, pg, n, Bq, dBq) + nrbf2 * tj[n]);
#pragma unroll
    for (int m = 0; m < 4; ++m) cR += static_cast<TA>(cb2[ii + Ni * m] * dBq[m]);
  } else {
    // base == nullptr: all nm = nrbf2 radial functions; otherwise nm = 4 B-splines from base[n]
    const int ii = idxi[n] + Ni * ((base ? base[n] : 0) + nrbf2 * tj[n]);
    for (int m = 0; m < nm; ++m) cR += static_cast<TA>(cb2[ii + Ni * m] * drbf2[n + Nij * m]);
  }
  T u, v, w;
  pg.uvw(n, u, v, w);
  cR *= static_cast<TA>(pg.r(n));
  fij[3 * n + 0] += static_cast<T>(cR * static_cast<TA>(u));
  fij[3 * n + 1] += static_cast<T>(cR * static_cast<TA>(v));
  fij[3 * n + 2] += static_cast<T>(cR * static_cast<TA>(w));
}

template <typename T, typename TA>
__global__ void __launch_bounds__(64)
    twobodydesc_bspline_kernel(T *__restrict__ d2, const T *__restrict__ rbf2,
                               const int *__restrict__ base, const int *__restrict__ tj,
                               const int *__restrict__ numij, const int nelements,
                               const int nrbf2, const int Ni, const int Nij,
                               const BsplineArgs<T> bs, const PairGeom<T> pg, const int da,
                               const int dm)
{
  extern __shared__ char lds_raw[];
  TA *s_s = reinterpret_cast<TA *>(lds_raw);    // [e][m]
  const int i = blockIdx.x;
  int start, nj;
  pg.range(numij, i, start, nj);
  const int end = start + nj;
  const int nd = nelements * nrbf2;
  for (int t = threadIdx.x; t < nd; t += blockDim.x) s_s[t] = TA(0);
  __syncthreads();
  for (int n = start + threadIdx.x; n < end; n += blockDim.x) {
    if (bs.r0v) {
      // B-splines evaluated here instead of read from rbf2/base
      T Bq[4], dBq[4];
      TA *se = s_s + pod_bspline_pair(bs, pg, n, Bq, dBq) + nrbf2 * ((nelements > 1) ? tj[n] : 0);
#pragma unroll
      for (int q = 0; q < 4; q++) atomicAdd(se + q, static_cast<TA>(Bq[q]));
    } else {
      TA *se = s_s + base[n] + nrbf2 * ((nelements > 1) ? tj[n] : 0);
      for (int q = 0; q < 4; q++) atomicAdd(se + q, static_cast<TA>(rbf2[n + Nij * q]));
    }
  }
  __syncthreads();
  for (int t = threadIdx.x; t < nd; t += blockDim.x)
    d2[(long) i * da + (long) dm * t] = static_cast<T>(s_s[t]);
}

/* ----------------------------------------------------------------------
   cross descriptors (five- to seven-body) and their force coefficients
------------------------------------------------------------------------- */

template <typename T>
__global__ void __launch_bounds__(256)
    crossdesc_kernel(T *__restrict__ d12, const T *__restrict__ d1, const T *__restrict__ d2,
                     const int *__restrict__ ind1, const int *__restrict__ ind2, const int n12,
                     const int Ni)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n12 * Ni) return;
  const int n = idx % Ni, i = idx / Ni;
  d12[n + Ni * i] = d1[n + Ni * ind1[i]] * d2[n + Ni * ind2[i]];
}

template <typename T>
__global__ void __launch_bounds__(256)
    crossdesc_reduction_kernel(T *__restrict__ cb1, T *__restrict__ cb2,
                               const T *__restrict__ c12, const T *__restrict__ d1,
                               const T *__restrict__ d2, const int *__restrict__ ind1,
                               const int *__restrict__ ind2, const int n12, const int Ni)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n12 * Ni) return;
  const int n = idx % Ni, m = idx / Ni;
  const int m1 = n + Ni * ind1[m], m2 = n + Ni * ind2[m];
  const T c = c12[n + Ni * m];
  atomicAdd(cb1 + m1, c * d2[m2]);
  atomicAdd(cb2 + m2, c * d1[m1]);
}

/* ----------------------------------------------------------------------
   linear (non-environment) model: atomic energies and force coefficients
   fused, one thread per atom
------------------------------------------------------------------------- */

template <typename T>
__global__ void __launch_bounds__(256)
    base_coefficients_kernel(T *__restrict__ ei, T *__restrict__ cb, const T *__restrict__ B,
                             const T *__restrict__ cefs, const int *__restrict__ typeai,
                             const int nDes, const int nCoeff, const int Ni)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= Ni) return;
  const T *c = cefs + nCoeff * typeai[n];
  T e = c[0];
  for (int m = 0; m < nDes; m++) {
    e += c[1 + m] * B[n + Ni * m];
    cb[n + Ni * m] = c[1 + m];
  }
  ei[n] = e;
}

/* ----------------------------------------------------------------------
   environment-adaptive model (global clusters).  the kernels follow the
   Kokkos kernels in blockatom_environment_descriptors(); base_coefficients
   and base_env_coefficients are fused into one pass over (i, m).
------------------------------------------------------------------------- */

struct EnvArgs {
  int nCom, nCls, nDes, nCoeff;
};

template <typename T>
__global__ void __launch_bounds__(256)
    env_pca_kernel(T *__restrict__ pca, const T *__restrict__ B, const T *__restrict__ proj,
                   const int *__restrict__ typeai, const EnvArgs a, const int Ni)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Ni * a.nCom) return;
  const int i = idx % Ni, k = idx / Ni;
  const int typei = typeai[i];
  T sum = 0.0;
  for (int m = 0; m < a.nDes; m++) sum += proj[k + a.nCom * m + a.nCom * a.nDes * typei] * B[i + Ni * m];
  pca[i + Ni * k] = sum;
}

// inverse square distances, probabilities and atomic energies, one thread per atom
template <typename T>
__global__ void __launch_bounds__(256)
    env_prob_energy_kernel(T *__restrict__ ei, T *__restrict__ P, T *__restrict__ D,
                           T *__restrict__ sumD, const T *__restrict__ pca,
                           const T *__restrict__ B, const T *__restrict__ cent,
                           const T *__restrict__ cefs, const int *__restrict__ typeai,
                           const EnvArgs a, const int Ni)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Ni) return;
  const int typei = typeai[i];
  for (int j = 0; j < a.nCls; j++) {
    T sum = static_cast<T>(1e-20);
    for (int k = 0; k < a.nCom; k++) {
      const T c = cent[k + j * a.nCom + a.nCls * a.nCom * typei];
      const T p = pca[i + Ni * k];
      sum += (p - c) * (p - c);
    }
    D[i + Ni * j] = static_cast<T>(1.0) / sum;
  }
  T sum = 0;
  for (int j = 0; j < a.nCls; j++) sum += D[i + Ni * j];
  sumD[i] = sum;
  for (int j = 0; j < a.nCls; j++) P[i + Ni * j] = D[i + Ni * j] / sum;

  const int nc = a.nCoeff * typei;
  T e = cefs[nc];
  for (int k = 0; k < a.nCls; k++)
    for (int m = 0; m < a.nDes; m++) e += cefs[1 + m + a.nDes * k + nc] * B[i + Ni * m] * P[i + Ni * k];
  ei[i] = e;
}

template <typename T>
__global__ void __launch_bounds__(256)
    env_cp_kernel(T *__restrict__ cp, const T *__restrict__ B, const T *__restrict__ cefs,
                  const int *__restrict__ typeai, const EnvArgs a, const int Ni)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Ni * a.nCls) return;
  const int n = idx % Ni, k = idx / Ni;
  const int nc = a.nCoeff * typeai[n];
  T sum = 0;
  for (int m = 0; m < a.nDes; m++) sum += cefs[1 + m + k * a.nDes + nc] * B[n + Ni * m];
  cp[n + Ni * k] = sum;
}

template <typename T>
__global__ void __launch_bounds__(256)
    env_cb_kernel(T *__restrict__ cb, const T *__restrict__ P, const T *__restrict__ D,
                  const T *__restrict__ sumD, const T *__restrict__ cp, const T *__restrict__ pca,
                  const T *__restrict__ cent, const T *__restrict__ proj,
                  const T *__restrict__ cefs, const int *__restrict__ typeai, const EnvArgs a,
                  const int Ni)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Ni * a.nDes) return;
  const int i = idx % Ni, m = idx / Ni;
  const int typei = typeai[i];
  const int nCom = a.nCom, nCls = a.nCls, nDes = a.nDes;

  // base coefficients
  const int nc = a.nCoeff * typei;
  T base = 0.0;
  for (int k = 0; k < nCls; k++) base += cefs[1 + m + k * nDes + nc] * P[i + Ni * k];

  // environment part
  const T S1 = 1 / sumD[i];
  const T S2 = sumD[i] * sumD[i];
  T sum = 0.0;
  for (int j = 0; j < nCls; j++) {
    T dP_dB = 0.0;
    for (int k = 0; k < nCls; k++) {
      T dP_dD = -D[i + Ni * j] / S2;
      if (k == j) dP_dD += S1;
      T dD_dB = 0.0;
      const T D2 = 2 * D[i + Ni * k] * D[i + Ni * k];
      for (int n = 0; n < nCom; n++) {
        const T dD_dpca = D2 * (cent[n + k * nCom + nCls * nCom * typei] - pca[i + Ni * n]);
        dD_dB += dD_dpca * proj[n + m * nCom + nCom * nDes * typei];
      }
      dP_dB += dP_dD * dD_dB;
    }
    sum += cp[i + Ni * j] * dP_dB;
  }
  cb[i + Ni * m] = base + sum;
}

/* ----------------------------------------------------------------------
   environment-adaptive model with local (active) clusters, following
   blockatom_local_environment_descriptors(); the first two Kokkos kernels
   are fused into one thread per atom.
------------------------------------------------------------------------- */

struct LocalEnvArgs {
  int nCls, nDes, nCoeff, nActMax;
  int dpp;    // 1: wave sums of localenv_wave_kernel with DPP
};

template <typename T, typename TA>
__global__ void __launch_bounds__(256)
    localenv_pca_kernel(T *__restrict__ pca, int *__restrict__ ks, int *__restrict__ kn,
                        const T *__restrict__ B, const T *__restrict__ proj,
                        const T *__restrict__ ledges, const T *__restrict__ redges,
                        const int *__restrict__ typeai, const LocalEnvArgs a, const int Ni)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Ni) return;
  const int nCls = a.nCls, nDes = a.nDes, nActMax = a.nActMax;
  const int typei = typeai[i];
  const int ncdt = typei * nDes;
  const int ncct = typei * nCls;
  TA pcai = 0.0;
  for (int m = 0; m < nDes; ++m) pcai += static_cast<TA>(proj[ncdt + m] * B[i + Ni * m]);
  pca[i] = static_cast<T>(pcai);
  int lo = 0, hi = nCls;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (static_cast<TA>(redges[ncct + mid]) < pcai) lo = mid + 1;
    else hi = mid;
  }
  const int ksi = lo;
  hi = ksi + nActMax;
  if (hi > nCls) hi = nCls;
  lo++;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (static_cast<TA>(ledges[ncct + mid]) <= pcai) lo = mid + 1;
    else hi = mid;
  }
  ks[i] = ksi;
  kn[i] = lo - ksi;
}

// U may alias pca: pca[i] is read before U[i] is written by the same thread
template <typename T, typename TA>
__global__ void __launch_bounds__(256)
    localenv_atom_kernel(T *__restrict__ ei, T *__restrict__ D, T *__restrict__ dD_dpca,
                         T *U, const T *pca, const int *__restrict__ ks,
                         const int *__restrict__ kn, const T *__restrict__ B,
                         const T *__restrict__ cent, const T *__restrict__ cefs,
                         const T *__restrict__ invlcut2, const T *__restrict__ invrcut2,
                         const int *__restrict__ typeai, const LocalEnvArgs a, const int Ni)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= Ni) return;
  const int nCls = a.nCls, nDes = a.nDes;
  const int kl = ks[i], kni = kn[i], typei = typeai[i];
  const int nc = a.nCoeff * typei;
  const int kncct = kl + nCls * typei;
  const int knc = 1 + nc + kl * nDes;
  TA sumfDi = 0.0, S = 0.0;
  const T pcai = pca[i];
  for (int k = 0; k < kni; ++k) {
    const int kg = k + kncct;
    const T pc = pcai - cent[kg];
    const T inv_rcut2 = (pc >= static_cast<T>(0.0)) ? invrcut2[kg] : invlcut2[kg];
    // cluster_cutoff_hat()
    const T y2 = (pc * pc) * inv_rcut2;
    const T omy = static_cast<T>(1.0) - y2 * y2;
    const T omy2 = omy * omy;
    const T fcut = omy2 * omy2;
    const T dfcut = static_cast<T>(-8.0) * inv_rcut2 * y2 * omy2 * omy;
    const T invDk = static_cast<T>(1.0) / (pc * pc + static_cast<T>(1e-20));
    const T fDk = fcut * invDk;
    const T dfDkdpc = static_cast<T>(2.0) * pc * invDk * (dfcut - fDk);
    D[i + Ni * k] = fDk;
    dD_dpca[i + Ni * k] = dfDkdpc;
    sumfDi += static_cast<TA>(fDk);
    S += static_cast<TA>(dfDkdpc);
  }
  sumfDi = static_cast<TA>(1.0) / sumfDi;

  TA eival = static_cast<TA>(cefs[nc]);
  TA Tt = 0.0, A = 0.0;
  for (int k = 0; k < kni; ++k) {
    const int idc = knc + k * nDes;
    TA sumE = 0.0;
    for (int m = 0; m < nDes; ++m) sumE += static_cast<TA>(cefs[m + idc] * B[i + Ni * m]);
    const int idx = i + Ni * k;
    const T Pk = static_cast<T>(static_cast<TA>(D[idx]) * sumfDi);
    const T cpk = static_cast<T>(sumE * sumfDi);
    D[idx] = Pk;
    eival += sumE * static_cast<TA>(Pk);
    Tt += static_cast<TA>(cpk * Pk);
    A += static_cast<TA>(cpk * dD_dpca[idx]);
  }
  ei[i] = static_cast<T>(eival);
  U[i] = static_cast<T>(A - S * Tt);
}

// split versions of the two kernels above: each block handles 64 atoms with
// LP threads per atom, which split the loops over the Mdesc descriptors (the
// loads of B stay coalesced) and combine their partial sums in LDS.  this
// gives LP times more parallelism to the latency-bound loops.

static constexpr int LP = 4;          // threads per atom
static constexpr int LP_KMAX = 8;     // maximum number of active clusters per atom

template <typename T, typename TA>
__global__ void __launch_bounds__(256)
    localenv_pca_split_kernel(T *__restrict__ pca, int *__restrict__ ks, int *__restrict__ kn,
                              const T *__restrict__ B, const T *__restrict__ proj,
                              const T *__restrict__ ledges, const T *__restrict__ redges,
                              const int *__restrict__ typeai, const LocalEnvArgs a, const int Ni)
{
  __shared__ TA part_s[LP][64];
  const int la = threadIdx.x & 63, lp = threadIdx.x >> 6;
  const int i = blockIdx.x * 64 + la;
  const int nCls = a.nCls, nDes = a.nDes, nActMax = a.nActMax;
  const int typei = (i < Ni) ? typeai[i] : 0;
  TA part = 0.0;
  if (i < Ni) {
    const T *pr = proj + typei * nDes;
    for (int m = lp; m < nDes; m += LP) part += static_cast<TA>(pr[m] * B[i + Ni * m]);
  }
  part_s[lp][la] = part;
  __syncthreads();
  if ((lp != 0) || (i >= Ni)) return;
  TA pcai = part_s[0][la];
  for (int q = 1; q < LP; q++) pcai += part_s[q][la];
  pca[i] = static_cast<T>(pcai);
  const int ncct = typei * nCls;
  int lo = 0, hi = nCls;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (static_cast<TA>(redges[ncct + mid]) < pcai) lo = mid + 1;
    else hi = mid;
  }
  const int ksi = lo;
  hi = ksi + nActMax;
  if (hi > nCls) hi = nCls;
  lo++;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (static_cast<TA>(ledges[ncct + mid]) <= pcai) lo = mid + 1;
    else hi = mid;
  }
  ks[i] = ksi;
  kn[i] = lo - ksi;
}

template <typename T, typename TA>
__global__ void __launch_bounds__(256)
    localenv_atom_split_kernel(T *__restrict__ ei, T *__restrict__ D, T *__restrict__ dD_dpca,
                               T *U, const T *pca, const int *__restrict__ ks,
                               const int *__restrict__ kn, const T *__restrict__ B,
                               const T *__restrict__ cent, const T *__restrict__ cefs,
                               const T *__restrict__ invlcut2, const T *__restrict__ invrcut2,
                               const int *__restrict__ typeai, const LocalEnvArgs a,
                               const int Ni)
{
  __shared__ TA part_s[LP][LP_KMAX][64];
  const int la = threadIdx.x & 63, lp = threadIdx.x >> 6;
  const int i = blockIdx.x * 64 + la;
  const int nCls = a.nCls, nDes = a.nDes;
  int kl = 0, kni = 0, typei = 0;
  if (i < Ni) {
    kl = ks[i];
    kni = kn[i];
    typei = typeai[i];
  }
  const int nc = a.nCoeff * typei;
  const int knc = 1 + nc + kl * nDes;

  // partial sumE(k) = sum_m cefs(m, k) B(i, m) over this thread's share of m
  for (int k = 0; k < kni; ++k) {
    const T *c = cefs + knc + k * nDes;
    TA part = 0.0;
    for (int m = lp; m < nDes; m += LP) part += static_cast<TA>(c[m] * B[i + Ni * m]);
    part_s[lp][k][la] = part;
  }
  __syncthreads();
  if ((lp != 0) || (i >= Ni)) return;

  const int kncct = kl + nCls * typei;
  TA sumfDi = 0.0, S = 0.0;
  const T pcai = pca[i];
  for (int k = 0; k < kni; ++k) {
    const int kg = k + kncct;
    const T pc = pcai - cent[kg];
    const T inv_rcut2 = (pc >= static_cast<T>(0.0)) ? invrcut2[kg] : invlcut2[kg];
    const T y2 = (pc * pc) * inv_rcut2;
    const T omy = static_cast<T>(1.0) - y2 * y2;
    const T omy2 = omy * omy;
    const T fcut = omy2 * omy2;
    const T dfcut = static_cast<T>(-8.0) * inv_rcut2 * y2 * omy2 * omy;
    const T invDk = static_cast<T>(1.0) / (pc * pc + static_cast<T>(1e-20));
    const T fDk = fcut * invDk;
    const T dfDkdpc = static_cast<T>(2.0) * pc * invDk * (dfcut - fDk);
    D[i + Ni * k] = fDk;
    dD_dpca[i + Ni * k] = dfDkdpc;
    sumfDi += static_cast<TA>(fDk);
    S += static_cast<TA>(dfDkdpc);
  }
  sumfDi = static_cast<TA>(1.0) / sumfDi;

  TA eival = static_cast<TA>(cefs[nc]);
  TA Tt = 0.0, A = 0.0;
  for (int k = 0; k < kni; ++k) {
    TA sumE = part_s[0][k][la];
    for (int q = 1; q < LP; q++) sumE += part_s[q][k][la];
    const int idx = i + Ni * k;
    const T Pk = static_cast<T>(static_cast<TA>(D[idx]) * sumfDi);
    const T cpk = static_cast<T>(sumE * sumfDi);
    D[idx] = Pk;
    eival += sumE * static_cast<TA>(Pk);
    Tt += static_cast<TA>(cpk * Pk);
    A += static_cast<TA>(cpk * dD_dpca[idx]);
  }
  ei[i] = static_cast<T>(eival);
  U[i] = static_cast<T>(A - S * Tt);
}

template <typename T, typename TA>
__global__ void __launch_bounds__(256)
    localenv_cb_kernel(T *__restrict__ cb, const T *__restrict__ D, const T *__restrict__ U,
                       const int *__restrict__ ks, const int *__restrict__ kn,
                       const T *__restrict__ proj, const T *__restrict__ cefs,
                       const int *__restrict__ typeai, const LocalEnvArgs a, const int Ni)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Ni * a.nDes) return;
  const int i = idx % Ni, m = idx / Ni;
  const int kl = ks[i], kni = kn[i], typei = typeai[i];
  const int knc = 1 + m + kl * a.nDes + a.nCoeff * typei;
  TA sum = static_cast<TA>(U[i] * proj[m + a.nDes * typei]);
  for (int k = 0; k < kni; ++k) sum += static_cast<TA>(cefs[knc + k * a.nDes] * D[i + Ni * k]);
  cb[i + Ni * m] = static_cast<T>(sum);
}

// out[i + Ni*m] = in[i*M + m]: per-atom rows of M values to the layout [i + Ni*m]
template <typename T>
__global__ void __launch_bounds__(256)
    atom_to_strided_kernel(T *__restrict__ out, const T *__restrict__ in, const int Ni,
                           const int M)
{
  const long idx = (long) blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (long) Ni * M) return;
  const int i = (int) (idx % Ni), m = (int) (idx / Ni);
  out[idx] = in[(long) i * M + m];
}

/* ----------------------------------------------------------------------
   local EA-POD (localenv_pca_split, localenv_atom_split and localenv_cb) in
   one kernel, one wavefront per atom: the projection and the sums over the
   descriptors are wave sums, and the two binary searches for the active
   clusters are ballots over the (sorted) cluster edges.  with g.d3 != nullptr
   the 3- and 4-body descriptors are computed first from sumU, as desc34_kernel,
   and stay in LDS; the other descriptors are read from B.  the descriptors
   of atom i are B[i*sa + sm*m] and its coefficients go to cb[i*sa + sm*m].
   at most KM <= LP_KMAX active clusters.
------------------------------------------------------------------------- */

// PRE (nDes <= 128, nCls <= 64, used with KM = 4): the cluster tables of the element and the projection are
// loaded into registers at the start, independent of the descriptors, and the coefficients of
// the active clusters are loaded once for the sums and the force coefficients
// HF: the sizes of the Hf model (PodHf) as constants (descriptors stored per atom)
template <typename T, typename TA, int KM, bool PRE, bool HF>
__global__ void __launch_bounds__(64)
    localenv_wave_kernel(T *__restrict__ ei, T *__restrict__ cb, const T *__restrict__ B,
                         const DescArgs<T> g_in, const T *__restrict__ sumU,
                         const T *__restrict__ proj, const T *__restrict__ ledges,
                         const T *__restrict__ redges, const T *__restrict__ cent,
                         const T *__restrict__ cefs, const T *__restrict__ invlcut2,
                         const T *__restrict__ invrcut2, const int *__restrict__ typeai,
                         const LocalEnvArgs a_in, const int nl2_in, const int nl3_in,
                         const int sa_in, const int sm_in)
{
  DescArgs<T> g = g_in;
  LocalEnvArgs a = a_in;
  if constexpr (HF) {
    g.nelements = 1;
    g.nabf3_active = PodHf::nabf3_active;
    g.nrbf4 = PodHf::nrbf4;
    g.nabf4 = PodHf::nabf4;
    g.K3 = PodHf::K3;
    g.Q4 = PodHf::Q4;
    g.do4 = 1;
    a.nCls = PodHf::nCls;
    a.nDes = PodHf::nDes;
    a.nCoeff = PodHf::nCoeff;
    a.nActMax = PodHf::nActMax;
  }
  const int nl2 = HF ? PodHf::nl2 : nl2_in, nl3 = HF ? PodHf::nl3 : nl3_in;
  const int sa = HF ? PodHf::nDes : sa_in, sm = HF ? 1 : sm_in;
  extern __shared__ char lds_raw[];
  const int nDes = a.nDes, nCls = a.nCls;
  T *d_s = reinterpret_cast<T *>(lds_raw);    // descriptors of atom i
  T *u_s = d_s + nDes;                        // sumU of atom i: [m][q][e]
  const int i = blockIdx.x;
  const int lane = threadIdx.x;
  const long ib = (long) i * sa;
  const int typei = HF ? 0 : typeai[i];    // one element in the Hf model
  const int ncct = typei * nCls;
  const T *pr = proj + typei * nDes;

  // PRE: lane c holds the edges, centroid and cutoffs of cluster c; lane m (and m + 64) the
  // projection of descriptor m
  T re = T(0), le = T(0), ce = T(0), irc = T(0), ilc = T(0), pv[2] = {T(0), T(0)};
  if (PRE) {
    if (lane < nCls) {
      re = redges[ncct + lane];
      le = ledges[ncct + lane];
      ce = cent[ncct + lane];
      irc = invrcut2[ncct + lane];
      ilc = invlcut2[ncct + lane];
    }
#pragma unroll
    for (int q = 0; q < 2; q++)
      if (lane + 64 * q < nDes) pv[q] = pr[lane + 64 * q];
  }

  if (g.d3) {
    const int S = g.nelements * g.K3 * g.nrbf3;
    for (int t = lane; t < S; t += 64) u_s[t] = sumU[(long) S * i + t];
    for (int m = lane; m < nDes; m += 64) d_s[m] = (m < nl2) ? B[ib + (long) sm * m] : T(0);
    __syncthreads();
    DescArgs<T> gl = g;
    gl.d3 = d_s + nl2;
    gl.d4 = d_s + nl2 + nl3;
    gl.Ni = 1;
    desc34_atom<T, TA>(gl, u_s, 0);
  } else {
    for (int m = lane; m < nDes; m += 64) d_s[m] = B[ib + (long) sm * m];
  }
  __syncthreads();

  // projection and active clusters, as localenv_pca_split_kernel
  TA p = TA(0);
  if (PRE) {
#pragma unroll
    for (int q = 0; q < 2; q++)
      if (lane + 64 * q < nDes) p += static_cast<TA>(pv[q] * d_s[lane + 64 * q]);
  } else
    for (int m = lane; m < nDes; m += 64) p += static_cast<TA>(pr[m] * d_s[m]);
  const TA pcai = a.dpp ? wave_sum_dpp(p) : wave_sum(p);
  int kl = 0, kni = 1;
  if (PRE) {
    kl = __popcll(__ballot((lane < nCls) && (static_cast<TA>(re) < pcai)));
    const int hi = min(kl + a.nActMax, nCls);
    kni += __popcll(__ballot((lane > kl) && (lane < hi) && (static_cast<TA>(le) <= pcai)));
  } else {
    for (int c0 = 0; c0 < nCls; c0 += 64) {
      const int cc = c0 + lane;
      kl += __popcll(__ballot((cc < nCls) && (static_cast<TA>(redges[ncct + cc]) < pcai)));
    }
    const int hi = min(kl + a.nActMax, nCls);
    for (int c0 = kl + 1; c0 < hi; c0 += 64) {
      const int cc = c0 + lane;
      kni += __popcll(__ballot((cc < hi) && (static_cast<TA>(ledges[ncct + cc]) <= pcai)));
    }
  }

  // sumE(k) = sum_m cefs(m, k) B(m), as localenv_atom_split_kernel
  const int nc = a.nCoeff * typei;
  const int knc = 1 + nc + kl * nDes;
  TA sumE[KM];
  T cf[PRE ? KM : 1][2];
#pragma unroll
  for (int k = 0; k < KM; k++) sumE[k] = TA(0);
  if (PRE) {
#pragma unroll
    for (int k = 0; k < KM; k++)
#pragma unroll
      for (int q = 0; q < 2; q++) {
        const int m = lane + 64 * q;
        cf[PRE ? k : 0][q] = ((k < kni) && (m < nDes)) ? cefs[knc + k * nDes + m] : T(0);
      }
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int m = lane + 64 * q;
      const T b = (m < nDes) ? d_s[m] : T(0);
#pragma unroll
      for (int k = 0; k < KM; k++)
        if (k < kni) sumE[k] += static_cast<TA>(cf[PRE ? k : 0][q] * b);
    }
  } else
    for (int m = lane; m < nDes; m += 64) {
      const T b = d_s[m];
#pragma unroll
      for (int k = 0; k < KM; k++)
        if (k < kni) sumE[k] += static_cast<TA>(cefs[knc + k * nDes + m] * b);
    }
#pragma unroll
  for (int k = 0; k < KM; k++)
    if (k < kni) sumE[k] = a.dpp ? wave_sum_dpp(sumE[k]) : wave_sum(sumE[k]);

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
        ck = __shfl(ce, kl + k);
        irk = __shfl(irc, kl + k);
        ilk = __shfl(ilc, kl + k);
      } else {
        ck = cent[kg];
        irk = invrcut2[kg];
        ilk = invlcut2[kg];
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
  TA eival = static_cast<TA>(cefs[nc]);
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
  if (lane == 0) ei[i] = static_cast<T>(eival);

  // force coefficients of the descriptors, as localenv_cb_kernel
  const T Ui = static_cast<T>(A - S * Tt);
  if (PRE) {
#pragma unroll
    for (int q = 0; q < 2; q++) {
      const int m = lane + 64 * q;
      if (m < nDes) {
        TA sum = static_cast<TA>(Ui * pv[q]);
#pragma unroll
        for (int k = 0; k < KM; k++)
          if (k < kni) sum += static_cast<TA>(cf[PRE ? k : 0][q] * Pk[k]);
        cb[ib + (long) sm * m] = static_cast<T>(sum);
      }
    }
  } else
    for (int m = lane; m < nDes; m += 64) {
      const int kncm = 1 + m + kl * nDes + nc;
      TA sum = static_cast<TA>(Ui * pr[m]);
#pragma unroll
      for (int k = 0; k < KM; k++)
        if (k < kni) sum += static_cast<TA>(cefs[kncm + k * nDes] * Pk[k]);
      cb[ib + (long) sm * m] = static_cast<T>(sum);
    }
}

template <typename T, typename TA>
bool localenv_wave(hipStream_t stream, T *ei, T *cb, const T *B, const DescArgs<T> &g,
                   const T *sumU, const T *proj, const T *ledges, const T *redges, const T *cent,
                   const T *cefs, const T *invlcut2, const T *invrcut2, const int *typeai,
                   const LocalEnvArgs &a, int nl2, int nl3, int Ni, int sa, int sm,
                   bool pre = false, bool hf = false)
{
  if ((Ni <= 0) || (a.nActMax > LP_KMAX) || (a.nActMax < 1)) return false;
  size_t lds = (size_t) a.nDes * sizeof(T);
  if (g.d3) lds += (size_t) g.nelements * g.K3 * g.nrbf3 * sizeof(T);
  if (lds > 65536) return false;
#define LMP_POD_LENV_WAVE1(KMV, PREV, H)                                                        \
  return launch(localenv_wave_kernel<T, TA, KMV, PREV, H>, Ni, 64, lds, stream, ei, cb, B, g,    \
                sumU, proj, ledges, redges, cent, cefs, invlcut2, invrcut2, typeai, a, nl2, nl3, \
                sa, sm)
#define LMP_POD_LENV_WAVE(KMV, PREV) LMP_POD_LENV_WAVE1(KMV, PREV, false)
  // the preloaded tables only with up to 4 active clusters: with 8 the kernel needs 110
  // VGPRs (4 waves) instead of 78 (6 waves)
  if (a.nActMax <= 4) {
    if (hf && pre && g.d3) LMP_POD_LENV_WAVE1(4, true, true);
    if (pre && (a.nDes <= 128) && (a.nCls <= 64)) LMP_POD_LENV_WAVE(4, true);
    LMP_POD_LENV_WAVE(4, false);
  }
  LMP_POD_LENV_WAVE(LP_KMAX, false);
#undef LMP_POD_LENV_WAVE
#undef LMP_POD_LENV_WAVE1
}

/* ----------------------------------------------------------------------
   energy and virial tallies.  the global sums are accumulated on the device
   (acc[0] energy, acc[1..6] virial) and read once per step by the caller;
   the per-atom tallies are direct ports.
------------------------------------------------------------------------- */

template <typename T, typename TA>
__global__ void __launch_bounds__(256)
    tallyenergy_kernel(TA *__restrict__ acc, TA *__restrict__ eatom, const T *__restrict__ ei,
                       const int istart, const int Ni, const int global)
{
  __shared__ TA sh[4];
  TA e = TA(0);
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < Ni; k += gridDim.x * blockDim.x) {
    const TA ek = static_cast<TA>(ei[k]);
    if (eatom) eatom[istart + k] += ek;
    e += ek;
  }
  if (global) {
    const TA b = block_sum(e, sh);
    if (threadIdx.x == 0) atomicAdd(acc, b);
  }
}

// acc[1 + d] += sum over the atoms of the block of vir[6*i + d]
template <typename TA>
__global__ void __launch_bounds__(256)
    virial_reduce_kernel(TA *__restrict__ acc, const TA *__restrict__ vir, const int Ni)
{
  __shared__ TA sh[4];
  TA v[6] = {TA(0), TA(0), TA(0), TA(0), TA(0), TA(0)};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < Ni; i += gridDim.x * blockDim.x)
    for (int d = 0; d < 6; d++) v[d] += vir[6 * i + d];
  for (int d = 0; d < 6; d++) {
    const TA b = block_sum(v[d], sh);
    if (threadIdx.x == 0) atomicAdd(acc + 1 + d, b);
  }
}

template <typename TA> bool virial_reduce(hipStream_t stream, TA *acc, const TA *vir, int Ni)
{
  return launch(virial_reduce_kernel<TA>, std::min(nblocks_for(Ni, 256), 1024), 256, 0, stream,
                acc, vir, Ni);
}

template <typename T, typename TA>
__global__ void __launch_bounds__(256)
    tallystress_kernel(TA *__restrict__ acc, TA *__restrict__ vatom, const int vs0,
                       const int vs1, const T *__restrict__ fij, const PairGeom<T> pg,
                       const int *__restrict__ ai, const int *__restrict__ aj, const int Nij,
                       const int global)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  T rx = 0, ry = 0, rz = 0, fx = 0, fy = 0, fz = 0;
  if (k < Nij) {
    T u, v, w;
    pg.uvw(k, u, v, w);
    const T r = pg.r(k);
    rx = r * u;
    ry = r * v;
    rz = r * w;
    fx = fij[3 * k + 0];
    fy = fij[3 * k + 1];
    fz = fij[3 * k + 2];
  }
  if (global) {
    __shared__ TA sh[4];
    const TA v[6] = {static_cast<TA>(rx * fx), static_cast<TA>(ry * fy),
                     static_cast<TA>(rz * fz), static_cast<TA>(rx * fy),
                     static_cast<TA>(rx * fz), static_cast<TA>(ry * fz)};
    for (int d = 0; d < 6; d++) {
      const TA b = block_sum(v[d], sh);
      if (threadIdx.x == 0) atomicAdd(acc + 1 + d, b);
    }
  }
  if (vatom && (k < Nij)) {
    const T hx = static_cast<T>(-0.5) * rx, hy = static_cast<T>(-0.5) * ry,
            hz = static_cast<T>(-0.5) * rz;
    const T vl[6] = {hx * fx, hy * fy, hz * fz, hx * fy, hx * fz, hy * fz};
    TA *vi = vatom + ai[k] * vs0;
    TA *vj = vatom + aj[k] * vs0;
    for (int d = 0; d < 6; ++d) {
      atomicAdd(vi + d * vs1, static_cast<TA>(vl[d]));
      atomicAdd(vj + d * vs1, static_cast<TA>(vl[d]));
    }
  }
}

}    // namespace PODHIP
}    // namespace LAMMPS_NS

#endif    // KOKKOS_ENABLE_HIP
#endif
