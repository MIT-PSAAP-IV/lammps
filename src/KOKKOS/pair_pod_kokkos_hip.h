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

/* ----------------------------------------------------------------------
   sumU(e, k, m, i) = sum_j [tj(j) == e] rbf(j, m) * abf(j, k)

   one block per atom i; the neighbors of atom i are processed in tiles that
   are staged in LDS with coalesced loads, laid out [j][k] and [j][m] so that
   neighboring threads (consecutive k) read consecutive LDS words.  each
   thread owns one (k, m) output and accumulates all elements in registers.
------------------------------------------------------------------------- */

template <typename T, typename TA>
__global__ void radialangularsum_kernel(T *__restrict__ sumU, const T *__restrict__ rbf,
                                        const T *__restrict__ abf, const int *__restrict__ tj,
                                        const int *__restrict__ numij, const int nelements,
                                        const int nrbf3, const int K3, const int Nij,
                                        const int jtile)
{
  extern __shared__ char lds_raw[];
  T *abf_s = reinterpret_cast<T *>(lds_raw);    // [jtile][K3]
  T *rbf_s = abf_s + jtile * K3;                // [jtile][nrbf3]
  int *tj_s = reinterpret_cast<int *>(rbf_s + jtile * nrbf3);

  const int i = blockIdx.x;
  const int start = numij[i];
  const int nj = numij[i + 1] - start;
  const int nout = nrbf3 * K3;
  const int o = threadIdx.x;    // output index o = k + K3*m, as in the Kokkos kernel
  const int k = o % K3;
  const int m = o / K3;

  TA acc[MAX_ELEMENTS];
  for (int e = 0; e < MAX_ELEMENTS; e++) acc[e] = TA(0);

  for (int j0 = 0; j0 < nj; j0 += jtile) {
    const int nt = min(jtile, nj - j0);

    // coalesced loads along j, transposed into LDS
    for (int t = threadIdx.x; t < nt * K3; t += blockDim.x) {
      const int jj = t % nt, kk = t / nt;
      abf_s[jj * K3 + kk] = abf[start + j0 + jj + Nij * kk];
    }
    for (int t = threadIdx.x; t < nt * nrbf3; t += blockDim.x) {
      const int jj = t % nt, mm = t / nt;
      rbf_s[jj * nrbf3 + mm] = rbf[start + j0 + jj + Nij * mm];
    }
    if (nelements > 1)
      for (int t = threadIdx.x; t < nt; t += blockDim.x) tj_s[t] = tj[start + j0 + t];
    __syncthreads();

    if (o < nout) {
      if (nelements == 1) {
        for (int jj = 0; jj < nt; jj++)
          acc[0] += static_cast<TA>(rbf_s[jj * nrbf3 + m] * abf_s[jj * K3 + k]);
      } else {
        for (int jj = 0; jj < nt; jj++)
          acc[tj_s[jj]] += static_cast<TA>(rbf_s[jj * nrbf3 + m] * abf_s[jj * K3 + k]);
      }
    }
    __syncthreads();
  }

  if (o < nout) {
    const int base = nelements * (o + nout * i);
    for (int e = 0; e < nelements; e++) sumU[base + e] = static_cast<T>(acc[e]);
  }
}

template <typename T, typename TA>
bool radialangularsum(hipStream_t stream, T *sumU, const T *rbf, const T *abf, const int *tj,
                      const int *numij, int nelements, int nrbf3, int K3, int Ni, int Nij)
{
  const int nout = nrbf3 * K3;
  if ((Ni <= 0) || (nout > MAX_THREADS) || (nelements > MAX_ELEMENTS)) return false;

  const int bytes_per_j = (K3 + nrbf3) * sizeof(T) + sizeof(int);
  int jtile = LDS_BUDGET / bytes_per_j;
  if (jtile > 128) jtile = 128;
  if (jtile < 8) return false;

  const int threads = ((nout + 63) / 64) * 64;
  const size_t lds = (size_t) jtile * bytes_per_j;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(radialangularsum_kernel<T, TA>), dim3(Ni), dim3(threads), lds,
                     stream, sumU, rbf, abf, tj, numij, nelements, nrbf3, K3, Nij, jtile);
  return hipGetLastError() == hipSuccess;
}

/* ----------------------------------------------------------------------
   fij(j) += sum_k sum_m forcecoeff(tj, k, m, i) * d/dr [ rbf(j, m) abf(j, k) ]

   one block per atom i with one thread per neighbor j.  the forcecoeff block
   of atom i is staged in LDS once and shared by all its neighbors, and the
   radial functions of pair j are kept in registers instead of being reloaded
   for every angular function.
------------------------------------------------------------------------- */

template <typename T, typename TA>
__global__ void allbody_forces_kernel(T *__restrict__ fij, const T *__restrict__ forcecoeff,
                                      const T *__restrict__ rbf, const T *__restrict__ drbf,
                                      const T *__restrict__ rij, const T *__restrict__ abf,
                                      const T *__restrict__ abfx, const T *__restrict__ abfy,
                                      const T *__restrict__ abfz, const int *__restrict__ tj,
                                      const int *__restrict__ numij, const int nelements,
                                      const int nrbf3, const int K3, const int Nij)
{
  extern __shared__ char lds_raw[];
  T *fc_s = reinterpret_cast<T *>(lds_raw);    // forcecoeff of atom i: [m][k][e]

  const int i = blockIdx.x;
  const int start = numij[i];
  const int nj = numij[i + 1] - start;
  const int NeK3 = nelements * K3;
  const int nfc = NeK3 * nrbf3;

  for (int t = threadIdx.x; t < nfc; t += blockDim.x) fc_s[t] = forcecoeff[t + nfc * i];
  __syncthreads();

  for (int jj = threadIdx.x; jj < nj; jj += blockDim.x) {
    const int j = start + jj;
    const int e = (nelements > 1) ? tj[j] : 0;

    T r[MAX_RADIAL], dr[MAX_RADIAL];
    for (int m = 0; m < nrbf3; m++) {
      r[m] = rbf[j + Nij * m];
      dr[m] = drbf[j + Nij * m];
    }

    TA fcAdR = TA(0), fx = TA(0), fy = TA(0), fz = TA(0);
    for (int k = 0; k < K3; k++) {
      const T *fck = fc_s + e + nelements * k;
      TA fcR = TA(0), fcdR = TA(0);
      for (int m = 0; m < nrbf3; m++) {
        const T fc = fck[NeK3 * m];
        fcR += static_cast<TA>(fc * r[m]);
        fcdR += static_cast<TA>(fc * dr[m]);
      }
      const int idxA = j + Nij * k;
      fcAdR += fcdR * static_cast<TA>(abf[idxA]);
      fx += fcR * static_cast<TA>(abfx[idxA]);
      fy += fcR * static_cast<TA>(abfy[idxA]);
      fz += fcR * static_cast<TA>(abfz[idxA]);
    }

    const int i1 = 3 * j;
    fx += fcAdR * static_cast<TA>(rij[i1 + 0]);
    fy += fcAdR * static_cast<TA>(rij[i1 + 1]);
    fz += fcAdR * static_cast<TA>(rij[i1 + 2]);
    fij[i1 + 0] += static_cast<T>(fx);
    fij[i1 + 1] += static_cast<T>(fy);
    fij[i1 + 2] += static_cast<T>(fz);
  }
}

template <typename T, typename TA>
bool allbody_forces(hipStream_t stream, T *fij, const T *forcecoeff, const T *rbf, const T *drbf,
                    const T *rij, const T *abf, const T *abfx, const T *abfy, const T *abfz,
                    const int *tj, const int *numij, int nelements, int nrbf3, int K3, int Ni,
                    int Nij)
{
  if ((Ni <= 0) || (nrbf3 > MAX_RADIAL) || (nelements > MAX_ELEMENTS)) return false;
  const size_t lds = (size_t) nelements * K3 * nrbf3 * sizeof(T);
  if (lds > 65536) return false;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(allbody_forces_kernel<T, TA>), dim3(Ni), dim3(64), lds, stream,
                     fij, forcecoeff, rbf, drbf, rij, abf, abfx, abfy, abfz, tj, numij, nelements,
                     nrbf3, K3, Nij);
  return hipGetLastError() == hipSuccess;
}

/* ----------------------------------------------------------------------
   three- and four-body force coefficients, fused:

   fb(e, q, m, i) = sum over the 3-body and 4-body descriptors of
                    cb(descriptor, i) * d descriptor / d sumU(e, q, m, i)

   this replaces zeroing forcecoeff followed by threebody_forcecoeff() and
   fourbody_forcecoeff().  one block per atom i: sumU and the result of atom i
   live in LDS.  the 3-body pass assigns one (m, q) item per thread; items
   write disjoint entries, so no atomics are needed.  the 4-body pass assigns
   one (m, q) item per thread as well, but its three updates go to arbitrary
   entries of the m-row and use LDS atomics.
------------------------------------------------------------------------- */

template <typename T>
__global__ void forcecoeff34_kernel(T *__restrict__ fb, const T *__restrict__ cb3,
                                    const T *__restrict__ cb4, const T *__restrict__ sumU,
                                    const int *__restrict__ pc3, const int *__restrict__ pn3,
                                    const int *__restrict__ p3_active, const int *__restrict__ pa4,
                                    const int *__restrict__ pb4, const int *__restrict__ pc4,
                                    const int nelements, const int nrbf3, const int nabf3_active,
                                    const int nrbf4, const int nabf4, const int K3, const int Q4,
                                    const int Ni, const int do4)
{
  extern __shared__ char lds_raw[];
  const int S = nelements * K3 * nrbf3;
  T *u_s = reinterpret_cast<T *>(lds_raw);    // sumU of atom i: [m][q][e]
  T *f_s = u_s + S;                           // result of atom i, same layout
  int *qa_s = reinterpret_cast<int *>(f_s + S);    // active angular index a of q, or -1
  int *qp_s = qa_s + K3;                           // 4-body angular index p of each term q

  const int i = blockIdx.x;
  const int base = S * i;
  const int NeK3 = nelements * K3;

  for (int t = threadIdx.x; t < S; t += blockDim.x) {
    u_s[t] = sumU[base + t];
    f_s[t] = T(0);
  }
  for (int q = threadIdx.x; q < K3; q += blockDim.x) qa_s[q] = -1;
  __syncthreads();
  for (int a = threadIdx.x; a < nabf3_active; a += blockDim.x) {
    const int p = p3_active[a];
    for (int q = pn3[p]; q < pn3[p + 1]; q++) qa_s[q] = a;
  }
  if (do4)
    for (int p = threadIdx.x; p < nabf4; p += blockDim.x)
      for (int q = pa4[p]; q < pa4[p + 1]; q++) qp_s[q] = p;
  __syncthreads();

  // three-body: cb3 index i + Ni*(a + nabf3_active*m) + N3*(element pair)

  const int N3 = Ni * nabf3_active * nrbf3;
  for (int item = threadIdx.x; item < nrbf3 * K3; item += blockDim.x) {
    const int q = item % K3;
    const int m = item / K3;
    const int a = qa_s[q];
    if (a < 0) continue;
    const T pk = static_cast<T>(pc3[q]);
    const int iu = NeK3 * m + nelements * q;
    int em = i + Ni * (a + nabf3_active * m);
    for (int i1 = 0; i1 < nelements; i1++) {
      const T u1 = u_s[iu + i1];
      for (int i2 = i1; i2 < nelements; i2++, em += N3) {
        const T w = pk * cb3[em];
        f_s[iu + i2] += w * u1;
        f_s[iu + i1] += w * u_s[iu + i2];
      }
    }
  }
  __syncthreads();

  // four-body: cb4 index i + Ni*(p + nabf4*m) + N4*(element triple)

  if (do4) {
    const int N4 = Ni * nabf4 * nrbf4;
    for (int item = threadIdx.x; item < nrbf4 * Q4; item += blockDim.x) {
      const int q = item % Q4;
      const int m = item / Q4;
      const int p = qp_s[q];
      const T c = static_cast<T>(pc4[q]);
      const int iu = NeK3 * m;
      const int idx1 = iu + nelements * pb4[q];
      const int idx2 = iu + nelements * pb4[q + Q4];
      const int idx3 = iu + nelements * pb4[q + 2 * Q4];
      int k = i + Ni * (p + nabf4 * m);
      for (int i1 = 0; i1 < nelements; i1++) {
        const T c1 = u_s[idx1 + i1];
        for (int i2 = i1; i2 < nelements; i2++) {
          const T c2 = u_s[idx2 + i2];
          for (int i3 = i2; i3 < nelements; i3++, k += N4) {
            const T c3 = u_s[idx3 + i3];
            const T c4 = c * cb4[k];
            atomicAdd(&f_s[idx3 + i3], c4 * (c1 * c2));
            atomicAdd(&f_s[idx2 + i2], c4 * (c1 * c3));
            atomicAdd(&f_s[idx1 + i1], c4 * (c2 * c3));
          }
        }
      }
    }
    __syncthreads();
  }

  for (int t = threadIdx.x; t < S; t += blockDim.x) fb[base + t] = f_s[t];
}

template <typename T>
bool forcecoeff34(hipStream_t stream, T *fb, const T *cb3, const T *cb4, const T *sumU,
                  const int *pc3, const int *pn3, const int *p3_active, const int *pa4,
                  const int *pb4, const int *pc4, int nelements, int nrbf3, int nabf3_active,
                  int nrbf4, int nabf4, int K3, int Q4, int Ni, bool do4)
{
  if ((Ni <= 0) || (nelements > MAX_ELEMENTS)) return false;
  const size_t lds = 2 * (size_t) nelements * K3 * nrbf3 * sizeof(T) +
      ((size_t) K3 + (do4 ? Q4 : 0)) * sizeof(int);
  if (lds > 65536) return false;

  hipLaunchKernelGGL(HIP_KERNEL_NAME(forcecoeff34_kernel<T>), dim3(Ni), dim3(128), lds, stream, fb,
                     cb3, cb4, sumU, pc3, pn3, p3_active, pa4, pb4, pc4, nelements, nrbf3,
                     nabf3_active, nrbf4, nabf4, K3, Q4, Ni, do4 ? 1 : 0);
  return hipGetLastError() == hipSuccess;
}


/* ----------------------------------------------------------------------
   kernels that compute the angular basis on the fly from rij instead of
   reading the abf/abfx/abfy/abfz arrays written by angularbasis(), so that
   angularbasis() can be skipped.  the angular functions of pair j are the
   monomials u^a v^b w^c (a+b+c <= L) of the unit vector (u,v,w) = rij/|rij|.
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
   radialPhi().
------------------------------------------------------------------------- */

template <typename T, typename TA>
__global__ void __launch_bounds__(256) radialbasis_phi_kernel(T *__restrict__ rbf, T *__restrict__ drbf,
                                       const T *__restrict__ rij, const T *__restrict__ rin,
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
  T *phi_s = reinterpret_cast<T *>(lds_raw);    // Phi(s, n) at s + ns*n
  for (int t = threadIdx.x; t < ns * nrbfmax; t += blockDim.x) phi_s[t] = Phi[t];
  __syncthreads();

  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= Nij) return;

  const T E_V = static_cast<T>(2.718281828459045235360287);
  const int itype = ti[n];
  const int jtype = tj[n];
  const int bc_base = (itype * nelements + jtype) * nbesselpars;

  const T xij = rij[3 * n + 0];
  const T yij = rij[3 * n + 1];
  const T zij = rij[3 * n + 2];
  const T dij = sqrt(xij * xij + yij * yij + zij * zij);
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
      if (q < nrbfmax) {
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
    if (q < nrbfmax) {
      rbf[n + Nij * q] = static_cast<T>(sum1[q]);
      drbf[n + Nij * q] = static_cast<T>(sum2[q]);
    }
}

template <typename T, typename TA>
bool radialbasis_phi(hipStream_t stream, T *rbf, T *drbf, const T *rij, const T *rin,
                     const T *invrdiff, int rs0, int rs1, int ds0, int ds1, const T *neg_alpha,
                     const T *pi_inv_t1, const T *dx_factor, const T *Phi, const int *ti,
                     const int *tj, int nelements, int besseldegree, int inversedegree,
                     int nbesselpars, int ns, int nrbfmax, int Nij)
{
  if ((Nij <= 0) || (nrbfmax > MAX_RADIAL)) return false;
  if (ns != nbesselpars * besseldegree + inversedegree) return false;
  const size_t lds = (size_t) ns * nrbfmax * sizeof(T);
  if (lds > 65536) return false;
  return launch(radialbasis_phi_kernel<T, TA>, nblocks_for(Nij, 256), 256, lds, stream, rbf, drbf,
                rij, rin, invrdiff, rs0, rs1, ds0, ds1, neg_alpha, pi_inv_t1, dx_factor, Phi, ti,
                tj, nelements, besseldegree, inversedegree, nbesselpars, ns, nrbfmax, Nij);
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
template <int MAXR, typename T, typename F>
__device__ inline void pair_radial(const RadialSrc<T> &s, const T *rij, const int j,
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
  const int pair = s.ti[j] * s.ne + s.tj[j];
  const T x = rij[3 * j + 0], y = rij[3 * j + 1], z = rij[3 * j + 2];
  const T r = sqrt(x * x + y * y + z * z);
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

template <typename T, typename TA>
__global__ void radialangularsum_otf_kernel(T *__restrict__ sumU, const RadialSrc<T> rs,
                                            const T *__restrict__ rij,
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
  const int start = numij[i];
  const int nj = numij[i + 1] - start;
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
      const T x = rij[3 * j + 0];
      const T y = rij[3 * j + 1];
      const T z = rij[3 * j + 2];
      const T invdij = static_cast<T>(1.0) / sqrt(x * x + y * y + z * z);
      const T u = x * invdij;
      const T v = y * invdij;
      const T w = z * invdij;
      abf_s[jj] = static_cast<T>(1.0);
      for (int k = 1; k < K3; k++) {
        const int d = pqd_s[k];
        const T c = (d == 1) ? u : ((d == 2) ? v : w);
        abf_s[k * ld + jj] = abf_s[pqm_s[k] * ld + jj] * c;
      }
      pair_radial<0>(rs, rij, j, nrbf3, [&](int m, T f, T) { rbf_s[m * ld + jj] = f; });
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
};

// 3- and 4-body descriptors of atom i from its sumU block u_s (in LDS),
// computed by the threads of the calling block
template <typename T, typename TA>
__device__ inline void desc34_atom(const DescArgs<T> &g, const T *u_s, const int i)
{
  const int nelements = g.nelements, Ni = g.Ni;
  const int NeK3 = nelements * g.K3;

  // three-body: d3 index i + Ni*(a + nabf3_active*m) + N3*(element pair)
  const int N3 = Ni * g.nabf3_active * g.nrbf3;
  for (int item = threadIdx.x; item < g.nrbf3 * g.nabf3_active; item += blockDim.x) {
    const int a = item % g.nabf3_active;
    const int m = item / g.nabf3_active;
    const int p = g.p3_active[a];
    const int n1 = g.pn3[p], n2 = g.pn3[p + 1];
    const T *u = u_s + NeK3 * m;
    int k = i + Ni * (a + g.nabf3_active * m);
    for (int i1 = 0; i1 < nelements; i1++)
      for (int i2 = i1; i2 < nelements; i2++, k += N3) {
        TA tmp = TA(0);
        for (int q = n1; q < n2; q++)
          tmp += static_cast<TA>(static_cast<T>(g.pc3[q]) * u[i1 + nelements * q] *
                                 u[i2 + nelements * q]);
        g.d3[k] = static_cast<T>(tmp);
      }
  }

  // four-body: d4 index i + Ni*(p + nabf4*m) + N4*(element triple)
  if (g.do4) {
    const int N4 = Ni * g.nabf4 * g.nrbf4;
    for (int item = threadIdx.x; item < g.nrbf4 * g.nabf4; item += blockDim.x) {
      const int p = item % g.nabf4;
      const int m = item / g.nabf4;
      const int n1 = g.pa4[p], n2 = g.pa4[p + 1];
      const T *u = u_s + NeK3 * m;
      int k = i + Ni * (p + g.nabf4 * m);
      for (int i1 = 0; i1 < nelements; i1++)
        for (int i2 = i1; i2 < nelements; i2++)
          for (int i3 = i2; i3 < nelements; i3++, k += N4) {
            TA tmp = TA(0);
            for (int q = n1; q < n2; q++) {
              const int c = g.pc4[q];
              const int j1 = g.pb4[q], j2 = g.pb4[q + g.Q4], j3 = g.pb4[q + 2 * g.Q4];
              tmp += static_cast<TA>(static_cast<T>(c) * u[i1 + nelements * j1] *
                                     u[i2 + nelements * j2] * u[i3 + nelements * j3]);
            }
            g.d4[k] = static_cast<T>(tmp);
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

// RTM x CTM output tiles (RTM >= ceil(K3/16), CTM >= ceil(Ne*nrbf3/16)) are kept in registers
template <typename T, typename TA, int RTM, int CTM>
__global__ void __launch_bounds__(64)
    radialangularsum_mfma_kernel(T *__restrict__ sumU, const RadialSrc<T> rs,
                                 const T *__restrict__ rij, const int *__restrict__ tj,
                                 const int *__restrict__ numij, const int *__restrict__ pq_m,
                                 const int *__restrict__ pq_d, const int nelements,
                                 const int nrbf3, const int K3, const int L, const int Nij,
                                 const int jtile)
{
  extern __shared__ char lds_raw[];
  const int L1 = L + 1;
  T *pow_s = reinterpret_cast<T *>(lds_raw);      // [jtile][3][L1]
  const int ldr = jtile + 1;                      // padded row: functions m of a pair in different banks
  T *rbf_s = pow_s + jtile * 3 * L1;              // [nrbf3][ldr]
  int *tj_s = reinterpret_cast<int *>(rbf_s + nrbf3 * ldr);    // [jtile]
  int *ex_s = tj_s + jtile;                                      // [K3] packed a | b<<8 | c<<16

  const int i = blockIdx.x;
  const int start = numij[i];
  const int nj = numij[i + 1] - start;
  const int nout = nrbf3 * K3;
  const int NC = nelements * nrbf3;
  const int lane = threadIdx.x;
  const int lr = lane & 15, lq = lane >> 4;
  const int RT = (K3 + 15) / 16, CT = (NC + 15) / 16;

  if (lane == 0) {
    ex_s[0] = 0;
    for (int k = 1; k < K3; k++) ex_s[k] = ex_s[pq_m[k]] + (1 << (8 * (pq_d[k] - 1)));
  }
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

  for (int j0 = 0; j0 < nj; j0 += jtile) {
    const int nt = min(jtile, nj - j0);
    const int nt4 = (nt + 3) & ~3;
    __syncthreads();
    for (int jj = lane; jj < nt4; jj += 64) {
      T *pw = pow_s + jj * 3 * L1;
      if (jj < nt) {
        const int j = start + j0 + jj;
        const T x = rij[3 * j + 0];
        const T y = rij[3 * j + 1];
        const T z = rij[3 * j + 2];
        const T invdij = static_cast<T>(1.0) / sqrt(x * x + y * y + z * z);
        const T u = x * invdij, v = y * invdij, w = z * invdij;
        T pu = static_cast<T>(1.0), pv = pu, pz = pu;
        for (int n = 0; n < L1; n++) {
          pw[n] = pu;
          pw[L1 + n] = pv;
          pw[2 * L1 + n] = pz;
          pu *= u;
          pv *= v;
          pz *= w;
        }
        pair_radial<0>(rs, rij, j, nrbf3, [&](int m, T f, T) { rbf_s[m * ldr + jj] = f; });
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
bool radialangularsum_otf(hipStream_t stream, T *sumU, const RadialSrc<T> &rs, const T *rij, const int *tj,
                          const int *numij, const int *pq_m, const int *pq_d, int nelements,
                          int nrbf3, int K3, int Ni, int Nij, bool use_mfma, int jtile)
{
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
#define LMP_POD_RAS_MFMA(R, C)                                                                   \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(radialangularsum_mfma_kernel<T, TA, R, C>), dim3(Ni),      \
                     dim3(threads), lds2, stream, sumU, rs, rij, tj, numij, pq_m, pq_d,        \
                     nelements, nrbf3, K3, L, Nij, jtile)
    if (CT == 1) {
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
  } else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(radialangularsum_otf_kernel<T, TA>), dim3(Ni),
                       dim3(threads), lds, stream, sumU, rs, rij, tj, numij, pq_m, pq_d,
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
__device__ inline void pod_pair_powers(T *pd, const T *rij, const int j, const int L1)
{
  const T x = rij[3 * j + 0];
  const T y = rij[3 * j + 1];
  const T z = rij[3 * j + 2];
  const T invdij = static_cast<T>(1.0) / sqrt(x * x + y * y + z * z);
  const T u = x * invdij, v = y * invdij, w = z * invdij;
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

// du/dx, du/dy, du/dz, dv/dy, dv/dz, dw/dz, x, y, z of pair j
template <typename T> __device__ inline void pod_pair_gradients(T *g, const T *rij, const int j)
{
  const T x = rij[3 * j + 0];
  const T y = rij[3 * j + 1];
  const T z = rij[3 * j + 2];
  const T xx = x * x, yy = y * y, zz = z * z;
  const T invdij = static_cast<T>(1.0) / sqrt(xx + yy + zz);
  const T invdij3 = invdij * invdij * invdij;
  g[0] = (yy + zz) * invdij3;
  g[1] = -(x * y) * invdij3;
  g[2] = -(x * z) * invdij3;
  g[3] = (xx + zz) * invdij3;
  g[4] = -(y * z) * invdij3;
  g[5] = (xx + yy) * invdij3;
  g[6] = x;
  g[7] = y;
  g[8] = z;
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

template <typename T, typename TA>
__device__ inline void pod_pair_force(T *fij, const T *g, const int j, const TA SA, const TA SU,
                                      const TA SV, const TA SW)
{
  const TA fx = SU * static_cast<TA>(g[0]) + SV * static_cast<TA>(g[1]) +
      SW * static_cast<TA>(g[2]) + SA * static_cast<TA>(g[6]);
  const TA fy = SU * static_cast<TA>(g[1]) + SV * static_cast<TA>(g[3]) +
      SW * static_cast<TA>(g[4]) + SA * static_cast<TA>(g[7]);
  const TA fz = SU * static_cast<TA>(g[2]) + SV * static_cast<TA>(g[4]) +
      SW * static_cast<TA>(g[5]) + SA * static_cast<TA>(g[8]);
  fij[3 * j + 0] += static_cast<T>(fx);
  fij[3 * j + 1] += static_cast<T>(fy);
  fij[3 * j + 2] += static_cast<T>(fz);
}

template <typename T>
__device__ inline void pod_exponents(int *ex_s, const int *pq_m, const int *pq_d, const int K3)
{
  if (threadIdx.x == 0) {
    ex_s[0] = 0;
    for (int k = 1; k < K3; k++) ex_s[k] = ex_s[pq_m[k]] + (1 << (8 * (pq_d[k] - 1)));
  }
}

// scalar version: one block per atom i, one thread per neighbor j

template <typename T, typename TA>
__global__ void __launch_bounds__(64)
    allbody_forces_otf_kernel(T *__restrict__ fij, const T *__restrict__ forcecoeff,
                              const RadialSrc<T> rs, const T *__restrict__ rij, const int *__restrict__ tj,
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

  const int i = blockIdx.x;
  const int start = numij[i];
  const int nj = numij[i + 1] - start;

  for (int t = threadIdx.x; t < nfc; t += blockDim.x) fc_s[t] = forcecoeff[t + nfc * i];
  pod_exponents<T>(ex_s, pq_m, pq_d, K3);
  __syncthreads();

  T *pd = pd_s + threadIdx.x * PD;
  for (int jj = threadIdx.x; jj < nj; jj += blockDim.x) {
    const int j = start + jj;
    const int e = (nelements > 1) ? tj[j] : 0;
    pod_pair_powers<T>(pd, rij, j, L1);

    T r[MAX_RADIAL], dr[MAX_RADIAL];
    pair_radial<MAX_RADIAL>(rs, rij, j, nrbf3, [&](int m, T f, T df) {
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
    pod_pair_gradients<T>(g, rij, j);
    pod_pair_force<T, TA>(fij, g, j, SA, SU, SV, SW);
  }
}

/* ----------------------------------------------------------------------
   matrix core version: one wavefront per atom i.  for a tile of 16 pairs,

     G (16 x K3) = R (16 x Ne*nrbf3) * F (Ne*nrbf3 x K3)

   with R(j, m + nrbf3*e) = [tj(j) == e] rbf(j, m) and F the forcecoeff
   block of atom i (H likewise with drbf), in 16x16 output tiles.  each lane
   then holds G and H for 4 pairs and one angular index k, multiplies them
   with P_k and its derivatives, and SA, SU, SV, SW of a pair are reduced
   across the 16 lanes that hold its row.
------------------------------------------------------------------------- */

static constexpr int MFMA_MAX_CS = 8;    // nelements * nrbf3 <= 32
static constexpr int MFMA_PAIR_TILE = 16;
// register budget of the matrix core force kernel.  without it the compiler keeps the
// accumulators in separate AGPRs (123 VGPRs + 16 AGPRs, 3 waves/SIMD); with the budget
// they share the VGPRs (123 in total, no spills, 4 waves/SIMD)
#ifndef LMP_POD_ALLBODY_VGPR
#define LMP_POD_ALLBODY_VGPR 128
#endif
#if LMP_POD_ALLBODY_VGPR > 0
#define LMP_POD_ALLBODY_ATTR __attribute__((amdgpu_num_vgpr(LMP_POD_ALLBODY_VGPR)))
#else
#define LMP_POD_ALLBODY_ATTR
#endif

template <typename T, typename TA, int CSM, int JT>
__global__ void __launch_bounds__(64) LMP_POD_ALLBODY_ATTR
    allbody_forces_mfma_kernel(T *__restrict__ fij, const T *__restrict__ forcecoeff,
                               const RadialSrc<T> rs, const T *__restrict__ rij, const int *__restrict__ tj,
                               const int *__restrict__ numij, const int *__restrict__ pq_m,
                               const int *__restrict__ pq_d, const int nelements,
                               const int nrbf3, const int K3, const int L, const int Nij)
{
  const int L1 = L + 1;
  const int PD = 3 * L1;    // only the powers; the gradients are recomputed per pair
  const int NeK3 = nelements * K3;
  const int nfc = NeK3 * nrbf3;
  const int NC = nelements * nrbf3;
  const int CS = (NC + 3) / 4;
  const int KT = (K3 + 15) / 16;
  extern __shared__ char lds_raw[];
  T *fc_s = reinterpret_cast<T *>(lds_raw);    // [m][k][e]
  T *pd_s = fc_s + nfc;                        // [JT][PD]
  T *r_s = pd_s + JT * PD;                     // [nrbf3][JT]
  T *dr_s = r_s + nrbf3 * JT;                  // [nrbf3][JT]
  int *tj_s = reinterpret_cast<int *>(dr_s + nrbf3 * JT);    // [JT]
  int *ex_s = tj_s + JT;                                     // [K3]

  const int i = blockIdx.x;
  const int start = numij[i];
  const int nj = numij[i + 1] - start;
  const int lane = threadIdx.x;
  const int lr = lane & 15, lq = lane >> 4;

  for (int t = lane; t < nfc; t += 64) fc_s[t] = forcecoeff[t + nfc * i];
  pod_exponents<T>(ex_s, pq_m, pq_d, K3);

  // (m, e) of the inner index c = 4*cs + lq owned by this lane
  int cm[CSM], ce[CSM];
#pragma unroll
  for (int cs = 0; cs < CSM; cs++) {
    const int c = 4 * cs + lq;
    cm[cs] = (c < NC) ? c % nrbf3 : 0;
    ce[cs] = (c < NC) ? c / nrbf3 : -1;
  }
  typedef typename Mfma<TA>::acc_t acc_t;

  for (int j0 = 0; j0 < nj; j0 += JT) {
    const int nt = min(JT, nj - j0);
    __syncthreads();
    for (int jj = lane; jj < JT; jj += 64) {
      T *pd = pd_s + jj * PD;
      if (jj < nt) {
        const int j = start + j0 + jj;
        pod_pair_powers<T>(pd, rij, j, L1);
        pair_radial<0>(rs, rij, j, nrbf3, [&](int m, T f, T df) {
          r_s[m * JT + jj] = f;
          dr_s[m * JT + jj] = df;
        });
        tj_s[jj] = (nelements > 1) ? tj[j] : 0;
      } else {
        for (int n = 0; n < PD; n++) pd[n] = T(0);
        for (int m = 0; m < nrbf3; m++) r_s[m * JT + jj] = dr_s[m * JT + jj] = T(0);
        tj_s[jj] = -1;
      }
    }
    __syncthreads();

    for (int s0 = 0; s0 < nt; s0 += 16) {
      TA SA[4], SU[4], SV[4], SW[4];
#pragma unroll
      for (int v = 0; v < 4; v++) SA[v] = SU[v] = SV[v] = SW[v] = TA(0);

      // A operands of this pair tile: row s0 + lr, inner index 4*cs + lq
      const int ja = s0 + lr;
      const int tja = tj_s[ja];
      TA ar[CSM], adr[CSM];
#pragma unroll
      for (int cs = 0; cs < CSM; cs++) {
        const bool on = (cs < CS) && (ce[cs] == tja);
        ar[cs] = on ? static_cast<TA>(r_s[cm[cs] * JT + ja]) : TA(0);
        adr[cs] = on ? static_cast<TA>(dr_s[cm[cs] * JT + ja]) : TA(0);
      }

      for (int kt = 0; kt < KT; kt++) {
        const int k = 16 * kt + lr;
        acc_t accG = acc_t{0, 0, 0, 0}, accH = acc_t{0, 0, 0, 0};
#pragma unroll
        for (int cs = 0; cs < CSM; cs++)
          if (cs < CS) {
            const TA b = ((k < K3) && (ce[cs] >= 0))
                ? static_cast<TA>(fc_s[ce[cs] + nelements * k + NeK3 * cm[cs]])
                : TA(0);
            accG = Mfma<TA>::mma(ar[cs], b, accG);
            accH = Mfma<TA>::mma(adr[cs], b, accH);
          }
        const int ex = (k < K3) ? ex_s[k] : 0;
#pragma unroll
        for (int v = 0; v < 4; v++) {
          const T *pd = pd_s + (s0 + Mfma<TA>::row(lane, v)) * PD;
          T P, Pu, Pv, Pw;
          pod_monomial<T>(pd, ex, L1, P, Pu, Pv, Pw);
          SA[v] += accH[v] * static_cast<TA>(P);
          SU[v] += accG[v] * static_cast<TA>(Pu);
          SV[v] += accG[v] * static_cast<TA>(Pv);
          SW[v] += accG[v] * static_cast<TA>(Pw);
        }
      }

      // reduce over the 16 lanes (columns k) that hold the same rows
#pragma unroll
      for (int v = 0; v < 4; v++)
#pragma unroll
        for (int off = 1; off < 16; off <<= 1) {
          SA[v] += __shfl_xor(SA[v], off);
          SU[v] += __shfl_xor(SU[v], off);
          SV[v] += __shfl_xor(SV[v], off);
          SW[v] += __shfl_xor(SW[v], off);
        }
      // lanes lr = 0..3 of a row group add the forces of rows v = lr
      if (lr < 4) {
        const int jj = s0 + Mfma<TA>::row(lane, lr);
        if (jj < nt) {
          const TA sa = (lr == 0) ? SA[0] : ((lr == 1) ? SA[1] : ((lr == 2) ? SA[2] : SA[3]));
          const TA su = (lr == 0) ? SU[0] : ((lr == 1) ? SU[1] : ((lr == 2) ? SU[2] : SU[3]));
          const TA sv = (lr == 0) ? SV[0] : ((lr == 1) ? SV[1] : ((lr == 2) ? SV[2] : SV[3]));
          const TA sw = (lr == 0) ? SW[0] : ((lr == 1) ? SW[1] : ((lr == 2) ? SW[2] : SW[3]));
          T g[9];
          pod_pair_gradients<T>(g, rij, start + j0 + jj);
          pod_pair_force<T, TA>(fij, g, start + j0 + jj, sa, su, sv, sw);
        }
      }
    }
  }
}

template <typename T, typename TA>
bool allbody_forces_otf(hipStream_t stream, T *fij, const T *forcecoeff, const RadialSrc<T> &rs,
                        const T *rij, const int *tj, const int *numij,
                        const int *pq_m, const int *pq_d, int nelements, int nrbf3, int K3,
                        int Ni, int Nij, bool use_mfma)
{
  if ((Ni <= 0) || (nrbf3 > MAX_RADIAL) || (nelements > MAX_ELEMENTS)) return false;
  int L = 0;
  while ((L + 1) * (L + 2) * (L + 3) / 6 < K3) L++;
  if ((L + 1) * (L + 2) * (L + 3) / 6 != K3) return false;
  const int PD = 3 * (L + 1);
  const size_t nfc = (size_t) nelements * K3 * nrbf3;
  const int CS = (nelements * nrbf3 + 3) / 4;

  if (use_mfma && device_has_mfma() && (CS <= MFMA_MAX_CS)) {
    constexpr int JT = MFMA_PAIR_TILE;
    const size_t lds = (nfc + (size_t) JT * (PD + 2 * nrbf3)) * sizeof(T) +
        ((size_t) JT + K3) * sizeof(int);
    if (lds <= 65536) {
#define LMP_POD_ALLBODY_MFMA(CSM)                                                                 \
  hipLaunchKernelGGL(HIP_KERNEL_NAME(allbody_forces_mfma_kernel<T, TA, CSM, JT>), dim3(Ni),       \
                     dim3(64), lds, stream, fij, forcecoeff, rs, rij, tj, numij, pq_m,            \
                     pq_d, nelements, nrbf3, K3, L, Nij)
      if (CS <= 2)
        LMP_POD_ALLBODY_MFMA(2);
      else if (CS <= 4)
        LMP_POD_ALLBODY_MFMA(4);
      else
        LMP_POD_ALLBODY_MFMA(8);
#undef LMP_POD_ALLBODY_MFMA
      return hipGetLastError() == hipSuccess;
    }
  }

  const size_t lds = (nfc + (size_t) 64 * PD) * sizeof(T) + (size_t) K3 * sizeof(int);
  if (lds > 65536) return false;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(allbody_forces_otf_kernel<T, TA>), dim3(Ni), dim3(64), lds,
                     stream, fij, forcecoeff, rs, rij, tj, numij, pq_m, pq_d, nelements,
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
                       const int Ni, const int Nij)
{
  extern __shared__ char lds_raw[];
  TA *s_s = reinterpret_cast<TA *>(lds_raw);    // [e][m]
  const int i = blockIdx.x;
  const int start = numij[i];
  const int end = numij[i + 1];
  const int nd = nelements * nrbf2;
  for (int t = threadIdx.x; t < nd; t += blockDim.x) s_s[t] = TA(0);
  __syncthreads();
  for (int n = start + threadIdx.x; n < end; n += blockDim.x) {
    TA *se = s_s + nrbf2 * ((nelements > 1) ? tj[n] : 0);
    for (int m = 0; m < nrbf2; m++) atomicAdd(se + m, static_cast<TA>(rbf2[n + Nij * m]));
  }
  __syncthreads();
  for (int t = threadIdx.x; t < nd; t += blockDim.x) d2[i + Ni * t] = static_cast<T>(s_s[t]);
}

template <typename T, typename TA>
bool twobodydesc(hipStream_t stream, T *d2, const T *rbf2, const int *tj, const int *numij,
                 int nelements, int nrbf2, int Ni, int Nij)
{
  if (Ni <= 0) return false;
  const size_t lds = (size_t) nelements * nrbf2 * sizeof(TA);
  if (lds > 65536) return false;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(twobodydesc_kernel<T, TA>), dim3(Ni), dim3(64), lds, stream,
                     d2, rbf2, tj, numij, nelements, nrbf2, Ni, Nij);
  return hipGetLastError() == hipSuccess;
}

/* ======================================================================
   native versions of the remaining pair pod/kk kernels.  unless noted,
   they use the same decomposition and arithmetic as the Kokkos kernels
   they replace; kernels that share an index space are fused.
====================================================================== */

// wavefront and block sums; blockDim.x must be a multiple of 64 and at most 1024
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
    neighbor_list_kernel(T *__restrict__ rij, int *__restrict__ typeai, int *__restrict__ idxi,
                         int *__restrict__ ai, int *__restrict__ aj, int *__restrict__ ti,
                         int *__restrict__ tj, const int *__restrict__ numij,
                         const T *__restrict__ x, const T *__restrict__ rcutsq,
                         const NeighArgs a, const int gi1, const int Ni)
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
      rij[3 * n + 0] = dx;
      rij[3 * n + 1] = dy;
      rij[3 * n + 2] = dz;
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
bool neighbor_list(hipStream_t stream, T *rij, int *typeai, int *idxi, int *ai, int *aj, int *ti,
                   int *tj, const int *numij, const T *x, const T *rcutsq, const NeighArgs &a,
                   int gi1, int Ni)
{
  return launch(neighbor_list_kernel<T>, nblocks_for(Ni, 4), 256, 0, stream, rij, typeai, idxi,
                ai, aj, ti, tj, numij, x, rcutsq, a, gi1, Ni);
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
                               const T *__restrict__ rij, const int *__restrict__ ti,
                               const int *__restrict__ tj, const T *__restrict__ r0v,
                               const T *__restrict__ invdrv, const T *__restrict__ coef,
                               const int ne, const int nb, const int nr, const int N,
                               const int *__restrict__ numij, T *__restrict__ d2,
                               const int nrbf2, const int Ni)
{
  __shared__ TA d2_s[PERATOM ? 256 : 1];    // PERATOM: 2-body descriptors of atom i, [e][m]
  if (PERATOM) {
    for (int t = threadIdx.x; t < ne * nrbf2; t += blockDim.x) d2_s[t] = TA(0);
    __syncthreads();
  }
  auto pair = [&](const int n) {
  const int pair = ti[n] * ne + tj[n];
  const T x = rij[3 * n + 0], y = rij[3 * n + 1], z = rij[3 * n + 2];
  const T r = sqrt(x * x + y * y + z * z);
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
    if (PERATOM && (k < nrbf2)) atomicAdd(d2_s + tj[n] * nrbf2 + k, static_cast<TA>(f));
  }
  };

  if (PERATOM) {
    const int i = blockIdx.x;
    for (int n = numij[i] + threadIdx.x; n < numij[i + 1]; n += blockDim.x) pair(n);
    __syncthreads();
    for (int t = threadIdx.x; t < ne * nrbf2; t += blockDim.x) d2[i + Ni * t] = static_cast<T>(d2_s[t]);
  } else {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) pair(n);
  }
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
__device__ inline int pod_bspline_pair(const BsplineArgs<T> &g, const T *__restrict__ rij,
                                       const int n, T (&Bq)[4], T (&dBq)[4])
{
  const int pair = g.ti[n] * g.ne + g.tj[n];
  const T x = rij[3 * n + 0], y = rij[3 * n + 1], z = rij[3 * n + 2];
  const T r = sqrt(x * x + y * y + z * z);
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
__global__ void __launch_bounds__(256)
    radialbasis_bspline_kernel(T *__restrict__ rbf2, T *__restrict__ drbf2,
                               int *__restrict__ base, const T *__restrict__ rij,
                               const BsplineArgs<T> g, const int N)
{
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  T Bq[4], dBq[4];
  base[n] = pod_bspline_pair(g, rij, n, Bq, dBq);
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
                          const T *__restrict__ drbf2, const T *__restrict__ rij,
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
    const int ii = idxi[n] + Ni * (pod_bspline_pair(bs, rij, n, Bq, dBq) + nrbf2 * tj[n]);
#pragma unroll
    for (int m = 0; m < 4; ++m) cR += static_cast<TA>(cb2[ii + Ni * m] * dBq[m]);
  } else {
    // base == nullptr: all nm = nrbf2 radial functions; otherwise nm = 4 B-splines from base[n]
    const int ii = idxi[n] + Ni * ((base ? base[n] : 0) + nrbf2 * tj[n]);
    for (int m = 0; m < nm; ++m) cR += static_cast<TA>(cb2[ii + Ni * m] * drbf2[n + Nij * m]);
  }
  fij[3 * n + 0] += static_cast<T>(cR * static_cast<TA>(rij[3 * n + 0]));
  fij[3 * n + 1] += static_cast<T>(cR * static_cast<TA>(rij[3 * n + 1]));
  fij[3 * n + 2] += static_cast<T>(cR * static_cast<TA>(rij[3 * n + 2]));
}

template <typename T, typename TA>
__global__ void __launch_bounds__(64)
    twobodydesc_bspline_kernel(T *__restrict__ d2, const T *__restrict__ rbf2,
                               const int *__restrict__ base, const int *__restrict__ tj,
                               const int *__restrict__ numij, const int nelements,
                               const int nrbf2, const int Ni, const int Nij,
                               const BsplineArgs<T> bs, const T *__restrict__ rij)
{
  extern __shared__ char lds_raw[];
  TA *s_s = reinterpret_cast<TA *>(lds_raw);    // [e][m]
  const int i = blockIdx.x;
  const int start = numij[i], end = numij[i + 1];
  const int nd = nelements * nrbf2;
  for (int t = threadIdx.x; t < nd; t += blockDim.x) s_s[t] = TA(0);
  __syncthreads();
  for (int n = start + threadIdx.x; n < end; n += blockDim.x) {
    if (bs.r0v) {
      // B-splines evaluated here instead of read from rbf2/base
      T Bq[4], dBq[4];
      TA *se = s_s + pod_bspline_pair(bs, rij, n, Bq, dBq) + nrbf2 * tj[n];
#pragma unroll
      for (int q = 0; q < 4; q++) atomicAdd(se + q, static_cast<TA>(Bq[q]));
    } else {
      TA *se = s_s + base[n] + nrbf2 * tj[n];
      for (int q = 0; q < 4; q++) atomicAdd(se + q, static_cast<TA>(rbf2[n + Nij * q]));
    }
  }
  __syncthreads();
  for (int t = threadIdx.x; t < nd; t += blockDim.x) d2[i + Ni * t] = static_cast<T>(s_s[t]);
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

template <typename T, typename TA>
__global__ void __launch_bounds__(256)
    tallystress_kernel(TA *__restrict__ acc, TA *__restrict__ vatom, const int vs0,
                       const int vs1, const T *__restrict__ fij, const T *__restrict__ rij,
                       const int *__restrict__ ai, const int *__restrict__ aj, const int Nij,
                       const int global)
{
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  T rx = 0, ry = 0, rz = 0, fx = 0, fy = 0, fz = 0;
  if (k < Nij) {
    rx = rij[3 * k + 0];
    ry = rij[3 * k + 1];
    rz = rij[3 * k + 2];
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
