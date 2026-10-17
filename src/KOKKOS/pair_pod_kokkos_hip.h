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
  const int threads = 256;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(radialbasis_phi_kernel<T, TA>),
                     dim3((Nij + threads - 1) / threads), dim3(threads), lds, stream, rbf, drbf,
                     rij, rin, invrdiff, rs0, rs1, ds0, ds1, neg_alpha, pi_inv_t1, dx_factor, Phi,
                     ti, tj, nelements, besseldegree, inversedegree, nbesselpars, ns, nrbfmax,
                     Nij);
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
static constexpr int MFMA_MAX_CT = 2;    // nelements * nrbf3 <= 32

template <typename T, typename TA>
__global__ void radialangularsum_otf_kernel(T *__restrict__ sumU, const T *__restrict__ rbf,
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
      for (int m = 0; m < nrbf3; m++) rbf_s[m * ld + jj] = rbf[j + Nij * m];
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
   matrix core version of radialangularsum_otf_kernel: one wavefront per
   atom computes the sums as a GEMM on the matrix cores,

     U (K3 x Ne*nrbf3) = A (K3 x nj) * B (nj x Ne*nrbf3),

   with A(k, j) = abf(j, k) and B(j, m + nrbf3*e) = [tj(j) == e] rbf(j, m),
   in 16x16 output tiles padded with zeros.  LDS holds only the powers u^n,
   v^n, w^n (n <= L) and the radial functions of a tile of neighbors; each
   lane builds its own element of A, abf(j, k) = u^a v^b w^c, from the
   powers.
------------------------------------------------------------------------- */

template <typename T, typename TA>
__global__ void __launch_bounds__(64)
    radialangularsum_mfma_kernel(T *__restrict__ sumU, const T *__restrict__ rbf,
                                 const T *__restrict__ rij, const int *__restrict__ tj,
                                 const int *__restrict__ numij, const int *__restrict__ pq_m,
                                 const int *__restrict__ pq_d, const int nelements,
                                 const int nrbf3, const int K3, const int L, const int Nij,
                                 const int jtile)
{
  extern __shared__ char lds_raw[];
  const int L1 = L + 1;
  T *pow_s = reinterpret_cast<T *>(lds_raw);      // [jtile][3][L1]
  T *rbf_s = pow_s + jtile * 3 * L1;              // [nrbf3][jtile]
  int *tj_s = reinterpret_cast<int *>(rbf_s + nrbf3 * jtile);    // [jtile]
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
  int oa[MFMA_MAX_RT], ob[MFMA_MAX_RT], oc[MFMA_MAX_RT];
#pragma unroll
  for (int rt = 0; rt < MFMA_MAX_RT; rt++) {
    const int k = rt * 16 + lr;
    const int ex = (k < K3) ? ex_s[k] : 0;
    oa[rt] = ex & 255;
    ob[rt] = L1 + ((ex >> 8) & 255);
    oc[rt] = 2 * L1 + ((ex >> 16) & 255);
  }
  int bm[MFMA_MAX_CT], be[MFMA_MAX_CT];
#pragma unroll
  for (int ct = 0; ct < MFMA_MAX_CT; ct++) {
    const int c = ct * 16 + lr;
    bm[ct] = (c < NC) ? c % nrbf3 : -1;
    be[ct] = (c < NC) ? c / nrbf3 : -1;
  }

  typedef typename Mfma<TA>::acc_t acc_t;
  acc_t acc[MFMA_MAX_RT][MFMA_MAX_CT];
#pragma unroll
  for (int rt = 0; rt < MFMA_MAX_RT; rt++)
#pragma unroll
    for (int ct = 0; ct < MFMA_MAX_CT; ct++) acc[rt][ct] = acc_t{0, 0, 0, 0};

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
        for (int m = 0; m < nrbf3; m++) rbf_s[m * jtile + jj] = rbf[j + Nij * m];
        tj_s[jj] = (nelements > 1) ? tj[j] : 0;
      } else {
        for (int n = 0; n < 3 * L1; n++) pw[n] = T(0);
        for (int m = 0; m < nrbf3; m++) rbf_s[m * jtile + jj] = T(0);
        tj_s[jj] = -1;
      }
    }
    __syncthreads();

    for (int q = 0; q < nt4; q += 4) {
      const int jj = q + lq;
      const T *pw = pow_s + jj * 3 * L1;
      const int tjj = tj_s[jj];
      TA a[MFMA_MAX_RT], b[MFMA_MAX_CT];
#pragma unroll
      for (int rt = 0; rt < MFMA_MAX_RT; rt++)
        a[rt] = (rt * 16 + lr < K3) ? static_cast<TA>(pw[oa[rt]] * pw[ob[rt]] * pw[oc[rt]])
                                    : TA(0);
#pragma unroll
      for (int ct = 0; ct < MFMA_MAX_CT; ct++)
        b[ct] = ((bm[ct] >= 0) && (be[ct] == tjj)) ? static_cast<TA>(rbf_s[bm[ct] * jtile + jj])
                                                   : TA(0);
#pragma unroll
      for (int rt = 0; rt < MFMA_MAX_RT; rt++)
        if (rt < RT)
#pragma unroll
          for (int ct = 0; ct < MFMA_MAX_CT; ct++)
            if (ct < CT) acc[rt][ct] = Mfma<TA>::mma(a[rt], b[ct], acc[rt][ct]);
    }
  }

#pragma unroll
  for (int rt = 0; rt < MFMA_MAX_RT; rt++)
#pragma unroll
    for (int ct = 0; ct < MFMA_MAX_CT; ct++)
#pragma unroll
      for (int v = 0; v < 4; v++) {
        const int k = rt * 16 + Mfma<TA>::row(lane, v);
        if ((rt < RT) && (ct < CT) && (k < K3) && (bm[ct] >= 0))
          sumU[be[ct] + nelements * (k + K3 * bm[ct] + nout * i)] = static_cast<T>(acc[rt][ct][v]);
      }
}

template <typename T, typename TA>
bool radialangularsum_otf(hipStream_t stream, T *sumU, const T *rbf, const T *rij, const int *tj,
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
    const size_t lds2 = (size_t) (3 * (L + 1) + nrbf3) * jtile * sizeof(T) +
        ((size_t) jtile + K3) * sizeof(int);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(radialangularsum_mfma_kernel<T, TA>), dim3(Ni),
                       dim3(threads), lds2, stream, sumU, rbf, rij, tj, numij, pq_m, pq_d,
                       nelements, nrbf3, K3, L, Nij, jtile);
  } else
    hipLaunchKernelGGL(HIP_KERNEL_NAME(radialangularsum_otf_kernel<T, TA>), dim3(Ni),
                       dim3(threads), lds, stream, sumU, rbf, rij, tj, numij, pq_m, pq_d,
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
                              const T *__restrict__ rbf, const T *__restrict__ drbf,
                              const T *__restrict__ rij, const int *__restrict__ tj,
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
#pragma unroll
    for (int m = 0; m < MAX_RADIAL; m++)
      if (m < nrbf3) {
        r[m] = rbf[j + Nij * m];
        dr[m] = drbf[j + Nij * m];
      }

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

template <typename T, typename TA, int CSM, int JT>
__global__ void __launch_bounds__(64)
    allbody_forces_mfma_kernel(T *__restrict__ fij, const T *__restrict__ forcecoeff,
                               const T *__restrict__ rbf, const T *__restrict__ drbf,
                               const T *__restrict__ rij, const int *__restrict__ tj,
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
        for (int m = 0; m < nrbf3; m++) {
          r_s[m * JT + jj] = rbf[j + Nij * m];
          dr_s[m * JT + jj] = drbf[j + Nij * m];
        }
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
      if (lr == 0) {
#pragma unroll
        for (int v = 0; v < 4; v++) {
          const int jj = s0 + Mfma<TA>::row(lane, v);
          if (jj < nt) {
            T g[9];
            pod_pair_gradients<T>(g, rij, start + j0 + jj);
            pod_pair_force<T, TA>(fij, g, start + j0 + jj, SA[v], SU[v], SV[v], SW[v]);
          }
        }
      }
    }
  }
}

template <typename T, typename TA>
bool allbody_forces_otf(hipStream_t stream, T *fij, const T *forcecoeff, const T *rbf,
                        const T *drbf, const T *rij, const int *tj, const int *numij,
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
                     dim3(64), lds, stream, fij, forcecoeff, rbf, drbf, rij, tj, numij, pq_m,     \
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
                     stream, fij, forcecoeff, rbf, drbf, rij, tj, numij, pq_m, pq_d, nelements,
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
__global__ void desc34_kernel(T *__restrict__ d3, T *__restrict__ d4, const T *__restrict__ sumU,
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

  const int i = blockIdx.x;
  const int NeK3 = nelements * K3;
  for (int t = threadIdx.x; t < S; t += blockDim.x) u_s[t] = sumU[S * i + t];
  __syncthreads();

  // three-body: d3 index i + Ni*(a + nabf3_active*m) + N3*(element pair)
  const int N3 = Ni * nabf3_active * nrbf3;
  for (int item = threadIdx.x; item < nrbf3 * nabf3_active; item += blockDim.x) {
    const int a = item % nabf3_active;
    const int m = item / nabf3_active;
    const int p = p3_active[a];
    const int n1 = pn3[p], n2 = pn3[p + 1];
    const T *u = u_s + NeK3 * m;
    int k = i + Ni * (a + nabf3_active * m);
    for (int i1 = 0; i1 < nelements; i1++)
      for (int i2 = i1; i2 < nelements; i2++, k += N3) {
        TA tmp = TA(0);
        for (int q = n1; q < n2; q++)
          tmp += static_cast<TA>(static_cast<T>(pc3[q]) * u[i1 + nelements * q] *
                                 u[i2 + nelements * q]);
        d3[k] = static_cast<T>(tmp);
      }
  }

  // four-body: d4 index i + Ni*(p + nabf4*m) + N4*(element triple)
  if (do4) {
    const int N4 = Ni * nabf4 * nrbf4;
    for (int item = threadIdx.x; item < nrbf4 * nabf4; item += blockDim.x) {
      const int p = item % nabf4;
      const int m = item / nabf4;
      const int n1 = pa4[p], n2 = pa4[p + 1];
      const T *u = u_s + NeK3 * m;
      int k = i + Ni * (p + nabf4 * m);
      for (int i1 = 0; i1 < nelements; i1++)
        for (int i2 = i1; i2 < nelements; i2++)
          for (int i3 = i2; i3 < nelements; i3++, k += N4) {
            TA tmp = TA(0);
            for (int q = n1; q < n2; q++) {
              const int c = pc4[q];
              const int j1 = pb4[q], j2 = pb4[q + Q4], j3 = pb4[q + 2 * Q4];
              tmp += static_cast<TA>(static_cast<T>(c) * u[i1 + nelements * j1] *
                                     u[i2 + nelements * j2] * u[i3 + nelements * j3]);
            }
            d4[k] = static_cast<T>(tmp);
          }
    }
  }
}

template <typename T, typename TA>
bool desc34(hipStream_t stream, T *d3, T *d4, const T *sumU, const int *pc3, const int *pn3,
            const int *p3_active, const int *pa4, const int *pb4, const int *pc4, int nelements,
            int nrbf3, int nabf3_active, int nrbf4, int nabf4, int K3, int Q4, int Ni, bool do4)
{
  if (Ni <= 0) return false;
  const size_t lds = (size_t) nelements * K3 * nrbf3 * sizeof(T);
  if (lds > 65536) return false;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(desc34_kernel<T, TA>), dim3(Ni), dim3(64), lds, stream, d3,
                     d4, sumU, pc3, pn3, p3_active, pa4, pb4, pc4, nelements, nrbf3, nabf3_active,
                     nrbf4, nabf4, K3, Q4, Ni, do4 ? 1 : 0);
  return hipGetLastError() == hipSuccess;
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

}    // namespace PODHIP
}    // namespace LAMMPS_NS

#endif    // KOKKOS_ENABLE_HIP
#endif
