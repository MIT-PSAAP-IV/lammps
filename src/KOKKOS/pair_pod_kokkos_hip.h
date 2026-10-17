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

}    // namespace PODHIP
}    // namespace LAMMPS_NS

#endif    // KOKKOS_ENABLE_HIP
#endif
