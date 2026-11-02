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

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(pod/atom/kk,ComputePODAtomKokkos<LMPDeviceType>);
ComputeStyle(pod/atom/kk/device,ComputePODAtomKokkos<LMPDeviceType>);
ComputeStyle(pod/atom/kk/host,ComputePODAtomKokkos<LMPHostType>);
// clang-format on
#else

#ifndef LMP_COMPUTE_POD_ATOM_KOKKOS_H
#define LMP_COMPUTE_POD_ATOM_KOKKOS_H

#include "compute_pod_atom.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<class DeviceType> class PairPODKokkos;

template<class DeviceType>
class ComputePODAtomKokkos : public ComputePODAtom {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef Kokkos::DualView<double**, Kokkos::LayoutRight, DeviceType> tdual_pod_2d;
  typedef Kokkos::View<double*, DeviceType> t_uq_1d;
  typedef Kokkos::View<int*, DeviceType> t_uq_1i;

  ComputePODAtomKokkos(class LAMMPS *, int, char **);
  ~ComputePODAtomKokkos() override;
  void init() override;
  void compute_peratom() override;
  double memory_usage() override;

  // setup (host)
  void setup_engine();
  void setup_gram();
  void grow_uq_atoms(int Ni);
  void grow_uq_pairs(int Nij);

  // kernels (device, RangePolicy only)
  void redirect_pair_forces(int Nij);
  void uq_gate(int gi1, int Ni);
  void uq_leverage(int Ni);
  void fill_cb(int w, int Ni);
  void pair_force_pass(int Ni, int Nij);
  void accumulate_norm2(t_uq_1d acc, int Ni);
  void uq_finalize(int gi1, int Ni);

  PairPODKokkos<DeviceType> *podkk;   // private kernel engine (never registered with force)
  int host_flag, use_kk, have_gram, ncomm;
  int nimax_uq, nijmax_uq;

  tdual_pod_2d k_pod;                          // nmax x 9 output
  typename tdual_pod_2d::t_dev d_pod;
  typename AT::t_int_1d_randomread mask;
  typename AT::t_kkacc_1d_3 d_fij;             // 2*Nij x 3: rows [0,Nij) = per-pair J_ij*c

  t_uq_1d d_Linv, d_uqref;                     // packed L^-1 per (elem,cluster), refs
  t_uq_1d d_U, d_sF2, d_phi2, d_P, d_lev;      // per block atom
  t_uq_1i d_ks, d_kn, d_win;                   // kn == 0 marks a skipped atom
};

}    // namespace LAMMPS_NS

#endif
#endif