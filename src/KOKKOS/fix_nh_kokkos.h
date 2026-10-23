// clang-format off
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

#ifndef LMP_FIX_NH_KOKKOS_H
#define LMP_FIX_NH_KOKKOS_H

#include "fix_nh.h"
#include "kokkos_type.h"

namespace LAMMPS_NS {

template<int TRICLINIC_FLAG>
struct TagFixNH_nh_v_press{};

template<int RMASS>
struct TagFixNH_nve_v{};

struct TagFixNH_nve_x{};

struct TagFixNH_nh_v_temp{};

template<int TRICLINIC_FLAG, int RMASS>
struct TagFixNH_fused_initial{};

template<int TRICLINIC_FLAG, int RMASS>
struct TagFixNH_fused_final{};

template<class DeviceType>
class FixNHKokkos : public FixNH {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;

  FixNHKokkos(class LAMMPS *, int, char **);

  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void pre_exchange() override;

  template<int TRICLINIC_FLAG>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nh_v_press<TRICLINIC_FLAG>, const int&) const;

  template<int RMASS>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nve_v<RMASS>, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nve_x, const int&) const;

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_nh_v_temp, const int&) const;

  template<int TRICLINIC_FLAG, int RMASS>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_fused_initial<TRICLINIC_FLAG,RMASS>, const int&) const;

  template<int TRICLINIC_FLAG, int RMASS>
// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(TagFixNH_fused_final<TRICLINIC_FLAG,RMASS>, const int&) const;

 protected:
  void remap() override;

  void nve_x() override;            // may be overwritten by child classes
  void nve_v() override;
  void nh_v_press() override;
  void nh_v_temp() override;

  void remap_box();
  int fuse_ok();
  void set_press_factor();
  void fused_initial();
  void fused_final();

  int fusable;    // 0 if a derived class changes the per-atom updates
  KK_FLOAT factor[3];

  // box before and after each of the two half-step remaps of initial_integrate()
  KK_FLOAT b0_lo[3], b0_hinv[6], b1_lo[3], b1_h[6], b1_hinv[6], b2_lo[3], b2_h[6];

  class DomainKokkos *domainKK;

  typename AT::t_kkfloat_1d_3_lr x;
  typename AT::t_kkfloat_1d_3 v;
  typename AT::t_kkacc_1d_3_const f;
  typename AT::t_kkfloat_1d rmass;
  typename AT::t_kkfloat_1d mass;
  typename AT::t_int_1d type;
  typename AT::t_int_1d mask;
};

}

#endif

