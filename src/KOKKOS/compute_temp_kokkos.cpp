// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_temp_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "update.h"

using namespace LAMMPS_NS;

// the reductions use the whole compute as functor; see fix_nh_kokkos.cpp
// for why it is passed as kernel argument

using LightWeight = Kokkos::Experimental::WorkItemProperty::HintLightWeight_t;

namespace {

// the scalar temperature as per-team partial sums of CHUNK atoms each,
// added up on the host.  This avoids the inter-block step of a Kokkos
// reduction, which on an MI300A costs about 30 us per call.

static constexpr int CHUNK = 256;

template<class DeviceType, int RMASS, class VType, class MType, class IType, class PType>
struct ComputeTempScalarPartial {
  typedef DeviceType device_type;
  typedef typename Kokkos::TeamPolicy<DeviceType>::member_type member_type;
  VType v;
  MType rmass, mass;
  IType type, mask;
  PType part;
  int groupbit, n;

  ComputeTempScalarPartial(const VType &v_, const MType &rmass_, const MType &mass_,
                           const IType &type_, const IType &mask_, const PType &part_,
                           int groupbit_, int n_) :
    v(v_), rmass(rmass_), mass(mass_), type(type_), mask(mask_), part(part_),
    groupbit(groupbit_), n(n_) {}

// NOLINTNEXTLINE
  KOKKOS_INLINE_FUNCTION
  void operator()(const member_type &team) const {
    const int i0 = team.league_rank() * CHUNK;
    double sum = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, CHUNK), [&](const int k, double &lsum) {
      const int i = i0 + k;
      if ((i < n) && (mask[i] & groupbit)) {
        if (RMASS)
          lsum += static_cast<double>((v(i,0)*v(i,0) + v(i,1)*v(i,1) + v(i,2)*v(i,2)) * rmass[i]);
        else
          lsum += static_cast<double>((v(i,0)*v(i,0) + v(i,1)*v(i,1) + v(i,2)*v(i,2)) *
            mass[type[i]]);
      }
    }, sum);
    Kokkos::single(Kokkos::PerTeam(team), [&]() { part(team.league_rank()) = sum; });
  }
};

}    // namespace

/* ---------------------------------------------------------------------- */

template<class DeviceType>
ComputeTempKokkos<DeviceType>::ComputeTempKokkos(LAMMPS *lmp, int narg, char **arg) :
  ComputeTemp(lmp, narg, arg)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  datamask_read = V_MASK | MASK_MASK | RMASS_MASK | TYPE_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double ComputeTempKokkos<DeviceType>::compute_scalar()
{
  atomKK->sync(execution_space,datamask_read);
  atomKK->k_mass.sync<DeviceType>();

  invoked_scalar = update->ntimestep;

  v = atomKK->k_v.view<DeviceType>();
  if (atomKK->rmass)
    rmass = atomKK->k_rmass.view<DeviceType>();
  else
    mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atom->nlocal;

  double t = 0.0;

  const int nchunk = (nlocal + CHUNK - 1) / CHUNK;
  if (nchunk > 0) {
    if ((int) d_part.extent(0) < nchunk) {
      d_part = Kokkos::View<double*, DeviceType>("temp/kk:part", nchunk);
      h_part = Kokkos::View<double*, LMPPinnedHostType>("temp/kk:h_part", nchunk);
    }
    using VType = decltype(v);
    using MType = decltype(mass);
    using IType = decltype(type);
    using PType = decltype(d_part);
    if (atomKK->rmass) {
      ComputeTempScalarPartial<DeviceType,1,VType,MType,IType,PType>
        f(v,rmass,mass,type,mask,d_part,groupbit,nlocal);
      Kokkos::parallel_for(Kokkos::TeamPolicy<DeviceType>(nchunk,Kokkos::AUTO),f);
    } else {
      ComputeTempScalarPartial<DeviceType,0,VType,MType,IType,PType>
        f(v,rmass,mass,type,mask,d_part,groupbit,nlocal);
      Kokkos::parallel_for(Kokkos::TeamPolicy<DeviceType>(nchunk,Kokkos::AUTO),f);
    }
    auto d_sub = Kokkos::subview(d_part,std::make_pair(0,nchunk));
    auto h_sub = Kokkos::subview(h_part,std::make_pair(0,nchunk));
    Kokkos::deep_copy(h_sub,d_sub);
    for (int k = 0; k < nchunk; k++) t += h_sub(k);
  }

  MPI_Allreduce(&t,&scalar,1,MPI_DOUBLE,MPI_SUM,world);
  if (dynamic) dof_compute();
  if (dof < 0.0 && natoms_temp > 0.0)
    error->all(FLERR,"Temperature compute degrees of freedom < 0");
  scalar *= tfactor;

  return scalar;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void ComputeTempKokkos<DeviceType>::compute_vector()
{
  atomKK->sync(execution_space,datamask_read);
  atomKK->k_mass.sync<DeviceType>();

  int i;

  invoked_vector = update->ntimestep;

  v = atomKK->k_v.view<DeviceType>();
  if (atomKK->rmass)
    rmass = atomKK->k_rmass.view<DeviceType>();
  else
    mass = atomKK->k_mass.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  mask = atomKK->k_mask.view<DeviceType>();
  int nlocal = atom->nlocal;

  double t[6];
  for (i = 0; i < 6; i++) t[i] = 0.0;
  CTEMP t_kk;

  copymode = 1;
  if (atomKK->rmass)
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagComputeTempVector<1>, LightWeight>(0,nlocal),*this,t_kk);
  else
    Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagComputeTempVector<0>, LightWeight>(0,nlocal),*this,t_kk);
  copymode = 0;

  t[0] = t_kk.t0;
  t[1] = t_kk.t1;
  t[2] = t_kk.t2;
  t[3] = t_kk.t3;
  t[4] = t_kk.t4;
  t[5] = t_kk.t5;

  MPI_Allreduce(t,vector,6,MPI_DOUBLE,MPI_SUM,world);
  for (i = 0; i < 6; i++) vector[i] *= force->mvv2e;
}

template<class DeviceType>
template<int RMASS>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void ComputeTempKokkos<DeviceType>::operator()(TagComputeTempVector<RMASS>, const int &i, CTEMP& t_kk) const {
  if (mask[i] & groupbit) {
    KK_FLOAT massone = 0.0;
    if (RMASS) massone = rmass[i];
    else massone = mass[type[i]];
    t_kk.t0 += static_cast<double>(massone * v(i,0)*v(i,0));
    t_kk.t1 += static_cast<double>(massone * v(i,1)*v(i,1));
    t_kk.t2 += static_cast<double>(massone * v(i,2)*v(i,2));
    t_kk.t3 += static_cast<double>(massone * v(i,0)*v(i,1));
    t_kk.t4 += static_cast<double>(massone * v(i,0)*v(i,2));
    t_kk.t5 += static_cast<double>(massone * v(i,1)*v(i,2));
  }
}

namespace LAMMPS_NS {
template class ComputeTempKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class ComputeTempKokkos<LMPHostType>;
#endif
}
