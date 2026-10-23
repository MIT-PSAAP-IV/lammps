/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(pod,PairPOD);
// clang-format on
#else

#ifndef LMP_PAIR_POD_H
#define LMP_PAIR_POD_H

#include "pair.h"

namespace LAMMPS_NS {

class PairPOD : public Pair {
 public:
  PairPOD(class LAMMPS *);
  ~PairPOD() override;
  void compute(int, int) override;

  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  double memory_usage() override;

  void lammpsNeighborList(double *rij, int *ai, int *aj, int *ti, int *tj, double **x,
                          int **firstneigh, int *type, int *map, int *numneigh, int i);
  void tallystress(double *fij, double *rij, int *ai, int *aj, int nlocal, int N);
  void tallyforce(double **force, double *fij, int *ai, int *aj, int N);

  void copy_data_from_pod_class();

 protected:
  class EAPOD *fastpodptr;
  virtual void allocate();

  int nij;       // total number of pairs (i,j)
  int nijmax;    // maximum number of pairs (i,j)

  int nelements; // number of elements

  double **rin;     // inner cut-off radius
  double **rcut;    // outer cut-off radius
  double **rcutsq;  // outer cut-off radius squared
};

}    // namespace LAMMPS_NS

#endif
#endif
