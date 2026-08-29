/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing authors: Ngoc Cuong Nguyen (MIT), Dionysios Sema (MIT),
                         Andrew Rohskopf (SNL)
------------------------------------------------------------------------- */

#include "pair_pod.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "info.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"
#include "safe_pointers.h"

#include <cstring>
#include <exception>
#include <algorithm>
#include <cmath>

#include "eapod.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024

/* ---------------------------------------------------------------------- */

PairPOD::PairPOD(LAMMPS *lmp) : Pair(lmp), fastpodptr(nullptr)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  centroidstressflag = CENTROID_NOTAVAIL;

  nij = 0;
  nijmax = 0;

  rin = nullptr;
  rcut = nullptr;
  rcutsq = nullptr;
}

/* ---------------------------------------------------------------------- */

PairPOD::~PairPOD()
{
  if (copymode) return;

  memory->destroy(rin);
  memory->destroy(rcut);
  memory->destroy(rcutsq);

  delete fastpodptr;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairPOD::compute(int eflag, int vflag)
{
  if (copymode) ev_init(eflag, vflag, 0);
  else ev_init(eflag, vflag, 1);

  double **x = atom->x;
  double **f = atom->f;
  int **firstneigh = list->firstneigh;
  int *numneigh = list->numneigh;
  int *type = atom->type;
  int *ilist = list->ilist;
  int inum = list->inum;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  double evdwl = 0.0;

  // determine the maximum number of neighbor list candidates for all local atoms
  // and allocate temporary memory accordingly.  a minimum of one guarantees that
  // the buffers always exist, even if no atom has any neighbors at all.

  int jnummax = 1;
  for (int ii = 0; ii < inum; ii++) jnummax = MAX(jnummax, numneigh[ilist[ii]]);

  if (nijmax < jnummax) {
    nijmax = jnummax;
    fastpodptr->grow_rij(nijmax);
  }

  double *rij = &fastpodptr->tmpmem[0];
  double *fij = &fastpodptr->tmpmem[3*nijmax];
  double *tmp = &fastpodptr->tmpmem[6*nijmax];
  int *ai = &fastpodptr->tmpint[0];
  int *aj = &fastpodptr->tmpint[nijmax];
  int *ti = &fastpodptr->tmpint[2*nijmax];
  int *tj = &fastpodptr->tmpint[3*nijmax];

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    
    lammpsNeighborList(rij, ai, aj, ti, tj, x, firstneigh, type, map, numneigh, i);

    evdwl = fastpodptr->peratomenergyforce2(fij, rij, tmp, ti, tj, nij);

    // tally atomic energy to global energy
    ev_tally_full(i,2.0*evdwl,0.0,0.0,0.0,0.0,0.0);

    // tally atomic force to global force
    tallyforce(f, fij, ai, aj, nij);

    // tally atomic stress
    if (vflag) {
      for (int jj = 0; jj < nij; jj++) {
        int j = aj[jj];
        ev_tally_xyz(i,j,nlocal,newton_pair,0.0,0.0,
                    fij[0 + 3*jj],fij[1 + 3*jj],fij[2 + 3*jj],
                    -rij[0 + 3*jj], -rij[1 + 3*jj], -rij[2 + 3*jj]);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPOD::settings(int narg, char ** /* arg */)
{
  if (narg > 0) error->all(FLERR, "Pair style pod accepts no arguments");

  // POD potentials are parameterized in metal units
  if (strcmp("metal", update->unit_style) != 0)
    error->all(FLERR, "POD potentials require 'metal' units");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairPOD::coeff(int narg, char **arg)
{
  if (narg < 5) utils::missing_cmd_args(FLERR, "pair_coeff", error);
  if (!allocated) allocate();

  std::string pod_file = std::string(arg[2]);      // pod input file
  std::string coeff_file = std::string(arg[3]);    // coefficient input file
  map_element2type(narg - 4, arg + 4);

  delete fastpodptr;
  fastpodptr = new EAPOD(lmp, pod_file, coeff_file);

  copy_data_from_pod_class();

  // reset nijmax to re-allocate based on neighbor counts
  nijmax = 0;

  for (int ii = 0; ii < nelements; ii++)
    for (int jj = 0; jj < nelements; jj++)
      cutsq[ii][jj] = rcutsq[ii][jj];
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairPOD::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "Pair style pod requires newton pair on");

  // need a full neighbor list

  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairPOD::init_one(int i, int j)
{
  if (setflag[i][j] == 0)
    error->all(FLERR, Error::NOLASTLINE,
               "All pair coeffs are not set. Status:\n" + Info::get_pair_coeff_status(lmp));

  int itype = i-1;
  int jtype = j-1;
  return rcut[itype][jtype];
}

void PairPOD::allocate()
{
  allocated = 1;
  int np1 = atom->ntypes + 1;
  memory->destroy(setflag);
  memory->destroy(cutsq);
  memory->create(setflag, np1, np1, "pair:setflag");
  memory->create(cutsq, np1, np1, "pair:cutsq");
  map = new int[np1];
}

void PairPOD::lammpsNeighborList(double *rij, int *ai, int *aj, int *ti, int *tj,
                               double **x, int **firstneigh, int *type, int *map,
                               int *numneigh, int gi)
{
  nij = 0;
  int itype = map[type[gi]];
  ti[nij] = itype;
  int m = numneigh[gi];
  for (int l = 0; l < m; l++) {
    int gj = firstneigh[gi][l];

    double delx = x[gj][0] - x[gi][0];
    double dely = x[gj][1] - x[gi][1];
    double delz = x[gj][2] - x[gi][2];
    double rsq = delx * delx + dely * dely + delz * delz;

    int jtype = map[type[gj]];
    if (rsq < cutsq[itype][jtype]) {
      rij[nij * 3 + 0] = delx;
      rij[nij * 3 + 1] = dely;
      rij[nij * 3 + 2] = delz;
      ai[nij] = gi;
      aj[nij] = gj;
      ti[nij] = itype;
      tj[nij] = jtype;
      nij++;
    }
  }
}

void PairPOD::tallyforce(double **force, double *fij,  int *ai, int *aj, int N)
{
  for (int n=0; n<N; n++) {
    int im =  ai[n];
    int jm =  aj[n];
    int nm = 3*n;
    force[im][0] += fij[0 + nm];
    force[im][1] += fij[1 + nm];
    force[im][2] += fij[2 + nm];
    force[jm][0] -= fij[0 + nm];
    force[jm][1] -= fij[1 + nm];
    force[jm][2] -= fij[2 + nm];
  }
}

/* ----------------------------------------------------------------------
   tally eng_vdwl and virial into global or per-atom accumulators
   for virial, have delx,dely,delz and fx,fy,fz
------------------------------------------------------------------------- */

void PairPOD::tallystress(double *fij, double *rij, int *ai, int *aj, int nlocal, int N)
{
  double v[6];

  for (int k = 0; k < N; k++) {
    int k3 = 3 * k;
    v[0] = -rij[0 + k3] * fij[0 + k3]; // delx*fx
    v[1] = -rij[1 + k3] * fij[1 + k3]; // dely*fy
    v[2] = -rij[2 + k3] * fij[2 + k3]; // delz*fz
    v[3] = -rij[0 + k3] * fij[1 + k3]; // delx*fy
    v[4] = -rij[0 + k3] * fij[2 + k3]; // delx*fz
    v[5] = -rij[1 + k3] * fij[2 + k3]; // dely*fz

    if (vflag_global) {
      virial[0] += v[0];
      virial[1] += v[1];
      virial[2] += v[2];
      virial[3] += v[3];
      virial[4] += v[4];
      virial[5] += v[5];
    }

    if (vflag_atom) {
      int i = ai[k];
      int j = aj[k];
      if (i < nlocal) {
        vatom[i][0] += 0.5 * v[0];
        vatom[i][1] += 0.5 * v[1];
        vatom[i][2] += 0.5 * v[2];
        vatom[i][3] += 0.5 * v[3];
        vatom[i][4] += 0.5 * v[4];
        vatom[i][5] += 0.5 * v[5];
      }
      if (j < nlocal) {
        vatom[j][0] += 0.5 * v[0];
        vatom[j][1] += 0.5 * v[1];
        vatom[j][2] += 0.5 * v[2];
        vatom[j][3] += 0.5 * v[3];
        vatom[j][4] += 0.5 * v[4];
        vatom[j][5] += 0.5 * v[5];
      }
    }
  }
}

void PairPOD::copy_data_from_pod_class()
{
  nelements = fastpodptr->nelements;

  memory->create(rin, nelements, nelements, "pair_pod:rin");
  memory->create(rcut, nelements, nelements, "pair_pod:rcut");
  memory->create(rcutsq, nelements, nelements, "pair_pod:rcutsq");
  for (int i=0; i < nelements; i++) {
    for (int j=0; j < nelements; j++) {
      rin[i][j] = fastpodptr->rin[i][j];
      rcut[i][j] = fastpodptr->rcut[i][j];
      rcutsq[i][j] = fastpodptr->rcutsq[i][j];
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double PairPOD::memory_usage()
{
  double bytes = Pair::memory_usage();
  return bytes;
}
