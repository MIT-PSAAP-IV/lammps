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

#include "compute_podd_atom.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include "eapod.h"

using namespace LAMMPS_NS;

enum { SCALAR, VECTOR, ARRAY };

ComputePODDAtom::ComputePODDAtom(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg), list(nullptr), podptr(nullptr), pod(nullptr), tmpmem(nullptr),
    rij(nullptr), elements(nullptr), map(nullptr), ai(nullptr), aj(nullptr), ti(nullptr),
    tj(nullptr)
{
  int nargmin = 6;

  if (narg < nargmin) error->all(FLERR, "Illegal compute {} command", style);

  std::string pod_file = std::string(arg[3]);      // pod input file
  std::string coeff_file = std::string(arg[4]);    // coefficient input file
  podptr = new EAPOD(lmp, pod_file, coeff_file);

  int ntypes = atom->ntypes;
  memory->create(map, ntypes + 1, "compute_pod_global:map");
  map_element2type(narg - 5, arg + 5, podptr->nelements);

  cutmax = podptr->rcutmax;

  nmax = 0;
  nijmax = 0;

  natoms_cols = atom->natoms;
  size_peratom_cols = 3 * natoms_cols * podptr->Mdesc * podptr->nClusters;
  if (size_peratom_cols > (MAXSMALLINT*1.0))
      error->all(FLERR, "Too many atoms ({}) for compute {}", natoms_cols, style);
  
  peratom_flag = 1;
}

/* ---------------------------------------------------------------------- */

ComputePODDAtom::~ComputePODDAtom()
{
  if (elements) {
    for (int i = 0; i < atom->ntypes; i++) delete[] elements[i];
    delete[] elements;
  }
  memory->destroy(map);
  memory->destroy(pod);
  delete podptr;
}

/* ---------------------------------------------------------------------- */

void ComputePODDAtom::init()
{
  if (force->pair == nullptr)
    error->all(FLERR,"Compute pod requires a pair style be defined");

  if (cutmax > force->pair->cutforce)
    error->all(FLERR,"Compute pod cutoff is longer than pairwise cutoff");

  // need an occasional full neighbor list

  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);

  if (modify->get_compute_by_style("pod").size() > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute pod");
  
  // check for consecutive ID tags:
  // If "delete_atoms" gives: tag - 1 >= natoms, pod[i] writes OOB.
  tagint maxtag_one = 0, maxtag;
  for (int i = 0; i < atom->nlocal; i++) maxtag_one = MAX(maxtag_one, atom->tag[i]);
  MPI_Allreduce(&maxtag_one, &maxtag, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  if (maxtag != atom->natoms)
    error->all(FLERR, "Compute {} requires consecutive atom IDs. Use: reset_atoms id", style);
}


/* ---------------------------------------------------------------------- */

void ComputePODDAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputePODDAtom::compute_peratom()
{
  invoked_peratom = update->ntimestep;

  if (atom->natoms != natoms_cols)
    error->all(FLERR, "Compute {}: number of atoms changed since compute was defined", style);

  // grow pod array if necessary (rows = local storage, cols = global derivatives)

  if (atom->nmax > nmax) {
    memory->destroy(pod);
    nmax = atom->nmax;
    memory->create(pod, nmax, size_peratom_cols, "podd/atom:pod");
    array_atom = pod;
  }

  const int nlocal = atom->nlocal;
  for (int i = 0; i < nlocal; i++)
    for (int icoeff = 0; icoeff < size_peratom_cols; icoeff++)
      pod[i][icoeff] = 0.0;

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  double **x = atom->x;
  int **firstneigh = list->firstneigh;
  int *mask = atom->mask;
  int *numneigh = list->numneigh;
  int *type = atom->type;
  int *ilist = list->ilist;
  int inum = list->inum;
  int nClusters = podptr->nClusters;
  int Mdesc = podptr->Mdesc;
  int nelements = podptr->nelements;
  bool localeapod = podptr->localeapod;
  double **rcutsq = podptr->rcutsq;

  const int MC = Mdesc*nClusters;

  // determine the maximum number of neighbor list candidates for all local atoms
  // and allocate temporary memory accordingly.  a minimum of one guarantees that
  // the buffers always exist, even if no atom has any neighbors at all.

  int jnummax = 1;
  for (int ii = 0; ii < inum; ii++) jnummax = MAX(jnummax, numneigh[ilist[ii]]);

  if (nijmax < jnummax) {
    nijmax = jnummax;
    podptr->free_temp_memory();
    podptr->allocate_temp_memory(nijmax);
  }

  rij = &podptr->tmpmem[0];
  tmpmem = &podptr->tmpmem[3*nijmax];
  ai = &podptr->tmpint[0];
  aj = &podptr->tmpint[nijmax];
  ti = &podptr->tmpint[2*nijmax];
  tj = &podptr->tmpint[3*nijmax];

  for (int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    if (!(mask[i] & groupbit)) continue;
    // get neighbor list for atom i
    lammpsNeighborList(x, firstneigh, atom->tag, type, numneigh, rcutsq, nelements, i);
    if (nij == 0) continue;
    // peratom base descriptors
    double *bd = &podptr->bd[0];
    double *bdd = &podptr->bdd[0];
    podptr->peratombase_descriptors(bd, bdd, rij, tmpmem, ti, tj, nij);
    if (nClusters>1) {
      // peratom env descriptors
      double *pd = &podptr->pd[0];
      double *pdd = &podptr->pdd[0];
      if (localeapod)
        podptr->peratomlocalenvironment_descriptors(pd, pdd, bd, bdd, tmpmem, ti[0], nij);
      else
        podptr->peratomenvironment_descriptors(pd, pdd, bd, bdd, tmpmem, ti[0], nij);
      for (int n=0; n<nij; n++) {
        int ain = 3*ai[n];
        int ajn = 3*aj[n];
        for (int k = 0; k < nClusters; k++) {
          for (int m = 0; m < Mdesc; m++) {
            int mk = m + Mdesc*k;
            int nm = 3*n + 3*nij*m;
            int nk = 3*n + 3*nij*k;
            pod[i][mk + MC*0 + MC*ain] += bdd[0 + nm]*pd[k] + bd[m]*pdd[0+nk];
            pod[i][mk + MC*1 + MC*ain] += bdd[1 + nm]*pd[k] + bd[m]*pdd[1+nk];
            pod[i][mk + MC*2 + MC*ain] += bdd[2 + nm]*pd[k] + bd[m]*pdd[2+nk];
            pod[i][mk + MC*0 + MC*ajn] -= bdd[0 + nm]*pd[k] + bd[m]*pdd[0+nk];
            pod[i][mk + MC*1 + MC*ajn] -= bdd[1 + nm]*pd[k] + bd[m]*pdd[1+nk];
            pod[i][mk + MC*2 + MC*ajn] -= bdd[2 + nm]*pd[k] + bd[m]*pdd[2+nk];
          }
        }
      }
    }
    else {
      for (int n=0; n<nij; n++) {
        int ain = 3*ai[n];
        int ajn = 3*aj[n];
        for (int m = 0; m < Mdesc; m++) {
          int nm = 3*n + 3*nij*m;  // d D_im/ d R_j, m = 1, ..., M, i = 1, ..., N, j = 1, ..., N
          pod[i][m + Mdesc*0 + Mdesc*ain] += bdd[0 + nm];
          pod[i][m + Mdesc*1 + Mdesc*ain] += bdd[1 + nm];
          pod[i][m + Mdesc*2 + Mdesc*ain] += bdd[2 + nm];
          pod[i][m + Mdesc*0 + Mdesc*ajn] -= bdd[0 + nm];
          pod[i][m + Mdesc*1 + Mdesc*ajn] -= bdd[1 + nm];
          pod[i][m + Mdesc*2 + Mdesc*ajn] -= bdd[2 + nm];
        }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double ComputePODDAtom::memory_usage()
{
  double bytes = 0.0;

  return bytes;
}


void ComputePODDAtom::lammpsNeighborList(double **x, int **firstneigh, tagint *atomid, int *atomtypes,
                               int *numneigh, double **rcutsq, int nelements, int gi)
{
  nij = 0;
  int itype = map[atomtypes[gi]];
  ti[nij] = itype;
  int m = numneigh[gi];
  for (int l = 0; l < m; l++) {           // loop over each atom around atom i
    int gj = firstneigh[gi][l];           // atom j
    int jtype = map[atomtypes[gj]];   // type of neighboring atom j
    double delx = x[gj][0] - x[gi][0];    // xj - xi
    double dely = x[gj][1] - x[gi][1];    // xj - xi
    double delz = x[gj][2] - x[gi][2];    // xj - xi
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq < rcutsq[itype][jtype] && rsq > 1e-20) {
      rij[nij * 3 + 0] = delx;
      rij[nij * 3 + 1] = dely;
      rij[nij * 3 + 2] = delz;
      ai[nij] = atomid[gi]-1;
      aj[nij] = atomid[gj]-1;
      ti[nij] = itype;
      tj[nij] = jtype;
      nij++;
    }
  }
}

void ComputePODDAtom::map_element2type(int narg, char **arg, int nelements)
{
  int i,j;
  const int ntypes = atom->ntypes;

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if "NULL"
  // nelements = # of unique elements
  // elements = list of element names

  if (narg != ntypes)
    error->all(FLERR, "Number of element to type mappings does not match number of atom types");

  if (elements) {
    for (i = 0; i < nelements; i++) delete[] elements[i];
    delete[] elements;
  }
  elements = new char*[ntypes];
  for (i = 0; i < ntypes; i++) elements[i] = nullptr;

  nelements = 0;
  map[0] = -1;
  for (i = 1; i <= narg; i++) {
    std::string entry = arg[i-1];
    if (entry == "NULL") {
      map[i] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (entry == elements[j]) break;
    map[i] = j;
    if (j == nelements) {
      elements[j] = utils::strdup(entry);
      nelements++;
    }
  }
}
