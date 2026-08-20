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

#ifndef LMP_EAPOD_H
#define LMP_EAPOD_H

#include "pointers.h"

#define DDOT ddot_
#define DGEMV dgemv_
#define DGEMM dgemm_
#define DGETRF dgetrf_
#define DGETRI dgetri_
#define DSYEV dsyev_
#define DPOSV dposv_

extern "C" {
double DDOT(int *, double *, int *, double *, int *);
void DGEMV(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *,
           int *);
void DGEMM(char *, char *, int *, int *, int *, double *, double *, int *, double *, int *,
           double *, double *, int *);
void DGETRF(int *, int *, double *, int *, int *, int *);
void DGETRI(int *, double *, int *, int *, double *, int *, int *);
void DSYEV(char *, char *, int *, double *, int *, double *, double *, int *, int *);
void DPOSV(char *, int *, int *, double *, int *, double *, int *, int *);
}

namespace LAMMPS_NS {

class EAPOD : protected Pointers {
 private:
  int indexmap3(int *indx, int n1, int n2, int n3, int N1, int N2);
  int crossindices(int *dabf1, int nabf1, int nrbf1, int nebf1, int *dabf2, int nabf2, int nrbf2,
                   int nebf2, int dabf12, int nrbf12);
  int crossindices(int *ind1, int *ind2, int *dabf1, int nabf1, int nrbf1, int nebf1, int *dabf2,
                   int nabf2, int nrbf2, int nebf2, int dabf12, int nrbf12);

  void init3bodyarray(int *np, int *pq, int *pc, int Pa3);

  void init4bodyarray(int *pa4, int *pb4, int *pc4, int Pa4);

  void init2body();

  void init3body(int Pa3);

  void init4body(int Pa4);

  void eigenvaluedecomposition(double *Phi, double *Lambda, double rin, double rmax, int N);
  
  void myneighbors(double *rij, double *x, int *ai, int *aj, int *ti, int *tj, int *jlist,
                   int *pairnumsum, int *atomtype, int *alist, int i);
  
  void init_bessel_const();

  void init_spline_radialbasis();
  void radialbasis_spline(double *rbf, double *drbf, double *rij, int *ti, int *tj, int N);

  void radialbasis(double *rbf, double *drbf, double *rij, double **rin, double **invrdiff,
                   int *ti, int *tj, int besseldegree, int inversedegree, int nbesselpars, int N);
  
  void radialPhi(double *rbf, double *drbf, double *rbft, double *drbft, int *ti, int *tj, int N);
  
  void angularbasis(double *abf, double *abfx, double *abfy, double *abfz, double *rij, double *tm,
                    int *pq, int N, int K);

  void radialangularsum(double *sumU, double *rbf, double *abf, int *tj, int Nj, int K, int M, int Ne);
  
  void radialangularbasis(double *sumU, double *Ux, double *Uy, double *Uz, double *rbf, double *drbf, double *rij,
                          double *abf, double *abfx, double *abfy, double *abfz,
                          double *tm, int *atomtype, int N, int K, int M, int Ne);

 public:
  std::vector<std::string> species;

  struct BesselConst {
    double neg_alpha;
    double pi_inv_t1;
    double dx_factor;
  };

  std::vector<BesselConst> bessel_const_storage;
  // bessel_const[ntypes][ntypes][nbesselpars]
  inline const BesselConst* bessel_const(int it, int jt) const {
   return &bessel_const_storage[(it * nelements + jt) * nbesselpars];
  }

  int **elemindex;
  double **rin;
  double **rcut;
  double **rcutsq;
  double **rdiff;
  double **invrdiff;
  double rcutmax;
  double rinmin;
  double rdiffmax;
  double invrdiffmax;

  int nelements;    // number of elements
  int pbc[3];

  int onebody;    // one-body descriptors
  int nbesselrbf;
  int besseldegree;
  int inversedegree;
  int nbesselpars;
  int timing;
  double comptime[20];
  double *besselparams;
  double *Phi;       // eigenvectors
  double *Lambda;    // eigenvalues
  double *coeff;     // coefficients
  // --- precomputed radial-basis spline (fast inference path) ---
  bool use_spline;          // enable cubic-Hermite spline evaluation of rbf
  int  nspline_grid;        // number of spline nodes per element pair
  int  nspline_bins;        // = nspline_grid - 1
  double *rbf_spline_coeffs;// [pair][bin][k][4] cubic Hermite coeffs (in local t)
  double *spline_r0;        // [nelements*nelements] grid start (r) per pair
  double *spline_invdr;     // [nelements*nelements] 1/bin-width per pair
  //double *newcoeff ;  // coefficients
  double *tmpmem;

  // environmental variables
  bool uncertaintyflag;
  bool eapod; // flag for EAPOD
  bool localeapod; // flag for Local-EAPOD
  int nClusters;      // number of environment clusters
  int clusterSearchBox; // Range to search for active clusters around centroids
  int nMaxActiveClusters; // maximum (possible) number of active clusters
  int nComponents;    // number of principal components
  //int nNeighbors; // numbe of neighbors
  int Mdesc;    // number of base descriptors
  double nActiveClusters; // average number of active clusters

  double *Proj;         // PCA Projection matrix
  double *Centroids;    // centroids of the clusters
  double *invLeftClusterRcut2;   // Left-hand side squared cutoff redius for each cluster (nClusters)
  double *invRightClusterRcut2;  // Right-hand side squared cutoff redius for each cluster (nClusters)
  double *leftClusterEdges;   // Left-hand side edge activation position for each cluster (nClusters)
  double *rightClusterEdges;  // Right-hand side edge activation position for each cluster (nClusters)
  double *invPcaSpan;         // 1/(span centroid) per PCA component and element type (nComponents*nelements)
  int *clusterOccupancy;      // training-atom count per cluster and element type (nClusters*nelements)

  double *bd;           // base descriptors
  double *bdd;          // derivatives of the base descriptors with respect to the atomic positions
  double *pd;           //  multi-environment descriptors
  double *pdd;          // derivative of the multi-environment descriptors with respect to the atomic positions

  int nproj;         // number of elements in projection matrix (nComponents * Mdesc * nelements)
  int ncentroids;    // number of centroids (nComponents * nClusters * nelements)

  int hat_p;   // order of hat basis function (cluster)
  int hat_q;   // order of hat function (cluster)
  int hat_p1;
  int hat_q1;
  int h_pq;

  int Njmax;
  int nCoeffPerElement;    // number of coefficients per element = (nl1 + Mdesc*nClusters)
  int nCoeffAll;    // number of coefficients for all elements = (nl1 + Mdesc*nClusters)*nelements
  int ncoeff;       // number of coefficients in the input file
  int ns;           // number of snapshots for radial basis functions

  int nd1, nd2, nd3, nd4, nd5, nd6, nd7, nd;    // number of global descriptors
  int nl1, nl2, nl3, nl4, nl5, nl6, nl7, nl;    // number of local descriptors
  int nrbf2, nrbf3, nrbf4, nrbfmax;             // number of radial basis functions
  int nabf3, nabf4;                             // number of angular basis functions
  int P3, P4;                                   // angular polynomial degrees
  int K3, K4, Q4;                               // number of monomials
  int *pn3, *pq3, *pc3;                         // arrays to compute 3-body angular basis functions
  int *pa4, *pb4, *pc4;                         // arrays to compute 4-body angular basis functions
  int *tmpint;
  int nintmem;    // number of integers in tmpint array
  int ndblmem;    // number of doubles in tmpmem array

  int L3min, L3max;
  int L4min, L4max;

  int nabf3_active, nabf4_active;
  int *p3_active, *p4_active;         // dense active -> full channel index
  int *dabf3_active, *dabf4_active;   // degree per active channel
  int *deg4_full;   // per-channel angular degree for full 4-body enumeration (length nabf4)

  // five-body descriptors
  int *ind33, nrbf33, nabf33, P33, n33, nl33, nd33;

  // six-body descriptors
  int *ind34, *ind43, nrbf34, nabf34, nabf43, P34, n34, n43, nl34, nd34;

  // seven-body descriptors
  int *ind44, nrbf44, nabf44, P44, n44, nl44, nd44;

  int nld33, nld34, nld44, ngd33, ngd34, ngd44;
  int *ind33l, *ind33r, *ind34l, *ind34r, *ind44l, *ind44r;

  EAPOD(LAMMPS *, const std::string &pod_file, const std::string &coeff_file);

  EAPOD(LAMMPS *lmp) : Pointers(lmp){};
  ~EAPOD() override;

  void read_pod_file(const std::string &pod_file);
  void read_model_coeff_file(const std::string &coeff_file);
  void read_cluster_occupancy_file(const std::string &coeff_file);

  int estimate_temp_memory(int Nj);
  void free_temp_memory();
  void allocate_temp_memory(int Nj);

  //void mknewcoeff();

  void mknewcoeff(double *c, int nc);

  static inline int binom(int n, int k);
  static inline int npa(int d);
  static inline int nmono(int d);
  static inline int trinom(int p, int q, int r);
  static inline int monomial_idx(int p, int q, int r);
  static inline int fourbody_channels(int Pa, int *pa, int *deg);

  void snapshots(double *rbf, double *xij, double rin, double rmax, int N);
  void init_active_angular_ranges();

  static inline void cutoff_exp(double r, double invrmax, double e_v, double &fcut, double &dfcut);
  static inline void cutoff_poly_sq(double r, double invrmax, double &fcut, double &dfcut);
  static inline void cutoff_hat(double r, double invrmax, double &fcut, double &dfcut);
  static inline void cluster_cutoff_hat(double pc, double inv_rcut2, double &fcut, double &dfcut);
  static inline void cluster_cutoff_poly_sq(double pc, double inv_rcut2, double &fcut, double &dfcut);
  static inline void cluster_cutoff_hat_train(double u, double &fcut, double &dfcut_du);
  static inline void cluster_cutoff_poly_sq_train(double u, double &fcut, double &dfcut);

  static inline void find_active_clusters(double pca, double* ledges, double* redges,
                                          const int nClusters, const int nMaxActiveClusters,
                                          int& ks, int& ke);

  void twobodydesc(double *d2, double *rbf, int *tj, int N, int Ne);
  void twobodydescderiv(double *d2, double *dd2, double *rbf, double *drbf, double *rij, int *tj, int N);
  void twobody_forces(double *fij, double *cb2, double *drbf, double *rij, int *tj, int Nj);

  void threebodydesc(double *d3, double *sumU, int Ne);
  void threebodydescderiv(double *dd3, double *sumU, double *Ux, double *Uy, double *Uz,
                          int *tj, int N);
  void threebody_forcecoeff(double *fb3, double *cb3, double *sumU);

  void fourbodydesc(double *d4, double *sumU);
  void fourbodydescderiv(double *d4, double *dd4, double *sumU, double *Ux, double *Uy, double *Uz,
                         int *tj, int N);
  void fourbody_forcecoeff(double *fb4, double *cb4, double *sumU);

  void allbody_forces(double *fij, double *forcecoeff, double *rbf, double *drbf, double *rij,
                      double *abf, double *abfx, double *abfy, double *abfz, int *tj, int Nj);

  void descriptors(double *gd, double *gdd, double *basedesc, double *probdesc, double *x,
                   int *atomtype, int *alist, int *jlist, int *pairnumsum, int natom);

  void descriptors(double *gd, double *gdd, double *basedesc, double *x, int *atomtype, int *alist,
                   int *jlist, int *pairnumsum, int natom);

  void peratombase_descriptors(double *bd, double *bdd, double *rij, double *temp, int *ti, int *tj, int Nj);
  double peratombase_coefficients(double *cb, double *bd, int *ti);
  double peratom_environment_descriptors(double *cb, double *bd, double *tm, int *ti);
  double peratom_local_environment_descriptors(double *cb, double *bd, double *tm, int *ti);

  void peratomenvironment_descriptors(double *P, double *dP_dR, double *B, double *dB_dR,
                                      double *tmp, int elem, int nNeighbors);

  void peratomlocalenvironment_descriptors(double *P, double *dP_dR, double *B, double *dB_dR,
                                      double *tmp, int elem, int nNeighbors);
  
  void base_descriptors(double *basedesc, double *x, int *atomtype, int *alist, int *jlist,
                        int *pairnumsum, int natom);

  void descriptors(double *basedesc, double *probdesc, double *x, int *atomtype, int *alist,
                   int *jlist, int *pairnumsum, int natom);

  double peratomenergyforce(double *fij, double *rij, double *temp, int *ti, int *tj, int Nj);
  double peratomenergyforce2(double *fij, double *rij, double *temp, int *ti, int *tj, int Nj);

  double energyforce(double *force, double *x, int *atomtype, int *alist, int *jlist,
                     int *pairnumsum, int natom);

  void tallyforce(double *force, double *fij, int *ai, int *aj, int N);

  void crossdesc(double *d12, double *d1, double *d2, int *ind1, int *ind2, int n12);
  void crossdescderiv(double *dd12, double *d1, double *d2, double *dd1, double *dd2, int *ind1,
                      int *ind2, int n12, int N);
  void crossdesc_reduction(double *cb1, double *cb2, double *c12, double *d1, double *d2, int *ind1,
                           int *ind2, int n12);

  void calculateClusterEdges(int nClusters, double nActiveClusters, int nComponents, int nelements);

  void calculatePcaSpan();

  void peratom_uncertainty(double *out, double *bd, double *bdd, int Nj, double *tm, int *ti);
};

}    // namespace LAMMPS_NS

#endif
