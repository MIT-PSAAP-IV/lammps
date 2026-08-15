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

#include "fitpod_command.h"

#include "comm.h"
#include "error.h"
#include "memory.h"
#include "safe_pointers.h"
#include "tokenizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <random>
#include <unordered_map>
#include <utility>

#include "eapod.h"

using namespace LAMMPS_NS;

static constexpr int MAXLINE = 1024;
static constexpr double SMALL = 1.0e-10;

FitPOD::datastruct::datastruct() :
    file_format("extxyz"), file_extension("xyz"), filenametag("pod"), group_weight_type("global"),
    lattice(nullptr), energy(nullptr), stress(nullptr), position(nullptr), force(nullptr),
    atomtype(nullptr), we(nullptr), wf(nullptr),
    fitting_weights{100.0, 1.0, 0.0, 1, 1, 0, 0, 1, 1, 1, 1, 1e-10,
                    /*12 kappa*/ 0.0, /*13 escheme*/ 0.0,
                    /*14 dE*/ 1.0,   /*15 dF*/ 1.0, /*16 Ftol2*/ 1e-8,   /*17 wzero*/ 1e-3,
                    /*18 w0*/ 0.0,   /*19 w1*/ 0.0,  /*20 w2*/ 0.0,
                    /*21 L2*/ 0.0, /*22 adapt*/ 0, /*23 rho_eps*/ 0.05}
{
  training = 1;
  normalizeenergy = 1;
  training_analysis = 1;
  test_analysis = 1;
  training_calculation = 0;
  test_calculation = 0;
  randomize = 1;
  precision = 8;
  fraction = 1.0;
}

void FitPOD::datastruct::copydatainfo(datastruct &data) const
{
  data.data_path = data_path;
  data.file_format = file_format;
  data.file_extension = file_extension;
  data.data_files = data_files;
  data.filenametag = filenametag;
  data.filenames = filenames;
  data.training_analysis = training_analysis;
  data.test_analysis = test_analysis;
  data.training_calculation = training_calculation;
  data.test_calculation = test_calculation;
  data.fraction = fraction;
  data.randomize = randomize;
  data.precision = precision;
  data.training = training;
  data.normalizeenergy = normalizeenergy;
  for (int i = 0; i < 24; i++) data.fitting_weights[i] = fitting_weights[i];
  data.we_map = we_map;
  data.wf_map = wf_map;
}

FitPOD::neighborstruct::neighborstruct() :
    alist(nullptr), pairnum(nullptr), pairnum_cumsum(nullptr), pairlist(nullptr), y(nullptr)
{
  natom_max = 0;
  sze = 0;
  sza = 0;
  szy = 0;
  szp = 0;
}

FitPOD::descriptorstruct::descriptorstruct() :
    bd(nullptr), pd(nullptr), gd(nullptr), gdd(nullptr), A(nullptr), b(nullptr), c(nullptr)
{
  szd = 0;
  nCoeffAll = 0;
  nClusters = 0;
}

FitPOD::FitPOD(LAMMPS *_lmp) : Command(_lmp), fastpodptr(nullptr)
{
  save_descriptors = 0;
  compute_descriptors = 0;
  save_pca_histogram = 0;
  pca_histogram_num_bins = 100;
}

void FitPOD::command(int narg, char **arg)
{
  if (narg < 2) utils::missing_cmd_args(FLERR, "fitpod", error);

  std::string pod_file = std::string(arg[0]);      // pod input file
  std::string data_file = std::string(arg[1]);     // data input file
  std::string coeff_file, proj_file, cent_file;    // coefficient input files

  if (narg > 2)
    coeff_file = std::string(arg[2]);    // coefficient input file
  else
    coeff_file = "";

  fastpodptr = new EAPOD(lmp, pod_file, coeff_file);

  desc.nCoeffAll = fastpodptr->nCoeffAll;
  desc.nClusters = fastpodptr->nClusters;
  read_data_files(data_file, fastpodptr->species);

  estimate_memory_neighborstruct(traindata, fastpodptr->pbc, fastpodptr->rcutmax,
                                 fastpodptr->nelements);
  estimate_memory_neighborstruct(testdata, fastpodptr->pbc, fastpodptr->rcutmax,
                                 fastpodptr->nelements);
  if (((int) envdata.data_path.size() > 1))
    estimate_memory_neighborstruct(envdata, fastpodptr->pbc, fastpodptr->rcutmax,
                                   fastpodptr->nelements);
  allocate_memory_neighborstruct();
  estimate_memory_fastpod(traindata);
  estimate_memory_fastpod(testdata);
  allocate_memory_descriptorstruct(fastpodptr->nCoeffAll);

  if (coeff_file != "") podArrayCopy(desc.c, fastpodptr->coeff, fastpodptr->nCoeffAll);

  if (((int) envdata.data_path.size() > 1)) {
    const bool same_env_and_train_path = (envdata.data_path == traindata.data_path);

    if (same_env_and_train_path) {
      // Same dataset paths: do projection + clustering once, skip training clustering pass
      if (comm->me == 0) {
        utils::logmesg(lmp, "same env and train data paths, reusing env clustering descriptors.\n");
      }
      environment_cluster_calculation(envdata);
    } else {
      // Different datasets: split workflow
      environment_proj_calculation(envdata);
    }

    deallocate_memory_datastruct(envdata);

    if (!same_env_and_train_path) {
      training_cluster_calculation(traindata);
    }
  }

  if (compute_descriptors == 0) {

    // compute POD coefficients using least-squares method
    if (coeff_file == "") {
      least_squares_fit(traindata);

      if (comm->me == 0) {    // save coefficients into a text file
        std::string filename = traindata.filenametag + "_coefficients" + ".pod";
        SafeFilePtr fp = fopen(filename.c_str(), "w");

        int nCoeffAll = desc.nCoeffAll;
        int n1 = 0, n2 = 0;
        if (((int) envdata.data_path.size() > 1)) {
          n1 = fastpodptr->nComponents * fastpodptr->Mdesc * fastpodptr->nelements;
          n2 = fastpodptr->nComponents * fastpodptr->nClusters * fastpodptr->nelements;
        }

        utils::print(fp, "model_coefficients: {} {} {}\n", nCoeffAll, n1, n2);
        for (int count = 0; count < nCoeffAll; count++) {
          utils::print(fp, "{:<10.{}f}\n", desc.c[count], traindata.precision);
        }
        for (int count = 0; count < n1; count++) {
          utils::print(fp, "{:<10.{}f}\n", fastpodptr->Proj[count], 14);
        }
        for (int count = 0; count < n2; count++) {
          utils::print(fp, "{:<10.{}f}\n", fastpodptr->Centroids[count], 14);
        }
      }
    }

    // calculate errors for the training data set

    if ((traindata.training_analysis) && ((int) traindata.data_path.size() > 1))
      error_analysis(traindata, desc.c);

    //error->all(FLERR, "stop after error_analysis");

    // calculate energy and force for the training data set

    if ((traindata.training_calculation) && ((int) traindata.data_path.size() > 1))
      energyforce_calculation(traindata);

    if (!((testdata.data_path == traindata.data_path) && (testdata.fraction == 1.0) &&
          (traindata.fraction == 1.0))) {
      // calculate errors for the test data set

      if ((testdata.test_analysis) && ((int) testdata.data_path.size() > 1) &&
          (testdata.fraction > 0)) {
        error_analysis(testdata, desc.c);
      }

      // calculate energy and force for the test data set

      if ((testdata.test_analysis) && (testdata.test_calculation) &&
          ((int) testdata.data_path.size() > 1) && (testdata.fraction > 0))
        energyforce_calculation(testdata);

      // deallocate testing data

      if ((int) testdata.data_path.size() > 1 && (testdata.test_analysis) &&
          (testdata.fraction > 0)) {
        deallocate_memory_datastruct(testdata);
      }
    }
  } else if (compute_descriptors > 0) {
    // compute and save POD descriptors
    descriptors_calculation(traindata);

    if (!((testdata.data_path == traindata.data_path) && (testdata.fraction == 1.0))) {
      if ((int) testdata.data_path.size() > 1) {
        descriptors_calculation(testdata);
        deallocate_memory_datastruct(testdata);
      }
    }
  }

  // deallocate training data

  if ((int) traindata.data_path.size() > 1) {
    deallocate_memory_datastruct(traindata);
  }

  // deallocate descriptors

  memory->destroy(desc.A);
  memory->destroy(desc.b);
  memory->destroy(desc.c);
  memory->destroy(desc.bd);
  memory->destroy(desc.pd);
  memory->destroy(desc.gd);
  memory->destroy(desc.gdd);

  // // deallocate neighbor data
  memory->destroy(nb.alist);
  memory->destroy(nb.pairnum);
  memory->destroy(nb.pairnum_cumsum);
  memory->destroy(nb.pairlist);
  memory->destroy(nb.y);

  delete fastpodptr;
}

int FitPOD::read_data_file(double *fitting_weights, std::string &file_format,
                           std::string &file_extension, std::string &env_path,
                           std::string &test_path, std::string &training_path,
                           std::string &filenametag, const std::string &data_file,
                           std::string &group_weight_type,
                           std::unordered_map<std::string, double> &we_map,
                           std::unordered_map<std::string, double> &wf_map)
{
  int precision = 8;

  std::string datafilename = data_file;
  SafeFilePtr fpdata;
  if (comm->me == 0) {

    fpdata = utils::open_potential(datafilename, lmp, nullptr);
    if (fpdata == nullptr)
      error->one(FLERR, "Cannot open training data file {}: ", datafilename, utils::getsyserror());
  }

  // loop through lines of training data file and parse keywords

  char line[MAXLINE], *ptr;
  int eof = 0;
  while (true) {
    if (comm->me == 0) {
      ptr = fgets(line, MAXLINE, fpdata);
      if (ptr == nullptr) {
        eof = 1;
      }
    }
    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) break;
    MPI_Bcast(line, MAXLINE, MPI_CHAR, 0, world);

    // words = ptrs to all words in line
    // strip single and double quotes from words

    std::vector<std::string> words;
    try {
      words = Tokenizer(utils::trim_comment(line), "\"' \t\n\r\f").as_vector();
    } catch (TokenizerException &) {
      // ignore
    }

    if (words.size() == 0) continue;

    auto keywd = words[0];

    if (words.size() != 2) error->one(FLERR, "Improper POD data file.", utils::getsyserror());

    // settings for fitting weights

    if (keywd == "fitting_weight_energy")
      fitting_weights[0] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "fitting_weight_force")
      fitting_weights[1] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "fitting_weight_stress")
      fitting_weights[2] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "error_analysis_for_training_data_set")
      fitting_weights[3] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "error_analysis_for_test_data_set")
      fitting_weights[4] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "energy_force_calculation_for_training_data_set")
      fitting_weights[5] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "energy_force_calculation_for_test_data_set")
      fitting_weights[6] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "fraction_training_data_set")
      fitting_weights[7] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "fraction_test_data_set")
      fitting_weights[8] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "randomize_training_data_set")
      fitting_weights[9] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "randomize_test_data_set")
      fitting_weights[10] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "fitting_regularization_parameter")
      fitting_weights[11] = utils::numeric(FLERR, words[1], false, lmp);
    if (keywd == "fitting_weight_kappa")
      fitting_weights[12] = utils::numeric(FLERR,words[1],false,lmp);
    if (keywd == "energy_weighting_scheme")
      fitting_weights[13] = utils::numeric(FLERR,words[1],false,lmp); // 0=legacy, 1=pace, 2=flatE+paceF
    if (keywd == "energy_weight_shift")
      fitting_weights[14] = utils::numeric(FLERR,words[1],false,lmp); // ΔE
    if (keywd == "force_weight_shift")
      fitting_weights[15] = utils::numeric(FLERR,words[1],false,lmp); // ΔF
    if (keywd == "zero_force_tol")
      fitting_weights[16] = utils::numeric(FLERR,words[1],false,lmp); // |F|^2 below this = "zero force"
    if (keywd == "zero_force_weight")
      fitting_weights[17] = utils::numeric(FLERR,words[1],false,lmp); // fixed weight for those atoms
    if (keywd == "radial_smoothness_w0")
      fitting_weights[18] = utils::numeric(FLERR,words[1],false,lmp);
    if (keywd == "radial_smoothness_w1")
      fitting_weights[19] = utils::numeric(FLERR,words[1],false,lmp);
    if (keywd == "radial_smoothness_w2")
      fitting_weights[20] = utils::numeric(FLERR,words[1],false,lmp);
    if (keywd == "l2_regularization")
      fitting_weights[21] = utils::numeric(FLERR,words[1],false,lmp);
    if (keywd == "radial_density_adapt")
      fitting_weights[22] = utils::numeric(FLERR,words[1],false,lmp); // 0/1 enable
    if (keywd == "radial_density_eps")
      fitting_weights[23] = utils::numeric(FLERR,words[1],false,lmp); // eps in 1/(rho+eps)
    
    if (keywd == "precision_for_pod_coefficients")
      precision = utils::inumeric(FLERR, words[1], false, lmp);
    if (keywd == "save_pod_descriptors")
      save_descriptors = utils::inumeric(FLERR, words[1], false, lmp);
    if (keywd == "compute_pod_descriptors")
      compute_descriptors = utils::inumeric(FLERR, words[1], false, lmp);
    if (keywd == "save_pca_histogram")
      save_pca_histogram = utils::inumeric(FLERR, words[1], false, lmp);
    if (keywd == "pca_histogram_num_bins")
      pca_histogram_num_bins = utils::inumeric(FLERR, words[1], false, lmp);

    // other settings

    if (keywd == "file_format") file_format = words[1];
    if (keywd == "file_extension") file_extension = words[1];
    if (keywd == "path_to_training_data_set") training_path = words[1];
    if (keywd == "path_to_test_data_set") test_path = words[1];
    if (keywd == "path_to_environment_configuration_set") env_path = words[1];
    if (keywd == "basename_for_output_files") filenametag = words[1];

    // group weight table
    if (keywd == "group_weights") group_weight_type = words[1];
    if (std::strcmp(group_weight_type.c_str(), "table") == 0) {
      // Read the table as a hash map.
      // Get next line.
      if (comm->me == 0) {
        ptr = fgets(line, MAXLINE, fpdata);
        if (ptr == nullptr) eof = 1;
      }
      MPI_Bcast(&eof, 1, MPI_INT, 0, world);
      if (eof) break;
      MPI_Bcast(line, MAXLINE, MPI_CHAR, 0, world);
      // Tokenize.
      try {
        words = Tokenizer(utils::trim_comment(line), "\"' \t\n\r\f").as_vector();
      } catch (TokenizerException &) {
        // ignore
      }
      int numwords = words.size();

      // Loop over group table entries.
      while (numwords == 3) {

        // Insert in map.
        we_map[words[0]] = utils::numeric(FLERR, words[1], false, lmp);
        wf_map[words[0]] = utils::numeric(FLERR, words[2], false, lmp);

        // Get next line.
        if (comm->me == 0) {
          ptr = fgets(line, MAXLINE, fpdata);
          if (ptr == nullptr) eof = 1;
        }
        MPI_Bcast(&eof, 1, MPI_INT, 0, world);
        if (eof) break;
        MPI_Bcast(line, MAXLINE, MPI_CHAR, 0, world);
        // Tokenize.
        try {
          words = Tokenizer(utils::trim_comment(line), "\"' \t\n\r\f").as_vector();
        } catch (TokenizerException &) {
          // ignore
        }
        numwords = words.size();
      }
    }
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "**************** Begin of Data File ****************\n");
    utils::logmesg(lmp, "file format: {}\n", file_format);
    utils::logmesg(lmp, "file extension: {}\n", file_extension);
    utils::logmesg(lmp, "path to training data set: {}\n", training_path);
    utils::logmesg(lmp, "path to test data set: {}\n", test_path);
    utils::logmesg(lmp, "path to environment configuration set: {}\n", env_path);
    utils::logmesg(lmp, "basename for output files: {}\n", filenametag);
    utils::logmesg(lmp, "training fraction: {}\n", fitting_weights[7]);
    utils::logmesg(lmp, "test fraction: {}\n", fitting_weights[8]);
    utils::logmesg(lmp, "randomize training data set: {}\n", fitting_weights[9]);
    utils::logmesg(lmp, "randomize test data set: {}\n", fitting_weights[10]);
    utils::logmesg(lmp, "error analysis for training data set: {}\n", fitting_weights[3]);
    utils::logmesg(lmp, "error analysis for test data set: {}\n", fitting_weights[4]);
    utils::logmesg(lmp, "energy/force calculation for training data set: {}\n", fitting_weights[5]);
    utils::logmesg(lmp, "energy/force calculation for test data set: {}\n", fitting_weights[6]);
    utils::logmesg(lmp, "fitting weight for energy: {}\n", fitting_weights[0]);
    utils::logmesg(lmp, "fitting weight for force: {}\n", fitting_weights[1]);
    utils::logmesg(lmp, "fitting weight for stress: {}\n", fitting_weights[2]);
    utils::logmesg(lmp, "fitting_regularization_parameter: {}\n", fitting_weights[11]);
    utils::logmesg(lmp, "fitting_weight_kappa: {}\n", fitting_weights[12]);
    utils::logmesg(lmp, "energy_weighting_scheme: {}\n", fitting_weights[13]);
    utils::logmesg(lmp, "energy_weight_shift: {}\n", fitting_weights[14]);
    utils::logmesg(lmp, "force_weight_shift: {}\n", fitting_weights[15]);
    utils::logmesg(lmp, "zero_force_tol: {}\n", fitting_weights[16]);
    utils::logmesg(lmp, "zero_force_weight: {}\n", fitting_weights[17]);
    utils::logmesg(lmp, "radial_smoothness_w0: {}\n", fitting_weights[18]);
    utils::logmesg(lmp, "radial_smoothness_w1: {}\n", fitting_weights[19]);
    utils::logmesg(lmp, "radial_smoothness_w2: {}\n", fitting_weights[20]);
    utils::logmesg(lmp, "l2_regularization: {}\n", fitting_weights[21]);
    utils::logmesg(lmp, "radial_density_adapt: {}\n", fitting_weights[22]);
    utils::logmesg(lmp, "radial_density_eps: {}\n", fitting_weights[23]);
    utils::logmesg(lmp, "save pod descriptors: {}\n", save_descriptors);
    utils::logmesg(lmp, "compute pod descriptors: {}\n", compute_descriptors);
    utils::logmesg(lmp, "save pca histogram: {}\n", save_pca_histogram);
    utils::logmesg(lmp, "pca histogram number of bins: {}\n", pca_histogram_num_bins);
    utils::logmesg(lmp, "**************** End of Data File ****************\n");
  }

  return precision;
}

void FitPOD::get_exyz_files(std::vector<std::string> &files, std::vector<std::string> &group_names,
                            const std::string &datapath, const std::string &extension)
{
  auto allfiles = platform::list_directory(datapath);
  std::sort(allfiles.begin(), allfiles.end());
  for (const auto &fname : allfiles) {
    if (utils::strmatch(fname, fmt::format(".*\\.{}$", extension))) {
      files.push_back(datapath + platform::filepathsep + fname); // NOLINT
      int start_pos_erase = fname.find(extension) - 1;
      std::string substr = fname.substr(0, start_pos_erase);
      group_names.push_back(substr);
    }
  }
}

int FitPOD::get_number_atom_exyz(std::vector<int> &num_atom, int &num_atom_sum, std::string file)
{
  std::string filename = std::move(file);
  SafeFilePtr fp;
  if (comm->me == 0) {
    fp = utils::open_potential(filename, lmp, nullptr);
    if (fp == nullptr)
      error->one(FLERR, "Cannot open POD coefficient file {}: ", filename, utils::getsyserror());
  }

  char line[MAXLINE], *ptr;
  int eof = 0;
  int num_configs = 0;
  num_atom_sum = 0;

  // loop over all lines of this xyz file and extract number of atoms and number of configs

  while (true) {
    if (comm->me == 0) {
      ptr = fgets(line, MAXLINE, fp);
      if (ptr == nullptr) {
        eof = 1;
      }
    }
    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) break;
    MPI_Bcast(line, MAXLINE, MPI_CHAR, 0, world);

    // words = ptrs to all words in line
    // strip single and double quotes from words

    std::vector<std::string> words;
    try {
      words = Tokenizer(utils::trim_comment(line), "\"' \t\n\r\f").as_vector();
    } catch (TokenizerException &) {
      // ignore
    }

    if (words.size() == 0) continue;

    int natom;
    if (words.size() == 1) {
      natom = utils::inumeric(FLERR, words[0], false, lmp);
      num_atom.push_back(natom);
      num_configs += 1;
      num_atom_sum += natom;
    }
  }
  return num_configs;
}

int FitPOD::get_number_atoms(std::vector<int> &num_atom, std::vector<int> &num_atom_sum,
                             std::vector<int> &num_config, std::vector<std::string> training_files)
{
  int nfiles = training_files.size();    // number of files
  int d, n;

  for (int i = 0; i < nfiles; i++) {
    d = get_number_atom_exyz(num_atom, n, training_files[i]);
    num_config.push_back(d);
    num_atom_sum.push_back(n);
  }

  int num_atom_all = 0;
  for (auto i : num_atom) num_atom_all += i;

  return num_atom_all;
}

void FitPOD::read_exyz_file(double *lattice, double *stress, double *energy, double *we, double *wf,
                            double *pos, double *forces, int *atomtype, std::string file,
                            std::vector<std::string> species, double we_group, double wf_group)
{

  std::string filename = std::move(file);
  SafeFilePtr fp;
  if (comm->me == 0) {
    fp = utils::open_potential(filename, lmp, nullptr);
    if (fp == nullptr)
      error->one(FLERR, "Cannot open POD coefficient file {}: ", filename, utils::getsyserror());
  }

  char line[MAXLINE], *ptr;
  int eof = 0;
  int cfi = 0;
  int nat = 0;
  int ns = species.size();

  // loop over all lines of this xyz file and extract training data

  while (true) {
    if (comm->me == 0) {
      ptr = fgets(line, MAXLINE, fp);
      if (ptr == nullptr) {
        eof = 1;
      }
    }
    MPI_Bcast(&eof, 1, MPI_INT, 0, world);
    if (eof) break;
    MPI_Bcast(line, MAXLINE, MPI_CHAR, 0, world);

    // words = ptrs to all words in line
    // strip single and double quotes from words

    std::vector<std::string> words;
    try {
      words = Tokenizer(utils::trim_comment(line), "\"' \t\n\r\f").as_vector();
    } catch (TokenizerException &) {
      // ignore
    }

    if (words.size() == 0) continue;

    ValueTokenizer text(utils::trim_comment(line), "\"' \t\n\r\f");
    if (text.contains("attice")) {

      // find the word containing "lattice"

      auto it = std::find_if(words.begin(), words.end(), [](const std::string &str) {
        return str.find("attice") != std::string::npos;
      });

      // get index of element from iterator

      int index = std::distance(words.begin(), it);

      if (words[index].find('=') != std::string::npos) {

        // lattice numbers start at index + 1

        for (int k = 0; k < 9; k++) {
          lattice[k + 9 * cfi] = utils::numeric(FLERR, words[index + 1 + k], false, lmp);
        }
      } else {

        // lattice numbers start at index + 2

        for (int k = 0; k < 9; k++) {
          lattice[k + 9 * cfi] = utils::numeric(FLERR, words[index + 2 + k], false, lmp);
        }
      }

      if (compute_descriptors == 0) {

        // find the word containing "energy"

        it = std::find_if(words.begin(), words.end(), [](const std::string &str) {
          return str.find("nergy") != std::string::npos;
        });

        // get index of element from iterator

        index = std::distance(words.begin(), it);

        if (words[index].find('=') != std::string::npos) {

          // energy is after "=" inside this string

          std::size_t found = words[index].find('=');
          energy[cfi] = utils::numeric(FLERR, words[index].substr(found + 1), false, lmp);
        } else {

          // energy is at index + 2

          energy[cfi] = utils::numeric(FLERR, words[index + 2], false, lmp);
        }

        // find the word containing "stress"

        it = std::find_if(words.begin(), words.end(), [](const std::string &str) {
          return str.find("tress") != std::string::npos;
        });

        // get index of element from iterator

        index = std::distance(words.begin(), it);

        if (index < std::distance(words.begin(), words.end())) {
          if (words[index].find('=') != std::string::npos) {

            // stress numbers start at index + 1

            for (int k = 0; k < 9; k++) {
              stress[k + 9 * cfi] = utils::numeric(FLERR, words[index + 1 + k], false, lmp);
            }
          } else {

            // lattice numbers start at index + 2

            for (int k = 0; k < 9; k++) {
              stress[k + 9 * cfi] = utils::numeric(FLERR, words[index + 2 + k], false, lmp);
            }
          }
        }
      }

      // set fitting weights for this config

      we[cfi] = we_group;
      wf[cfi] = wf_group;

      cfi += 1;
    }

    // loop over atoms

    else if (words.size() > 1) {

      for (int ii = 0; ii < ns; ii++)
        if (species[ii] == words[0]) atomtype[nat] = ii + 1;

      if (compute_descriptors > 0) {
        for (int k = 0; k < 3; k++)
          pos[k + 3 * nat] = utils::numeric(FLERR, words[1 + k], false, lmp);
      } else {
        for (int k = 0; k < 6; k++) {
          if (k <= 2) pos[k + 3 * nat] = utils::numeric(FLERR, words[1 + k], false, lmp);
          if (k > 2) forces[k - 3 + 3 * nat] = utils::numeric(FLERR, words[1 + k], false, lmp);
        }
      }

      nat += 1;
    }
  }
}

void FitPOD::get_data(datastruct &data, const std::vector<std::string> &species)
{
  get_exyz_files(data.data_files, data.group_names, data.data_path, data.file_extension);
  data.num_atom_sum =
      get_number_atoms(data.num_atom, data.num_atom_each_file, data.num_config, data.data_files);
  data.num_config_sum = data.num_atom.size();
  size_t maxname = 9;
  for (const auto &fname : data.data_files) maxname = MAX(maxname, fname.size());
  maxname -= data.data_path.size() + 1;
  const std::string sepline(maxname + 46, '-');
  if (comm->me == 0)
    utils::logmesg(lmp, "{}\n {:^{}} | number of configurations | number of atoms\n{}\n", sepline,
                   "data file", maxname, sepline);
  int i = 0;
  for (const auto &fname : data.data_files) {
    std::string filename = fname.substr(data.data_path.size() + 1);
    data.filenames.push_back(filename);
    if (comm->me == 0)
      utils::logmesg(lmp, " {:<{}} |        {:>10}        |    {:>8}\n", filename, maxname,
                     data.num_config[i], data.num_atom_each_file[i]);
    ++i;
  }
  if (comm->me == 0) {
    utils::logmesg(lmp, "{}\n", sepline);
    utils::logmesg(lmp, "number of files: {}\n", data.data_files.size());
    utils::logmesg(lmp, "number of configurations in all files: {}\n", data.num_config_sum);
    utils::logmesg(lmp, "number of atoms in all files: {}\n", data.num_atom_sum);
  }

  if (data.data_files.size() < 1)
    error->all(FLERR,
               "Cannot fit potential without data files. The data paths may not be valid. Please "
               "check the data paths in the POD data file.");

  int n = data.num_config_sum;
  memory->create(data.lattice, 9 * n, "fitpod:lattice");
  memory->create(data.stress, 9 * n, "fitpod:stress");
  memory->create(data.energy, n, "fitpod:energy");
  // Group weights have same size as energy.
  memory->create(data.we, n, "fitpod:we");
  memory->create(data.wf, n, "fitpod:wf");

  n = data.num_atom_sum;
  memory->create(data.position, 3 * n, "fitpod:position");
  memory->create(data.force, 3 * n, "fitpod:force");
  memory->create(data.atomtype, n, "fitpod:atomtype");

  double we_group, wf_group;              // group weights
  int nfiles = data.data_files.size();    // number of files
  int nconfigs = 0;
  int natoms = 0;
  for (int i = 0; i < nfiles; i++) {
    std::string group_name = data.group_names[i];
    // If weight maps have this group, assign weight based on map.
    // Else assign weight based on global value.
    if (data.we_map.find(group_name) != data.we_map.end()) {
      we_group = data.we_map[group_name];
      wf_group = data.wf_map[group_name];
    } else {
      we_group = data.fitting_weights[0];
      wf_group = data.fitting_weights[1];
    }
    //utils::logmesg(lmp, "Read xyz file: {}\n", group_name);
    read_exyz_file(&data.lattice[9 * nconfigs], &data.stress[9 * nconfigs], &data.energy[nconfigs],
                   &data.we[nconfigs], &data.wf[nconfigs], &data.position[3 * natoms],
                   &data.force[3 * natoms], &data.atomtype[natoms], data.data_files[i], species,
                   we_group, wf_group);
    nconfigs += data.num_config[i];
    natoms += data.num_atom_each_file[i];
  }

  int len = data.num_atom.size();
  data.num_atom_min = podArrayMin(data.num_atom.data(), len);
  data.num_atom_max = podArrayMax(data.num_atom.data(), len);
  data.num_atom_cumsum.resize(len + 1);
  podCumsum(data.num_atom_cumsum.data(), data.num_atom.data(), len + 1);

  data.num_config_cumsum.resize(nfiles + 1);
  podCumsum(data.num_config_cumsum.data(), data.num_config.data(), nfiles + 1);

  // convert all structures to triclinic system

  constexpr int DIM = 3;
  double Qmat[DIM * DIM];
  for (int ci = 0; ci < len; ci++) {
    int natom = data.num_atom[ci];
    int natom_cumsum = data.num_atom_cumsum[ci];
    double *x = &data.position[DIM * natom_cumsum];
    double *f = &data.force[DIM * natom_cumsum];
    double *lattice = &data.lattice[9 * ci];
    double *a1 = &lattice[0];
    double *a2 = &lattice[3];
    double *a3 = &lattice[6];

    matrix33_inverse(Qmat, a1, a2, a3);
    triclinic_lattice_conversion(a1, a2, a3, a1, a2, a3);
    matrix33_multiplication(Qmat, lattice, Qmat, DIM);
    matrix33_multiplication(x, Qmat, x, natom);
    matrix33_multiplication(f, Qmat, f, natom);
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "minimum number of atoms: {}\n", data.num_atom_min);
    utils::logmesg(lmp, "maximum number of atoms: {}\n", data.num_atom_max);
  }
}

std::vector<int> FitPOD::linspace(int start_in, int end_in, int num_in)
{

  std::vector<int> linspaced;

  auto start = static_cast<double>(start_in);
  auto end = static_cast<double>(end_in);
  auto num = static_cast<double>(num_in);

  int elm;

  if (num == 0) { return linspaced; }
  if (num == 1) {
    elm = (int) std::round(start);
    linspaced.push_back(elm);
    return linspaced;
  }

  double delta = (end - start) / (num - 1);

  for (int i = 0; i < num - 1; ++i) {
    elm = (int) std::round(start + delta * i);
    linspaced.push_back(elm);
  }

  elm = (int) std::round(end);
  linspaced.push_back(elm);

  return linspaced;
}

std::vector<int> FitPOD::shuffle(int start_in, int end_in, int num_in)
{
  int sz = end_in - start_in + 1;
  std::vector<int> myvector(sz);

  for (int i = 0; i < sz; i++) myvector[i] = start_in + i;

  //unsigned seed = (unsigned) platform::walltime()*1.0e9;
  //std::shuffle (myvector.begin(), myvector.end(), std::default_random_engine(seed));
  std::shuffle(myvector.begin(), myvector.end(), std::random_device());

  std::vector<int> shuffle_vec(num_in);
  for (int i = 0; i < num_in; i++) shuffle_vec[i] = myvector[i];

  return shuffle_vec;
}

std::vector<int> FitPOD::select(int n, double fraction, int randomize)
{
  std::vector<int> selected;

  int m = (int) std::round(n * fraction);
  m = MAX(m, 1);

  selected = (randomize == 1) ? shuffle(1, n, m) : linspace(1, n, m);

  return selected;
}

void FitPOD::select_data(datastruct &newdata, const datastruct &data)
{
  double fraction = data.fraction;
  int randomize = data.randomize;

  if (comm->me == 0) {
    if (randomize == 1)
      utils::logmesg(lmp, "Select {} fraction of the data set at random using shuffle\n",
                     data.fraction);
    else
      utils::logmesg(lmp, "Select {} fraction of the data set deterministically using linspace\n",
                     data.fraction);
  }

  int nfiles = data.data_files.size();    // number of files
  std::vector<std::vector<int>> selected(nfiles);

  newdata.num_config.resize(nfiles);
  newdata.num_config_cumsum.resize(nfiles + 1);
  newdata.num_atom_each_file.resize(nfiles);

  for (int file = 0; file < nfiles; file++) {
    int nconfigs = data.num_config[file];
    selected[file] = select(nconfigs, fraction, randomize);
    int ns = (int) selected[file].size();    // number of selected configurations

    newdata.num_config[file] = ns;
    int num_atom_sum = 0;
    for (int ii = 0; ii < ns; ii++) {    // loop over each selected configuration in a file
      int ci = data.num_config_cumsum[file] + selected[file][ii] - 1;
      int natom = data.num_atom[ci];
      newdata.num_atom.push_back(natom);
      num_atom_sum += natom;
    }
    newdata.num_atom_each_file[file] = num_atom_sum;
  }
  int len = newdata.num_atom.size();
  newdata.num_atom_min = podArrayMin(newdata.num_atom.data(), len);
  newdata.num_atom_max = podArrayMax(newdata.num_atom.data(), len);
  newdata.num_atom_cumsum.resize(len + 1);
  podCumsum(newdata.num_atom_cumsum.data(), newdata.num_atom.data(), len + 1);
  newdata.num_atom_sum = newdata.num_atom_cumsum[len];
  podCumsum(newdata.num_config_cumsum.data(), newdata.num_config.data(), nfiles + 1);
  newdata.num_config_sum = newdata.num_atom.size();

  int n = newdata.num_config_sum;
  memory->create(newdata.lattice, 9 * n, "fitpod:newdata_lattice");
  memory->create(newdata.stress, 9 * n, "fitpod:newdata_stress");
  memory->create(newdata.energy, n, "fitpod:newdata_energy");
  // Group weights have same size as energy.
  memory->create(newdata.we, n, "fitpod:we");
  memory->create(newdata.wf, n, "fitpod:wf");

  n = newdata.num_atom_sum;
  memory->create(newdata.position, 3 * n, "fitpod:newdata_position");
  memory->create(newdata.force, 3 * n, "fitpod:newdata_force");
  memory->create(newdata.atomtype, n, "fitpod:newdata_atomtype");

  int cn = 0;
  int dim = 3;
  for (int file = 0; file < nfiles; file++) {
    int ns = (int) selected[file].size();    // number of selected configurations
    for (int ii = 0; ii < ns; ii++) {        // loop over each selected configuration in a file
      int ci = data.num_config_cumsum[file] + selected[file][ii] - 1;
      int natom = data.num_atom[ci];
      int natom_cumsum = data.num_atom_cumsum[ci];

      int natomnew = newdata.num_atom[cn];
      int natomnew_cumsum = newdata.num_atom_cumsum[cn];

      if (natom != natomnew)
        error->all(
            FLERR,
            "number of atoms in the new data set must be the same as that in the old data set.");

      int *atomtype = &data.atomtype[natom_cumsum];
      double *position = &data.position[dim * natom_cumsum];
      double *force = &data.force[dim * natom_cumsum];

      newdata.energy[cn] = data.energy[ci];
      newdata.we[cn] = data.we[ci];
      newdata.wf[cn] = data.wf[ci];
      for (int j = 0; j < 9; j++) {
        newdata.stress[j + 9 * cn] = data.stress[j + 9 * ci];
        newdata.lattice[j + 9 * cn] = data.lattice[j + 9 * ci];
      }

      for (int na = 0; na < natom; na++) {
        newdata.atomtype[na + natomnew_cumsum] = atomtype[na];
        for (int j = 0; j < dim; j++) {
          newdata.position[j + 3 * na + dim * natomnew_cumsum] = position[j + 3 * na];
          newdata.force[j + 3 * na + dim * natomnew_cumsum] = force[j + 3 * na];
        }
      }
      cn += 1;
    }
  }

  data.copydatainfo(newdata);
  size_t maxname = 9;
  for (const auto &fname : data.data_files) maxname = MAX(maxname, fname.size());
  maxname -= data.data_path.size() + 1;

  if (comm->me == 0)
    utils::logmesg(lmp,
                   "{:-<{}}\n {:^{}} | # configs (selected) | # atoms (selected) "
                   "| # configs (original) | # atoms (original)\n{:-<{}}\n",
                   "", maxname + 90, "data_file", maxname, "", maxname + 90);
  for (int i = 0; i < (int) newdata.data_files.size(); i++) {
    std::string filename =
        newdata.data_files[i].substr(newdata.data_path.size() + 1, newdata.data_files[i].size());
    newdata.filenames.emplace_back(filename.c_str());
    if (comm->me == 0)
      utils::logmesg(
          lmp, " {:<{}} |       {:>8}       |      {:>8}      |       {:>8}       |     {:>8}\n",
          newdata.filenames[i], maxname, newdata.num_config[i], newdata.num_atom_each_file[i],
          data.num_config[i], data.num_atom_each_file[i]);
  }
  if (comm->me == 0) {
    utils::logmesg(lmp, "{:-<{}}\nnumber of files: {}\n", "", maxname + 90,
                   newdata.data_files.size());
    utils::logmesg(lmp,
                   "number of configurations in all files (selected and original): {} and {}\n",
                   newdata.num_config_sum, data.num_config_sum);
    utils::logmesg(lmp, "number of atoms in all files (selected and original: {} and {}\n",
                   newdata.num_atom_sum, data.num_atom_sum);
  }
}

void FitPOD::read_data_files(const std::string &data_file, const std::vector<std::string> &species)
{
  datastruct data;

  // read data input file to datastruct

  data.precision =
      read_data_file(data.fitting_weights, data.file_format, data.file_extension, envdata.data_path,
                     testdata.data_path, data.data_path, data.filenametag, data_file,
                     data.group_weight_type, data.we_map, data.wf_map);

  data.training_analysis = (int) data.fitting_weights[3];
  data.test_analysis = (int) data.fitting_weights[4];
  data.training_calculation = (int) data.fitting_weights[5];
  data.test_calculation = (int) data.fitting_weights[6];
  data.fraction = data.fitting_weights[7];
  data.randomize = (int) data.fitting_weights[9];

  data.copydatainfo(traindata);

  if (data.fraction >= 1.0) {
    if (comm->me == 0)
      utils::logmesg(lmp, "**************** Begin of Training Data Set ****************\n");
    if (traindata.data_path.size() > 1)
      get_data(traindata, species);
    else
      error->all(FLERR, "data set is not found");
    if (comm->me == 0)
      utils::logmesg(lmp, "**************** End of Training Data Set ****************\n");
  } else {
    if (comm->me == 0)
      utils::logmesg(lmp, "**************** Begin of Training Data Set ****************\n");
    if (data.data_path.size() > 1)
      get_data(data, species);
    else
      error->all(FLERR, "data set is not found");
    if (comm->me == 0)
      utils::logmesg(lmp, "**************** End of Training Data Set ****************\n");

    if (comm->me == 0)
      utils::logmesg(lmp, "**************** Begin of Select Training Data Set ****************\n");
    select_data(traindata, data);
    if (comm->me == 0)
      utils::logmesg(lmp, "**************** End of Select Training Data Set ****************\n");

    deallocate_memory_datastruct(data);
  }

  testdata.fraction = traindata.fitting_weights[8];
  testdata.test_analysis = traindata.test_analysis;
  testdata.filenametag = traindata.filenametag;

  if (((int) envdata.data_path.size() > 1)) {
    envdata.filenametag = traindata.filenametag;
    envdata.file_format = traindata.file_format;
    envdata.file_extension = traindata.file_extension;
    int tmp = compute_descriptors;
    compute_descriptors = 1;
    if (comm->me == 0)
      utils::logmesg(lmp,
                     "**************** Begin of Environment Configuration Set ****************\n");
    get_data(envdata, species);
    if (comm->me == 0)
      utils::logmesg(lmp,
                     "**************** End of Environment Configuration Set ****************\n");
    compute_descriptors = tmp;
  }

  if ((testdata.data_path == traindata.data_path) && (testdata.fraction == 1.0) &&
      (traindata.fraction == 1.0)) {
    testdata.data_path = traindata.data_path;
  } else if (((int) testdata.data_path.size() > 1) && (testdata.fraction > 0) &&
             (testdata.test_analysis)) {
    testdata.training = 0;
    testdata.file_format = traindata.file_format;
    testdata.file_extension = traindata.file_extension;
    testdata.training_analysis = traindata.training_analysis;
    testdata.training_calculation = traindata.training_calculation;
    testdata.test_calculation = traindata.test_calculation;
    testdata.randomize = (int) traindata.fitting_weights[10];

    if (testdata.fraction >= 1.0) {
      if (comm->me == 0)
        utils::logmesg(lmp, "**************** Begin of Test Data Set ****************\n");
      get_data(testdata, species);
      if (comm->me == 0)
        utils::logmesg(lmp, "**************** End of Test Data Set ****************\n");
    } else {
      datastruct datatm;
      testdata.copydatainfo(datatm);

      if (comm->me == 0)
        utils::logmesg(lmp, "**************** Begin of Test Data Set ****************\n");
      get_data(datatm, species);
      if (comm->me == 0)
        utils::logmesg(lmp, "**************** End of Test Data Set ****************\n");

      if (comm->me == 0)
        utils::logmesg(lmp, "**************** Begin of Select Test Data Set ****************\n");
      select_data(testdata, datatm);
      if (comm->me == 0)
        utils::logmesg(lmp, "**************** End of Select Test Data Set ****************\n");

      deallocate_memory_datastruct(datatm);
    }
  } else {
    testdata.data_path = traindata.data_path;
  }
}

int FitPOD::latticecoords(double *y, int *alist, double *x, double *a1, double *a2, double *a3,
                          double rcutmax, int *pbc, int nx)
{
  int m = 0, n = 0, p = 0;
  if (pbc[0] == 1) m = (int) ceil(rcutmax / a1[0]);
  if (pbc[1] == 1) n = (int) ceil(rcutmax / a2[1]);
  if (pbc[2] == 1) p = (int) ceil(rcutmax / a3[2]);

  // index for the center lattice

  int ind = m + (2 * m + 1) * n + (2 * m + 1) * (2 * n + 1) * p;

  // number of lattices

  int nl = (2 * m + 1) * (2 * n + 1) * (2 * p + 1);

  for (int j = 0; j < 3 * nx; j++) y[j] = x[j];
  int q = nx;

  for (int i = 0; i < (2 * p + 1); i++)
    for (int j = 0; j < (2 * n + 1); j++)
      for (int k = 0; k < (2 * m + 1); k++) {
        int ii = k + (2 * m + 1) * j + (2 * m + 1) * (2 * n + 1) * i;
        if (ii != ind) {
          double x0 = a1[0] * (k - m) + a2[0] * (j - n) + a3[0] * (i - p);
          double x1 = a1[1] * (k - m) + a2[1] * (j - n) + a3[1] * (i - p);
          double x2 = a1[2] * (k - m) + a2[2] * (j - n) + a3[2] * (i - p);
          for (int jj = 0; jj < nx; jj++) {
            y[0 + 3 * q] = x0 + x[0 + 3 * jj];
            y[1 + 3 * q] = x1 + x[1 + 3 * jj];
            y[2 + 3 * q] = x2 + x[2 + 3 * jj];
            q++;
          }
        }
      }

  for (int i = 0; i < nl; i++)
    for (int j = 0; j < nx; j++) alist[j + nx * i] = j;

  return nl;
}

int FitPOD::podneighborlist(int *neighlist, int *numneigh, double *r, double **rcutsq, 
                            int *atomtype, int *alist, int nx, int N, int dim, int nelements)
{
  int k = 0;
  for (int i = 0; i < nx; i++) {
    double *ri = &r[i * dim];
    int inc = 0;
    int itype = atomtype[i] - 1;
    for (int j = 0; j < N; j++) {
      double *rj = &r[j * dim];
      double rijsq = (ri[0] - rj[0]) * (ri[0] - rj[0]) + 
                     (ri[1] - rj[1]) * (ri[1] - rj[1]) + 
                     (ri[2] - rj[2]) * (ri[2] - rj[2]);
      
      int jtype = atomtype[alist[j]] - 1;
      
      if ((rijsq > SMALL) && (rijsq < rcutsq[itype][jtype])) {
        inc += 1;
        neighlist[k] = j;
        k += 1;
      }
    }
    numneigh[i] = inc;
  }
  return k;
}

int FitPOD::podfullneighborlist(double *y, int *alist, int *neighlist, int *numneigh,
                                int *numneighsum, double *x, double *a1, double *a2, 
                                double *a3, double **rcutsq, int *pbc, int *atomtype, 
                                int nx, int nelements)
{
  int dim = 3, nl = 0, nn = 0;
  
  // maximum cutoff for the neighbor list construction
  // number of lattices
  nl = latticecoords(y, alist, x, a1, a2, a3, fastpodptr->rcutmax, pbc, nx);
  int N = nx * nl;

  // total number of neighbors
  nn = podneighborlist(neighlist, numneigh, y, rcutsq, atomtype, alist, nx, N, dim, nelements);
  podCumsum(numneighsum, numneigh, nx + 1);
  
  return nn;
}

void FitPOD::estimate_memory_neighborstruct(const datastruct &data, int *pbc,
                                            double rcutmax, int nelements)
{
  int dim = 3;
  int natom_max = data.num_atom_max;
  int m = 0, n = 0, p = 0, nl = 0, ny = 0, na = 0, np = 0;

  for (int ci = 0; ci < (int) data.num_atom.size(); ci++) {
    int natom = data.num_atom[ci];
    double *lattice = &data.lattice[9 * ci];
    double *a1 = &lattice[0];
    double *a2 = &lattice[3];
    double *a3 = &lattice[6];
    if (pbc[0] == 1) m = (int) ceil(rcutmax / a1[0]);
    if (pbc[1] == 1) n = (int) ceil(rcutmax / a2[1]);
    if (pbc[2] == 1) p = (int) ceil(rcutmax / a3[2]);

    // number of lattices

    nl = (2 * m + 1) * (2 * n + 1) * (2 * p + 1);
    ny = MAX(ny, dim * natom * nl);
    na = MAX(na, natom * nl);
    np = MAX(np, natom * natom * nl);
  }

  nb.natom_max = MAX(nb.natom_max, natom_max);
  nb.sze = nelements * nelements;
  nb.sza = MAX(nb.sza, na);
  nb.szy = MAX(nb.szy, ny);
  nb.szp = MAX(nb.szp, np);
}

void FitPOD::allocate_memory_neighborstruct()
{
  memory->create(nb.y, nb.szy, "fitpod:nb_y");
  memory->create(nb.alist, nb.sza, "fitpod:nb_alist");
  memory->create(nb.pairnum, nb.natom_max, "fitpod:nb_pairnum");
  memory->create(nb.pairnum_cumsum, nb.natom_max + 1, "fitpod:nb_pairnum_cumsum");
  memory->create(nb.pairlist, nb.szp, "fitpod:nb_pairlist");
}

void FitPOD::allocate_memory_descriptorstruct(int nCoeffAll)
{
  memory->create(desc.bd, nb.natom_max * fastpodptr->Mdesc, "fitpod:desc_ld");
  memory->create(desc.pd, nb.natom_max * fastpodptr->nClusters, "fitpod:desc_ld");
  memory->create(desc.gd, nCoeffAll, "fitpod:desc_gd");
  memory->create(desc.A, nCoeffAll * nCoeffAll, "fitpod:desc_A");
  memory->create(desc.b, nCoeffAll, "fitpod:desc_b");
  memory->create(desc.c, nCoeffAll, "fitpod:desc_c");
  memory->create(desc.gdd, desc.szd, "fitpod:desc_gdd");
  podArraySetValue(desc.A, 0.0, nCoeffAll * nCoeffAll);
  podArraySetValue(desc.b, 0.0, nCoeffAll);
  podArraySetValue(desc.c, 0.0, nCoeffAll);

  if (comm->me == 0) {
    utils::logmesg(lmp, "**************** Begin of Memory Allocation ****************\n");
    utils::logmesg(lmp, "maximum number of atoms in periodic domain: {}\n", nb.natom_max);
    utils::logmesg(lmp, "maximum number of atoms in extended domain: {}\n", nb.sza);
    utils::logmesg(lmp, "maximum number of neighbors in extended domain: {}\n", nb.szp);
    utils::logmesg(lmp, "size of double memory: {}\n", desc.szd);
    utils::logmesg(lmp, "size of descriptor matrix: {} x {}\n", nCoeffAll, nCoeffAll);
    utils::logmesg(lmp, "**************** End of Memory Allocation ****************\n");
  }
}

void FitPOD::deallocate_memory_datastruct(datastruct &data)
{
  memory->destroy(data.lattice);
  memory->destroy(data.energy);
  memory->destroy(data.stress);
  memory->destroy(data.position);
  memory->destroy(data.force);
  memory->destroy(data.atomtype);
  memory->destroy(data.we);
  memory->destroy(data.wf);
}

void FitPOD::estimate_memory_fastpod(const datastruct &data)
{
  int dim = 3;
  int *pbc = fastpodptr->pbc;
  double **rcutsq = fastpodptr->rcutsq;

  int Nij = 0, Nijmax = 0;
  for (int ci = 0; ci < (int) data.num_atom.size(); ci++) {
    int natom = data.num_atom[ci];
    int natom_cumsum = data.num_atom_cumsum[ci];
    int *atomtype = &data.atomtype[natom_cumsum];
    double *x = &data.position[dim * natom_cumsum];
    double *lattice = &data.lattice[9 * ci];
    double *a1 = &lattice[0];
    double *a2 = &lattice[3];
    double *a3 = &lattice[6];

    Nij = podfullneighborlist(nb.y, nb.alist, nb.pairlist, nb.pairnum, nb.pairnum_cumsum, 
                              x, a1, a2, a3, rcutsq, pbc, atomtype, natom, 
                              fastpodptr->nelements);

    Nijmax = MAX(Nijmax, Nij);
  }

  desc.szd = MAX(desc.szd, 3 * Nijmax * fastpodptr->nCoeffAll);
}

void FitPOD::local_descriptors_fastpod(const datastruct &data, int ci)
{
  int dim = 3;
  int *pbc = fastpodptr->pbc;
  double **rcutsq = fastpodptr->rcutsq;

  int natom = data.num_atom[ci];
  int natom_cumsum = data.num_atom_cumsum[ci];
  int *atomtype = &data.atomtype[natom_cumsum];
  double *position = &data.position[dim * natom_cumsum];
  double *lattice = &data.lattice[9 * ci];
  double *a1 = &lattice[0];
  double *a2 = &lattice[3];
  double *a3 = &lattice[6];

  // neighbor list
  podfullneighborlist(nb.y, nb.alist, nb.pairlist, nb.pairnum, nb.pairnum_cumsum, position,
                      a1, a2, a3, rcutsq, pbc, atomtype, natom, fastpodptr->nelements);

  if (desc.nClusters > 1) {
    fastpodptr->descriptors(desc.gd, desc.gdd, desc.bd, desc.pd, nb.y, atomtype, nb.alist,
                            nb.pairlist, nb.pairnum_cumsum, natom);
  } else {
    fastpodptr->descriptors(desc.gd, desc.gdd, desc.bd, nb.y, atomtype, nb.alist, nb.pairlist,
                            nb.pairnum_cumsum, natom);
  }
}

void FitPOD::base_descriptors_fastpod(const datastruct &data, int ci)
{
  int dim = 3;
  int *pbc = fastpodptr->pbc;
  double **rcutsq = fastpodptr->rcutsq;

  int natom = data.num_atom[ci];
  int natom_cumsum = data.num_atom_cumsum[ci];
  int *atomtype = &data.atomtype[natom_cumsum];
  double *position = &data.position[dim * natom_cumsum];
  double *lattice = &data.lattice[9 * ci];
  double *a1 = &lattice[0];
  double *a2 = &lattice[3];
  double *a3 = &lattice[6];

  // neighbor list
  podfullneighborlist(nb.y, nb.alist, nb.pairlist, nb.pairnum, nb.pairnum_cumsum, position,
                      a1, a2, a3, rcutsq, pbc, atomtype, natom, fastpodptr->nelements);

  fastpodptr->base_descriptors(desc.bd, nb.y, atomtype, nb.alist, nb.pairlist, nb.pairnum_cumsum,
                               natom);
}

void FitPOD::descriptors_calculation(const datastruct &data)
{
  if (comm->me == 0)
    utils::logmesg(lmp, "**************** Begin Calculating Descriptors ****************\n");

  // loop over each configuration in the training data set

  double sz[2];
  for (int ci = 0; ci < (int) data.num_atom.size(); ci++) {

    if ((ci % 100) == 0) {
      if (comm->me == 0) utils::logmesg(lmp, "Configuration: # {}\n", ci + 1);
    }

    if ((ci % comm->nprocs) == comm->me) {

      // compute local POD descriptors
      local_descriptors_fastpod(data, ci);

      std::string filename0 =
          data.data_path + "/basedescriptors_config" + std::to_string(ci + 1) + ".bin";
      SafeFilePtr fp0 = fopen(filename0.c_str(), "wb");
      sz[0] = (double) data.num_atom[ci];
      sz[1] = (double) fastpodptr->Mdesc;
      fwrite(reinterpret_cast<char *>(sz), sizeof(double) * 2, 1, fp0);
      fwrite(reinterpret_cast<char *>(desc.bd),
             sizeof(double) * (data.num_atom[ci] * fastpodptr->Mdesc), 1, fp0);

      if (desc.nClusters > 1) {
        std::string filename1 =
            data.data_path + "/environmentdescriptors_config" + std::to_string(ci + 1) + ".bin";
        SafeFilePtr fp1 = fopen(filename1.c_str(), "wb");
        sz[0] = (double) data.num_atom[ci];
        sz[1] = (double) fastpodptr->nClusters;
        fwrite(reinterpret_cast<char *>(sz), sizeof(double) * 2, 1, fp1);
        fwrite(reinterpret_cast<char *>(desc.pd),
               sizeof(double) * (data.num_atom[ci] * fastpodptr->nClusters), 1, fp1);
      }

      std::string filename =
          data.data_path + "/globaldescriptors_config" + std::to_string(ci + 1) + ".bin";
      SafeFilePtr fp = fopen(filename.c_str(), "wb");

      sz[0] = (double) data.num_atom[ci];
      sz[1] = (double) desc.nCoeffAll;
      fwrite(reinterpret_cast<char *>(sz), sizeof(double) * 2, 1, fp);
      fwrite(reinterpret_cast<char *>(desc.gd), sizeof(double) * (desc.nCoeffAll), 1, fp);
      if (compute_descriptors == 2) {
        fwrite(reinterpret_cast<char *>(desc.gdd),
               sizeof(double) * (3 * data.num_atom[ci] * desc.nCoeffAll), 1, fp);
      }
    }
  }

  if (comm->me == 0)
    utils::logmesg(lmp, "**************** End Calculating Descriptors ****************\n");
}

void FitPOD::environment_pipeline(const datastruct &data, int mode)
{
  const bool do_proj   = (mode == 0 || mode == 1);
  const bool do_kmeans = (mode == 0 || mode == 2);
  const bool localeapod = fastpodptr->localeapod;

  int nComponents = fastpodptr->nComponents;
  int Mdesc       = fastpodptr->Mdesc;
  int nClusters   = fastpodptr->nClusters;
  int nelements   = fastpodptr->nelements;
  int nMaxActiveClusters = fastpodptr->nMaxActiveClusters;
  double nActiveClusters = fastpodptr->nActiveClusters;

  if (do_proj && fastpodptr->Proj == nullptr)
    memory->create(fastpodptr->Proj, Mdesc * nComponents * nelements, "fitpod:P");

  if (do_kmeans && fastpodptr->Centroids == nullptr)
    memory->create(fastpodptr->Centroids, nClusters * nComponents * nelements, "fitpod:centroids");

  if (do_kmeans && localeapod) {
    if (fastpodptr->invLeftClusterRcut2 == nullptr)
      memory->create(fastpodptr->invLeftClusterRcut2, nClusters * nComponents * nelements, "fitpod:invLeftClusterRcut2");
    if (fastpodptr->invRightClusterRcut2 == nullptr)
      memory->create(fastpodptr->invRightClusterRcut2, nClusters * nComponents * nelements, "fitpod:invRightClusterRcut2");
    if (fastpodptr->leftClusterEdges == nullptr)
      memory->create(fastpodptr->leftClusterEdges, nClusters * nComponents * nelements, "fitpod:leftClusterEdges");
    if (fastpodptr->rightClusterEdges == nullptr)
      memory->create(fastpodptr->rightClusterEdges, nClusters * nComponents * nelements, "fitpod:rightClusterEdges");
    // temp memory for extra training arrays
    if (fastpodptr->ClusterFcut == nullptr)
      memory->create(fastpodptr->ClusterFcut, nMaxActiveClusters * nComponents * nelements, "fitpod:ClusterFcut");
    if (fastpodptr->ClusterDFcut == nullptr)
      memory->create(fastpodptr->ClusterDFcut, nMaxActiveClusters * nComponents * nelements, "fitpod:ClusterDFcut");
  }

  int nAtoms = 0, nTotalAtoms = 0;
  for (int ci = 0; ci < (int)data.num_atom.size(); ci++) {
    if ((ci % comm->nprocs) == comm->me) nAtoms += data.num_atom[ci];
    nTotalAtoms += data.num_atom[ci];
  }

  double *basedescmatrix = nullptr;
  double *pca = nullptr;
  double *A = nullptr, *work = nullptr, *b = nullptr, *Lambda = nullptr;
  int *clusterSizes = nullptr, *assignments = nullptr;
  int *nElemAtoms = nullptr, *nElemAtomsCumSum = nullptr, *nElemAtomsCount = nullptr;

  // histogram of per-atom pca descriptors over the centroid span
  // layout per (elem, comp): nbins in-span bins plus one ghost bin on each side (nbins+2)
  const bool do_hist = (do_kmeans && save_pca_histogram);
  const int nbins = pca_histogram_num_bins;
  int *pcaHist = nullptr;      // nelements*nComponents*(nbins+2)
  double *pcaSpan = nullptr;   // nelements*nComponents*2

  // per-cluster training occupancy (atom counts)
  // inference-time active-learning metric can weight clusters by training density
  int *clusterOcc = nullptr;   // nClusters*nelements

  memory->create(basedescmatrix, nAtoms * Mdesc, "fitpod:basedescmatrix");
  if (do_kmeans) {
    memory->create(pca, nAtoms * nComponents, "fitpod:pca");
    memory->create(clusterSizes, nClusters * nelements, "fitpod:clusterSizes");
    memory->create(assignments, nAtoms, "fitpod:assignments");
    memory->create(clusterOcc, nClusters * nelements, "fitpod:clusterOcc");
    for (int i = 0; i < nClusters * nelements; i++) clusterOcc[i] = 0;
  }
  if (do_hist) {
    memory->create(pcaHist, nelements * nComponents * (nbins + 2), "fitpod:pcaHist");
    memory->create(pcaSpan, nelements * nComponents * 2, "fitpod:pcaSpan");
    for (int i = 0; i < nelements * nComponents * (nbins + 2); i++) pcaHist[i] = 0;
    for (int i = 0; i < nelements * nComponents * 2; i++) pcaSpan[i] = 0.0;
  }
  if (do_proj) {
    memory->create(A, Mdesc * Mdesc, "fitpod:A");
    memory->create(work, Mdesc * Mdesc, "fitpod:work");
    memory->create(b, Mdesc, "fitpod:b");
    memory->create(Lambda, Mdesc * nelements, "fitpod:Lambda");
  }

  memory->create(nElemAtoms, nelements, "fitpod:nElemAtoms");
  memory->create(nElemAtomsCumSum, 1 + nelements, "fitpod:nElemAtomsCumSum");
  memory->create(nElemAtomsCount, nelements, "fitpod:nElemAtomsCount");

  char chn = 'N', cht = 'T', chv = 'V', chu = 'U';
  double alpha = 1.0, beta = 0.0;

  for (int elem = 0; elem < nelements; elem++) nElemAtoms[elem] = 0;

  for (int ci = 0; ci < (int)data.num_atom.size(); ci++) {
    if ((ci % comm->nprocs) == comm->me) {
      int natom = data.num_atom[ci];
      int natom_cumsum = data.num_atom_cumsum[ci];
      int *atomtype = &data.atomtype[natom_cumsum];
      for (int n = 0; n < natom; n++) nElemAtoms[atomtype[n] - 1] += 1;
    }
  }

  nElemAtomsCumSum[0] = 0;
  for (int elem = 0; elem < nelements; elem++) {
    nElemAtomsCumSum[elem + 1] = nElemAtomsCumSum[elem] + nElemAtoms[elem];
    nElemAtomsCount[elem] = 0;
  }

  for (int ci = 0; ci < (int)data.num_atom.size(); ci++) {
    if ((ci % 100) == 0 && comm->me == 0) utils::logmesg(lmp, "Configuration: # {}\n", ci + 1);

    if ((ci % comm->nprocs) == comm->me) {
      base_descriptors_fastpod(data, ci);

      int natom = data.num_atom[ci];
      int natom_cumsum = data.num_atom_cumsum[ci];
      int *atomtype = &data.atomtype[natom_cumsum];

      for (int n = 0; n < natom; n++) {
        int elem = atomtype[n] - 1;
        nElemAtomsCount[elem] += 1;
        int k = nElemAtomsCumSum[elem] + nElemAtomsCount[elem] - 1;
        for (int m = 0; m < Mdesc; m++) basedescmatrix[m + Mdesc * k] = desc.bd[n + natom * m];
      }
    }
  }

  for (int elem = 0; elem < nelements; elem++) {
    nAtoms = nElemAtoms[elem];
    nTotalAtoms = nAtoms;
    MPI_Allreduce(MPI_IN_PLACE, &nTotalAtoms, 1, MPI_INT, MPI_SUM, world);

    double *descmatrix = &basedescmatrix[Mdesc * nElemAtomsCumSum[elem]];
    double *Proj = &fastpodptr->Proj[nComponents * Mdesc * elem];

    if (do_proj) {
      DGEMM(&chn, &cht, &Mdesc, &Mdesc, &nAtoms, &alpha, descmatrix, &Mdesc, descmatrix, &Mdesc,
            &beta, A, &Mdesc);
      MPI_Allreduce(MPI_IN_PLACE, A, Mdesc * Mdesc, MPI_DOUBLE, MPI_SUM, world);

      int lwork = Mdesc * Mdesc, info = 1;
      DSYEV(&chv, &chu, &Mdesc, A, &Mdesc, b, work, &lwork, &info);

      for (int i = 0; i < Mdesc; i++) Lambda[Mdesc - i - 1] = b[i];

      for (int j = 0; j < nComponents; j++)
        for (int i = 0; i < Mdesc; i++)
          Proj[j + nComponents * i] =
              A[i + Mdesc * (Mdesc - j - 1)] * sqrt(fabs(b[Mdesc - j - 1] / Lambda[0]));
    }

    if (do_kmeans) {
      if (Proj == nullptr) error->all(FLERR, "Projection matrix is not initialized for k-means stage");

      double *centroids = &fastpodptr->Centroids[nComponents * nClusters * elem];

      DGEMM(&chn, &chn, &nComponents, &nAtoms, &Mdesc, &alpha, Proj, &nComponents, descmatrix, &Mdesc,
            &beta, pca, &nComponents);

      for (int i = 0; i < nClusters * nComponents; i++) centroids[i] = 0.0;
      for (int i = 0; i < nAtoms; i++) {
        int m = (i * nClusters) / nAtoms;
        for (int j = 0; j < nComponents; j++) centroids[j + nComponents * m] += pca[j + nComponents * i];
      }

      MPI_Allreduce(MPI_IN_PLACE, centroids, nClusters * nComponents, MPI_DOUBLE, MPI_SUM, world);
      double fac = ((double)nClusters) / ((double)nTotalAtoms);
      for (int i = 0; i < nClusters * nComponents; i++) centroids[i] *= fac;

      int max_iter = 100;
      KmeansClustering(pca, centroids, assignments, clusterSizes, nAtoms, nClusters, nComponents, max_iter);

      if (nComponents == 1) std::sort(centroids, centroids + nClusters);

      int *occ = &clusterOcc[nClusters * elem];
      for (int i = 0; i < nAtoms; i++) {
        double mind = squareDistance(&pca[nComponents * i], &centroids[0], nComponents);
        int kbest = 0;
        for (int j = 1; j < nClusters; j++) {
          double d = squareDistance(&pca[nComponents * i], &centroids[nComponents * j], nComponents);
          if (d < mind) { mind = d; kbest = j; }
        }
        occ[kbest] += 1;
      }

      // accumulate per-atom pca descriptors into the histogram for this element
      // for r = r_in, maps to pca = 0
      // ghost bins on poles
      if (do_hist) {
        for (int c = 0; c < nComponents; c++) {
          double cmin = centroids[c];
          double cmax = centroids[c];
          for (int j = 1; j < nClusters; j++) {
            double v = centroids[c + j * nComponents];
            if (v < cmin) cmin = v;
            if (v > cmax) cmax = v;
          }

          // pick the family from the dominant sign of the centroids and anchor at 0
          bool posfamily = (cmax >= -cmin);
          double lo = posfamily ? 0.0 : cmin;
          double hi = posfamily ? cmax : 0.0;
          pcaSpan[0 + 2 * (c + nComponents * elem)] = lo;
          pcaSpan[1 + 2 * (c + nComponents * elem)] = hi;

          double span = hi - lo;
          int *hist = &pcaHist[(elem * nComponents + c) * (nbins + 2)];
          for (int i = 0; i < nAtoms; i++) {
            double v = pca[c + nComponents * i];
            int bin;
            if (v < lo) bin = 0;                // left ghost bin
            else if (v >= hi) bin = nbins + 1;  // right ghost bin
            else if (span > 0.0) {
              int k = (int) ((v - lo) / span * nbins);
              if (k < 0) k = 0;
              if (k >= nbins) k = nbins - 1;
              bin = k + 1;    // in-span bins are 1..nbins
            } else {
              bin = 1;    // zero span: all in-span atoms into first bin
            }
            hist[bin] += 1;
          }
        }
      }
    }
  }

  if (do_kmeans && localeapod) {
    fastpodptr->calculateClusterEdges(nClusters, nActiveClusters, nComponents, nelements);
  }

  if (do_kmeans) {
    MPI_Allreduce(MPI_IN_PLACE, clusterOcc, nClusters * nelements, MPI_INT, MPI_SUM, world);

    if (comm->me == 0) {
      std::string filename = data.filenametag + "_cluster_occupancy.pod";
      SafeFilePtr fp = fopen(filename.c_str(), "w");
      utils::print(fp, "cluster_occupancy: {} {}\n", nClusters, nelements);
      for (int i = 0; i < nClusters * nelements; i++) utils::print(fp, "{}\n", clusterOcc[i]);
    }
  }
  
  if (do_hist) {
    MPI_Allreduce(MPI_IN_PLACE, pcaHist, nelements * nComponents * (nbins + 2), MPI_INT,
                  MPI_SUM, world);

    if (comm->me == 0) {
      std::string filename = data.filenametag + "_pca_histogram.txt";
      SafeFilePtr fp = fopen(filename.c_str(), "w");
      utils::print(fp, "# pca_histogram nelements nComponents nbins "
                       "(in-span range is zero-anchored; 2 ghost bins per row: left, right)\n");
      utils::print(fp, "{} {} {}\n", nelements, nComponents, nbins);
      utils::print(fp, "# elem comp lo hi ghostL bin_1 ... bin_nbins ghostR\n");
      for (int elem = 0; elem < nelements; elem++) {
        for (int c = 0; c < nComponents; c++) {
          double lo = pcaSpan[0 + 2 * (c + nComponents * elem)];
          double hi = pcaSpan[1 + 2 * (c + nComponents * elem)];
          int *hist = &pcaHist[(elem * nComponents + c) * (nbins + 2)];
          utils::print(fp, "{} {} {:.15g} {:.15g}", elem, c, lo, hi);
          for (int bin = 0; bin < nbins + 2; bin++) utils::print(fp, " {}", hist[bin]);
          utils::print(fp, "\n");
        }
      }
    }
  }

  memory->destroy(basedescmatrix);
  if (do_kmeans) {
    memory->destroy(pca);
    memory->destroy(clusterSizes);
    memory->destroy(assignments);
    memory->destroy(clusterOcc);
  }
  if (do_hist) {
    memory->destroy(pcaHist);
    memory->destroy(pcaSpan);
  }
  if (do_proj) {
    memory->destroy(A);
    memory->destroy(work);
    memory->destroy(b);
    memory->destroy(Lambda);
  }
  memory->destroy(nElemAtoms);
  memory->destroy(nElemAtomsCumSum);
  memory->destroy(nElemAtomsCount);
}

void FitPOD::environment_cluster_calculation(const datastruct &data)
{
  if (comm->me == 0)
    utils::logmesg(lmp, "**************** Begin Calculating Environment Descriptor Matrix ****************\n");
  environment_pipeline(data, 0);
  if (comm->me == 0)
    utils::logmesg(lmp, "**************** End Calculating Environment Descriptor Matrix ****************\n");
}

void FitPOD::environment_proj_calculation(const datastruct &data)
{
  if (comm->me == 0)
    utils::logmesg(lmp, "**************** Begin Calculating Environment Projection Matrix ****************\n");
  environment_pipeline(data, 1);
  if (comm->me == 0)
    utils::logmesg(lmp, "**************** End Calculating Environment PCA Matrix ****************\n");
}

void FitPOD::training_cluster_calculation(const datastruct &data)
{
  if (comm->me == 0)
    utils::logmesg(lmp, "**************** Begin Calculating Training K-means Clustering ****************\n");
  environment_pipeline(data, 2);
  if (comm->me == 0)
    utils::logmesg(lmp, "**************** End Calculating Training K-means Clustering ****************\n");
}

void FitPOD::least_squares_matrix(const datastruct &data, int ci,
                                  double eweight, double *fweight)
{
  int dim = 3;
  int natom = data.num_atom[ci];
  int natom_cumsum = data.num_atom_cumsum[ci];
  int nCoeffAll = desc.nCoeffAll;
  int nforce = dim * natom;

  double normconst = (data.normalizeenergy == 1) ? 1.0/natom : 1.0;
  double energy = data.energy[ci];
  double *force = &data.force[dim * natom_cumsum];

  // --- energy block:  A += we2 * gd*gd ,  b += we2*energy*gd
  double we2 = eweight * (normconst * normconst);
  podKron(desc.A, desc.gd, desc.gd, we2, nCoeffAll, nCoeffAll);
  double wee = we2 * energy;
  for (int i = 0; i < nCoeffAll; i++) desc.b[i] += wee * desc.gd[i];

  // --- force block with per-atom weights:  A += gdd^T W gdd , b += gdd^T W f
  std::vector<double> fw(nforce);
  for (int a = 0; a < natom; a++) {
    double s = sqrt(fweight[a]);
    for (int d = 0; d < 3; d++) {
      int row = 3*a + d;
      fw[row] = s * force[row];
      double *col = &desc.gdd[row];                 // stride nforce over columns
      for (int j = 0; j < nCoeffAll; j++) col[nforce*j] *= s;
    }
  }

  char cht = 'T', chn = 'N';
  double one = 1.0;
  int inc1 = 1;
  DGEMM(&cht, &chn, &nCoeffAll, &nCoeffAll, &nforce, &one,
        desc.gdd, &nforce, desc.gdd, &nforce, &one, desc.A, &nCoeffAll);
  DGEMV(&cht, &nforce, &nCoeffAll, &one, desc.gdd, &nforce,
        fw.data(), &inc1, &one, desc.b, &inc1);
}

void FitPOD::least_squares_fit(const datastruct &data)
{
  if (comm->me == 0)
    utils::logmesg(lmp, "**************** Begin of Least-Squares Fitting ****************\n");

  int nconfig = (int) data.num_atom.size();
  std::vector<double> ew(nconfig);
  std::vector<double> fwt(data.num_atom_sum);
  compute_loss_weights(data, ew.data(), fwt.data());

  int  adapt = (int) data.fitting_weights[22];
  double eps = data.fitting_weights[23];
  if (eps <= 0.0) eps = 1e-2;

  if (adapt) {
    const int Nrho = 200;
    std::vector<double> rho;
    build_pair_distance_density(data, rho, Nrho);
    radial_smoothness_matrices(rho.data(), Nrho, eps);  // density-adaptive
  } else {
    radial_smoothness_matrices(nullptr, 0, 0.0);        // uniform gains
  }

  for (int ci = 0; ci < nconfig; ci++) {
    if ((ci % 100) == 0 && comm->me == 0)
      utils::logmesg(lmp, "Configuration: # {}\n", ci + 1);

    if ((ci % comm->nprocs) == comm->me) {
      local_descriptors_fastpod(data, ci);

      if (save_descriptors > 0) {
        std::string filename =
            data.data_path + "/descriptors_config" + std::to_string(ci + 1) + ".bin";
        SafeFilePtr fp = fopen(filename.c_str(), "wb");
        fwrite(reinterpret_cast<char *>(desc.gd), sizeof(double) * (desc.nCoeffAll), 1, fp);
        if (save_descriptors == 2) {
          fwrite(reinterpret_cast<char *>(desc.gdd),
                 sizeof(double) * (3 * data.num_atom[ci] * desc.nCoeffAll), 1, fp);
        }
      }

      least_squares_matrix(data, ci, ew[ci], &fwt[data.num_atom_cumsum[ci]]);
    }
  }

  int nCoeffAll = desc.nCoeffAll;
  MPI_Allreduce(MPI_IN_PLACE, desc.b, nCoeffAll, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, desc.A, nCoeffAll*nCoeffAll, MPI_DOUBLE, MPI_SUM, world);

  if (comm->me == 0) {
    // symmetrize A
    for (int i = 0; i < nCoeffAll; i++)
      for (int j = i; j < nCoeffAll; j++) {
        double a = 0.5*(desc.A[i+nCoeffAll*j] + desc.A[j+nCoeffAll*i]);
        desc.A[i+nCoeffAll*j] = a;
        desc.A[j+nCoeffAll*i] = a;
      }

    // --- scale regularization to the matrix magnitude ---
    // (scheme 1/2 normalize weights to sum=1, so A is tiny vs the legacy path;
    //  absolute reg/l2/floor must be made relative to mean(diag A))
    double tr = 0.0;
    for (int i = 0; i < nCoeffAll; i++) tr += desc.A[i+nCoeffAll*i];
    double scaleA = (tr > 0.0) ? tr / nCoeffAll : 1.0;

    // radial Sobolev smoothness (PSD block)
    add_radial_smoothness(desc.A, nCoeffAll,
                          data.fitting_weights[18]*scaleA,   // w0
                          data.fitting_weights[19]*scaleA,   // w1
                          data.fitting_weights[20]*scaleA);  // w2

    double reg   = data.fitting_weights[11];
    double l2    = data.fitting_weights[21];
    double l2eff = l2  * scaleA;     // additive L2 in units of mean diagonal
    double floor = reg * scaleA;     // diagonal floor in same units

    for (int i = 0; i < nCoeffAll; i++) {
      desc.c[i] = desc.b[i];
      desc.A[i+nCoeffAll*i] = desc.A[i+nCoeffAll*i]*(1.0 + reg) + l2eff;
      if (desc.A[i+nCoeffAll*i] < floor) desc.A[i+nCoeffAll*i] = floor;
    }

    int nrhs = 1, info;
    char chu = 'U';
    DPOSV(&chu, &nCoeffAll, &nrhs, desc.A, &nCoeffAll, desc.c, &nCoeffAll, &info);
    if (info != 0) error->all(FLERR, "DPOSV failed (info={}): A not SPD; "
                                     "reduce regularization/smoothness weights", info);
  }

  MPI_Bcast(desc.c, nCoeffAll, MPI_DOUBLE, 0, world);
  fastpodptr->mknewcoeff(desc.c, nCoeffAll);
}

void FitPOD::compute_loss_weights(const datastruct &data, double *ew, double *fw)
{
  int    nconfig = (int) data.num_atom.size();
  int    escheme = (int) data.fitting_weights[13];
  double kappa   = data.fitting_weights[12];
  double dE      = data.fitting_weights[14];
  double dF      = data.fitting_weights[15];
  double Ftol2   = data.fitting_weights[16];   // normF threshold for zero-force
  double wzero   = data.fitting_weights[17];   // fixed weight for zero-force atoms
  if (dE <= 0.0) dE = 1.0;
  if (dF <= 0.0) dF = 1.0;

  // ---------- legacy path (scheme 0): original ----------
  if (escheme == 0) {
    for (int ci = 0; ci < nconfig; ci++) ew[ci] = data.we[ci]*data.we[ci];
    for (int ci = 0; ci < nconfig; ci++) {
      double w = data.wf[ci]*data.wf[ci];
      int nc = data.num_atom_cumsum[ci];
      for (int a = 0; a < data.num_atom[ci]; a++) fw[nc + a] = w;
    }
    return;
  }

  // ---------- energy weights ----------
  if (escheme == 1) {
    // PACE energy-based emphasizes near-ground-state structures
    double emin = 1e300;
    for (int ci = 0; ci < nconfig; ci++) {
      double epa = data.energy[ci] / data.num_atom[ci];
      if (epa < emin) emin = epa;
    }
    double sumE = 0.0;
    for (int ci = 0; ci < nconfig; ci++) {
      double epa = data.energy[ci] / data.num_atom[ci];
      double d   = epa - emin + dE;
      ew[ci] = data.we[ci] / (d*d);
      sumE  += ew[ci];
    }
    if (sumE <= 0.0) sumE = 1.0;
    for (int ci = 0; ci < nconfig; ci++) ew[ci] /= sumE;
  }
  else {  // escheme == 2 : flat / group energy weighting (no E_min dependence)
    double sumE = 0.0;
    for (int ci = 0; ci < nconfig; ci++) { ew[ci] = data.we[ci]; sumE += ew[ci]; }
    if (sumE <= 0.0) sumE = 1.0;
    for (int ci = 0; ci < nconfig; ci++) ew[ci] /= sumE;
  }

  // ---------- force weights (shared by scheme 1 and 2) ----------
  // 1/(|F|^2+dF)
  double sumF = 0.0;
  for (int ci = 0; ci < nconfig; ci++) {
    int natom = data.num_atom[ci];
    int nc    = data.num_atom_cumsum[ci];
    double *force = &data.force[3*nc];
    double wE = ew[ci];
    for (int a = 0; a < natom; a++) {
      double F2 = force[3*a]*force[3*a] + force[3*a+1]*force[3*a+1]
                + force[3*a+2]*force[3*a+2];
      if (F2 <= Ftol2) {
        fw[nc + a] = data.wf[ci] * wzero;  // normalize to group weights
        //sumF += fw[nc + a];
      } else {
        fw[nc + a] = data.wf[ci] * wE / (F2 + dF);
        sumF += fw[nc + a];
      }
    }
  }
  if (sumF <= 0.0) sumF = 1.0;
  for (int ci = 0; ci < nconfig; ci++) {
    int natom = data.num_atom[ci];
    int nc    = data.num_atom_cumsum[ci];
    double *force = &data.force[3*nc];
    for (int a = 0; a < natom; a++) {
      double F2 = force[3*a]*force[3*a] + force[3*a+1]*force[3*a+1]
                + force[3*a+2]*force[3*a+2];
      if (F2 > Ftol2) fw[nc + a] /= sumF;   // only non-zero-force atoms renormalized
    }
  }

  // ---------- kappa blending: L = (1-kappa) E + kappa F ----------
  for (int ci = 0; ci < nconfig; ci++)        ew[ci] *= (1.0 - kappa);
  for (int i = 0; i < data.num_atom_sum; i++) fw[i]  *= kappa;
}

void FitPOD::add_radial_smoothness(double *A, int nCoeffAll,
                                   double w0, double w1, double w2)
{
  if (w0 == 0.0 && w1 == 0.0 && w2 == 0.0) return;

  int Ne        = fastpodptr->nelements;
  int nrbf2     = fastpodptr->nrbf2;
  int nl1       = fastpodptr->nl1;
  int Mdesc     = fastpodptr->Mdesc;
  int ncpe      = fastpodptr->nCoeffPerElement;
  int nClusters = fastpodptr->nClusters;

  std::vector<double> W(nrbf2*nrbf2);
  radial_regularization_matrix(W.data(), w0, w1, w2);

  for (int ic = 0; ic < Ne; ic++)            // center element
    for (int k = 0; k < nClusters; k++)      // environment cluster block
      for (int e = 0; e < Ne; e++) {         // neighbor element
        int base = ncpe*ic + nl1 + Mdesc*k + nrbf2*e;  // start of this 2-body block
        for (int j = 0; j < nrbf2; j++)
          for (int i = 0; i < nrbf2; i++)
            A[(base+i) + nCoeffAll*(base+j)] += W[i + nrbf2*j];
      }
}

inline void FitPOD::radial_smoothness_matrices(double *rho, int Nrho, double eps)
{
  int ns = fastpodptr->ns;
  double rinmin = fastpodptr->rinmin;
  double rcutmax = fastpodptr->rcutmax;
  double rdiffmax = fastpodptr->rdiffmax;
  double *Phi = fastpodptr->Phi;

  int N = 2000;
  double *xij, *S, *Q, *D1, *D2, *tw, *r2, *gain;
  memory->create(xij, N,    "fitpod:sx");
  memory->create(S,   N*ns, "fitpod:sS");
  memory->create(Q,   N*ns, "fitpod:sQ");
  memory->create(D1,  N*ns, "fitpod:sD1");
  memory->create(D2,  N*ns, "fitpod:sD2");
  memory->create(tw,  N,    "fitpod:stw");
  memory->create(r2,  N,    "fitpod:sr2");
  memory->create(gain,N,    "fitpod:sg");

  double r0 = rinmin + 1e-6;
  double L  = rdiffmax - 1e-6;
  for (int i = 0; i < N; i++) xij[i] = r0 + L*(i*1.0/(N-1));
  double h = xij[1] - xij[0];

  fastpodptr->snapshots(S, xij, rinmin, rdiffmax, N);

  char chn = 'N'; double alpha = 1.0, beta = 0.0;
  DGEMM(&chn, &chn, &N, &ns, &ns, &alpha, S, &N, Phi, &ns, &beta, Q, &N);

  for (int i = 0; i < N; i++) { r2[i] = xij[i]*xij[i]; tw[i] = h; }
  tw[0] = 0.5*h; tw[N-1] = 0.5*h;

  if (rho != nullptr && Nrho > 1) {
    double hr = L / (Nrho - 1);
    double mean = 0.0;
    for (int i = 0; i < N; i++) {
      double x = (xij[i] - r0) / hr;
      int j = (int) x; if (j < 0) j = 0; if (j > Nrho-2) j = Nrho-2;
      double f = x - j;
      double rv = (1.0-f)*rho[j] + f*rho[j+1];  // interpolate density
      gain[i] = 1.0 / (rv + eps);
      mean += gain[i];
    }
    mean /= N;
    if (mean > 0.0) for (int i = 0; i < N; i++) gain[i] /= mean;  // unit scale
  } else {
    for (int i = 0; i < N; i++) gain[i] = 1.0;  // uniform
  }

  // finite-difference derivatives
  for (int m = 0; m < ns; m++) {
    double *v = &Q[N*m], *d1 = &D1[N*m], *d2 = &D2[N*m];
    d1[0]   = (v[1]   - v[0])   / h;
    d1[N-1] = (v[N-1] - v[N-2]) / h;
    for (int i = 1; i < N-1; i++) d1[i] = (v[i+1] - v[i-1]) / (2.0*h);
    for (int i = 1; i < N-1; i++) d2[i] = (v[i+1] - 2.0*v[i] + v[i-1]) / (h*h);
    d2[0] = d2[1]; d2[N-1] = d2[N-2];
  }

  memory->create(radialW0, ns*ns, "radialW0");
  memory->create(radialW1, ns*ns, "radialW1");
  memory->create(radialW2, ns*ns, "radialW2");
  for (int i = 0; i < ns*ns; i++) radialW0[i] = radialW1[i] = radialW2[i] = 0.0;

  for (int a = 0; a < ns; a++) {
    double *va=&Q[N*a], *d1a=&D1[N*a], *d2a=&D2[N*a];
    for (int b = 0; b < ns; b++) {
      double *vb=&Q[N*b], *d1b=&D1[N*b], *d2b=&D2[N*b];
      double s0=0.0, s1=0.0, s2=0.0;
      for (int i = 0; i < N; i++) {
        double w = tw[i]*r2[i]*gain[i];   // density-adaptive weight
        s0 += w*va[i]*vb[i];
        s1 += w*d1a[i]*d1b[i];
        s2 += w*d2a[i]*d2b[i];
      }
      radialW0[a+ns*b]=s0; radialW1[a+ns*b]=s1; radialW2[a+ns*b]=s2;
    }
  }

  double inv_rc2 = 1.0/(rcutmax*rcutmax);
  for (int i = 0; i < ns*ns; i++) {
    radialW0[i]*=inv_rc2; radialW1[i]*=inv_rc2; radialW2[i]*=inv_rc2;
  }

  memory->destroy(xij); memory->destroy(S);  memory->destroy(Q);
  memory->destroy(D1);  memory->destroy(D2); memory->destroy(tw);
  memory->destroy(r2);  memory->destroy(gain);
}

void FitPOD::radial_regularization_matrix(double *Wreg, double w0, double w1, double w2)
{
  int nrbf2 = fastpodptr->nrbf2;
  int ns = fastpodptr->ns;
  for (int j = 0; j < nrbf2; j++)
    for (int i = 0; i < nrbf2; i++)
      Wreg[i + nrbf2*j] = w0*radialW0[i + ns*j]
                        + w1*radialW1[i + ns*j]
                        + w2*radialW2[i + ns*j];
}

void FitPOD::build_pair_distance_density(const datastruct &data,
                                         std::vector<double> &rho, int Nrho)
{
  int dim = 3;
  int *pbc = fastpodptr->pbc;
  double **rcutsq = fastpodptr->rcutsq;
  double rinmin = fastpodptr->rinmin;
  double L = fastpodptr->rdiffmax;
  double hr = L / (Nrho - 1);

  double rinmimsq = rinmin * rinmin + 1e-6;

  rho.assign(Nrho, 0.0);

  int nconfig = (int) data.num_atom.size();
  for (int ci = 0; ci < nconfig; ci++) {

    int natom = data.num_atom[ci];
    int nc = data.num_atom_cumsum[ci];
    int *atomtype = &data.atomtype[nc];

    double *x  = &data.position[dim*nc];
    double *a1 = &data.lattice[9*ci];
    double *a2 = &data.lattice[9*ci + 3];
    double *a3 = &data.lattice[9*ci + 6];

    podfullneighborlist(nb.y, nb.alist, nb.pairlist, nb.pairnum, nb.pairnum_cumsum,
                        x, a1, a2, a3, rcutsq, pbc, atomtype, natom,
                        fastpodptr->nelements);

    for (int i = 0; i < natom; i++) {
      int n     = nb.pairnum[i];
      int start = nb.pairnum_cumsum[i];
      double xi = nb.y[dim*i], yi = nb.y[dim*i+1], zi = nb.y[dim*i+2];
      int itype = atomtype[i] - 1;
      for (int p = 0; p < n; p++) {
        int j = nb.pairlist[start + p];       // ghost/local index into nb.y
        int jtype = atomtype[nb.alist[j]] - 1;
        double dx = nb.y[dim*j]   - xi;
        double dy = nb.y[dim*j+1] - yi;
        double dz = nb.y[dim*j+2] - zi;
        double rsq = dx*dx + dy*dy + dz*dz;
        if (rsq < rinmimsq || rsq > rcutsq[itype][jtype]) continue;
        double r  = sqrt(rsq);
        double xb = (r - rinmin)/hr;
        int b = (int)(xb + 0.5);
        if (b < 0) b = 0;
        if (b > Nrho-1) b = Nrho-1;
        rho[b] += 1.0;                       // full list double-counts uniformly
      }
    }
  }

  // normalize shell
  for (int b = 0; b < Nrho; b++) {
    double r = rinmin + b*hr;
    double shell = 4.0*M_PI*r*r*hr + 1e-12;
    rho[b] /= shell;
  }

  // light 3-point smoothing to avoid spiky gains from discrete crystal shells
  std::vector<double> tmp = rho;
  for (int b = 1; b < Nrho-1; b++)
    rho[b] = 0.25*tmp[b-1] + 0.5*tmp[b] + 0.25*tmp[b+1];

  // scale to mean 1 so eps in 1/(rho+eps) is dimensionless-ish and scheme-independent
  double mean = 0.0;
  for (double v : rho) mean += v;
  mean /= Nrho;
  if (mean > 0.0) for (double &v : rho) v /= mean;
}

static double latticevolume(double *lattice)
{
  double *v1 = &lattice[0];
  double *v2 = &lattice[3];
  double *v3 = &lattice[6];

  double b0 = v1[1] * v2[2] - v1[2] * v2[1];
  double b1 = v1[2] * v2[0] - v1[0] * v2[2];
  double b2 = v1[0] * v2[1] - v1[1] * v2[0];

  return (b0 * v3[0] + b1 * v3[1] + b2 * v3[2]);
}

double FitPOD::energyforce_calculation_fastpod(double *force, const datastruct &data, int ci)
{
  int dim = 3;
  int *pbc = fastpodptr->pbc;
  double **rcutsq = fastpodptr->rcutsq;

  int natom = data.num_atom[ci];
  int natom_cumsum2 = data.num_atom_cumsum[ci];
  int *atomtype = &data.atomtype[natom_cumsum2];
  double *position = &data.position[dim * natom_cumsum2];
  double *lattice = &data.lattice[9 * ci];
  double *a1 = &lattice[0];
  double *a2 = &lattice[3];
  double *a3 = &lattice[6];

  podfullneighborlist(nb.y, nb.alist, nb.pairlist, nb.pairnum, nb.pairnum_cumsum, 
                      position, a1, a2, a3, rcutsq, pbc, atomtype, natom, 
                      fastpodptr->nelements);

  double energy = fastpodptr->energyforce(force, nb.y, atomtype, nb.alist, nb.pairlist,
                                          nb.pairnum_cumsum, natom);

  return energy;
}

void FitPOD::print_analysis(const datastruct &data, double *outarray, double *errors)
{
  int nfiles = data.data_files.size();    // number of files
  int lm = 10;
  for (int i = 0; i < nfiles; i++) lm = MAX(lm, (int) data.filenames[i].size());
  lm = lm + 2;

  std::string filename_errors =
      fmt::format("{}_{}_errors.pod", data.filenametag, data.training ? "training" : "test");
  std::string filename_analysis =
      fmt::format("{}_{}_analysis.pod", data.filenametag, data.training ? "training" : "test");

  SafeFilePtr fp_errors(fopen(filename_errors.c_str(), "w"));
  SafeFilePtr fp_analysis(fopen(filename_analysis.c_str(), "w"));
  if (!fp_errors || !fp_analysis) return;

  std::string mystr =
      fmt::format("**************** Begin of Error Analysis for the {} Data Set ****************\n",
                  data.training ? "Training" : "Test");

  utils::logmesg(lmp, mystr);
  utils::print(fp_errors, mystr);

  std::string sa(lm + 80, '-');
  sa += '\n';
  std::string sb = fmt::format(
      " {:^{}} | # configs |  # atoms  | MAE energy  | RMSE energy | MAE force  | RMSE force\n",
      "File", lm);
  utils::logmesg(lmp, sa + sb + sa);
  utils::print(fp_errors, sa + sb + sa);

  int ci = 0, m = 8, nc = 0, nf = 0;
  for (int file = 0; file < nfiles; file++) {
    utils::print(fp_analysis, "# {}\n", data.filenames[file]);
    utils::print(fp_analysis,
               "  config   # atoms       volume        energy        DFT energy     energy error   "
               "  force          DFT force       force error\n");

    int nforceall = 0;
    int nconfigs = data.num_config[file];
    nc += nconfigs;
    for (int ii = 0; ii < nconfigs; ii++) {    // loop over each configuration in a file
      utils::print(fp_analysis, "{:6}   {:8}    ", outarray[m * ci], outarray[1 + m * ci]);

      double vol = latticevolume(&data.lattice[9 * ci]);
      utils::print(fp_analysis, "{:<15.10} ", vol);

      for (int count = 2; count < m; count++)
        utils::print(fp_analysis, "{:<15.10} ", outarray[count + m * ci]);
      utils::print(fp_analysis, "\n");

      nforceall += 3 * data.num_atom[ci];
      ci += 1;
    }
    nf += nforceall;

    int q = file + 1;
    auto s =
        fmt::format("{:<{}} {:>10} {:>11}     {:<10.8f}    {:<10.8f}    {:<10.8f}    {:<10.8f}\n",
                    data.filenames[file], lm, nconfigs, nforceall / 3, errors[0 + 4 * q],
                    errors[1 + 4 * q], errors[2 + 4 * q], errors[3 + 4 * q]);
    utils::logmesg(lmp, s);
    utils::print(fp_errors, s);
  }
  utils::logmesg(lmp, sa);
  utils::print(fp_errors, sa);

  auto s =
      fmt::format("{:<{}} {:>10} {:>11}     {:<10.8f}    {:<10.8f}    {:<10.8f}    {:<10.8f}\n",
                  "All files", lm, nc, nf / 3, errors[0], errors[1], errors[2], errors[3]);
  utils::logmesg(lmp, s + sa);
  utils::print(fp_errors, "{}", s + sa);

  mystr =
      fmt::format("**************** End of Error Analysis for the {} Data Set ****************\n",
                  data.training ? "Training" : "Test");

  utils::logmesg(lmp, mystr);
  utils::print(fp_errors, mystr);
}

void FitPOD::error_analysis(const datastruct &data, double *coeff)
{
  int dim = 3;
  int nCoeffAll = desc.nCoeffAll;
  double energy;
  std::vector<double> force(dim * data.num_atom_max);

  int nfiles = data.data_files.size();       // number of files
  int num_configs = data.num_atom.size();    // number of configurations in all files

  int m = 8;
  std::vector<double> outarray(m * num_configs);
  for (int i = 0; i < m * num_configs; i++) outarray[i] = 0.0;

  std::vector<double> ssrarray(num_configs);
  for (int i = 0; i < num_configs; i++) ssrarray[i] = 0.0;

  std::vector<double> errors(4 * (nfiles + 1));
  for (int i = 0; i < 4 * (nfiles + 1); i++) errors[i] = 0.0;

  std::vector<double> newcoeff(nCoeffAll);
  for (int j = 0; j < nCoeffAll; j++) newcoeff[j] = coeff[j];

  if (comm->me == 0)
    utils::logmesg(lmp, "**************** Begin of Error Calculation ****************\n");

  int ci = 0;                                    // configuration counter
  for (int file = 0; file < nfiles; file++) {    // loop over each file in the training data set

    int nconfigs = data.num_config[file];
    for (int ii = 0; ii < nconfigs; ii++) {    // loop over each configuration in a file

      if ((ci % 100) == 0) {
        if (comm->me == 0) utils::logmesg(lmp, "Configuration: # {}\n", ci + 1);
      }

      if ((ci % comm->nprocs) == comm->me) {
        int natom = data.num_atom[ci];
        int nforce = dim * natom;

        energy = energyforce_calculation_fastpod(force.data(), data, ci);

        double DFTenergy = data.energy[ci];
        int natom_cumsum = data.num_atom_cumsum[ci];
        double *DFTforce = &data.force[dim * natom_cumsum];

        outarray[0 + m * ci] = ci + 1;
        outarray[1 + m * ci] = natom;
        outarray[2 + m * ci] = energy;
        outarray[3 + m * ci] = DFTenergy;
        outarray[4 + m * ci] = fabs(DFTenergy - energy) / natom;
        outarray[5 + m * ci] = podArrayNorm(force.data(), nforce);
        outarray[6 + m * ci] = podArrayNorm(DFTforce, nforce);

        double diff, sum = 0.0, ssr = 0.0;
        for (int j = 0; j < dim * natom; j++) {
          diff = DFTforce[j] - force[j];
          sum += fabs(diff);
          ssr += diff * diff;
        }
        outarray[7 + m * ci] = sum / nforce;
        ssrarray[ci] = ssr;
      }

      ci += 1;
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, outarray.data(), m * num_configs, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, ssrarray.data(), num_configs, MPI_DOUBLE, MPI_SUM, world);

  ci = 0;    // configuration counter
  int nc = 0, nf = 0;
  for (int file = 0; file < nfiles; file++) {    // loop over each file in the training data set

    double emae = 0.0, essr = 0.0, fmae = 0.0, fssr = 0.0;
    int nforceall = 0;

    int nconfigs = data.num_config[file];
    nc += nconfigs;
    for (int ii = 0; ii < nconfigs; ii++) {    // loop over each configuration in a file

      int natom = data.num_atom[ci];
      int nforce = dim * natom;

      emae += outarray[4 + m * ci];    // sum_c |ePOD_c - eDFT_c|/natom_c
      essr +=
          outarray[4 + m * ci] * outarray[4 + m * ci];    // sum_c |ePOD_c - eDFT_c|^2/(natom_c)^2
      fmae += outarray[7 + m * ci] * nforce;              // sum_c |fPOD_c - fDFT_c|
      fssr += ssrarray[ci];
      nforceall += nforce;
      ci += 1;
    }

    int q = file + 1;
    if (nconfigs == 0) nconfigs = 1;
    if (nforceall == 0) nforceall = 1;
    errors[0 + 4 * q] = emae / nconfigs;
    errors[1 + 4 * q] = sqrt(essr / nconfigs);
    errors[2 + 4 * q] = fmae / nforceall;
    errors[3 + 4 * q] = sqrt(fssr / nforceall);

    nf += nforceall;
    errors[0] += emae;    // sum_c |ePOD_c - eDFT_c|/natom_c
    errors[1] += essr;    // sum_c |ePOD_c - eDFT_c|^2/(natom_c)^2
    errors[2] += fmae;
    errors[3] += fssr;
  }

  if (nc == 0) nc = 1;
  if (nf == 0) nf = 1;
  errors[0] = errors[0] / nc;          // (1/Nc) * sum_c |ePOD_c - eDFT_c|/natom_c
  errors[1] = sqrt(errors[1] / nc);    // sqrt { (1/Nc) *  sum_c |ePOD_c - eDFT_c|^2/(natom_c)^2 }
  errors[2] = errors[2] / nf;
  errors[3] = sqrt(errors[3] / nf);

  if (comm->me == 0) {
    utils::logmesg(lmp, "**************** End of Error Calculation ****************\n");
    print_analysis(data, outarray.data(), errors.data());
  }
}

void FitPOD::energyforce_calculation(const datastruct &data)
{
  int dim = 3;
  double energy;
  std::vector<double> force(1 + dim * data.num_atom_max);

  int nfiles = data.data_files.size();    // number of files

  if (comm->me == 0)
    utils::logmesg(lmp, "**************** Begin of Energy/Force Calculation ****************\n");

  int ci = 0;                                    // configuration counter
  for (int file = 0; file < nfiles; file++) {    // loop over each file in the data set

    int nconfigs = data.num_config[file];
    for (int ii = 0; ii < nconfigs; ii++) {    // loop over each configuration in a file
      if ((ci % 100) == 0) {
        if (comm->me == 0) utils::logmesg(lmp, "Configuration: # {}\n", ci + 1);
      }

      int natom = data.num_atom[ci];
      int nforce = dim * natom;

      if ((ci % comm->nprocs) == comm->me) {
        energy = energyforce_calculation_fastpod(force.data() + 1, data, ci);

        // save energy and force into a binary file
        force[0] = energy;
        std::string filename = "energyforce_config" + std::to_string(ci + 1) + ".bin";

        SafeFilePtr fp = fopen(filename.c_str(), "wb");

        fwrite(reinterpret_cast<char *>(force.data()), sizeof(double) * (1 + nforce), 1, fp);
      }
      ci += 1;
    }
  }
  if (comm->me == 0)
    utils::logmesg(lmp, "**************** End of Energy/Force Calculation ****************\n");
}

void FitPOD::podArrayFill(int *output, int start, int length)
{
  for (int j = 0; j < length; ++j) output[j] = start + j;
}

double FitPOD::podArraySum(double *a, int n)
{
  double e = a[0];
  for (int i = 1; i < n; i++) e += a[i];
  return e;
}

double FitPOD::podArrayMin(double *a, int n)
{
  double b = a[0];
  for (int i = 1; i < n; i++)
    if (a[i] < b) b = a[i];
  return b;
}

double FitPOD::podArrayMax(double *a, int n)
{
  double b = a[0];
  for (int i = 1; i < n; i++)
    if (a[i] > b) b = a[i];
  return b;
}

int FitPOD::podArrayMin(int *a, int n)
{
  int b = a[0];
  for (int i = 1; i < n; i++)
    if (a[i] < b) b = a[i];
  return b;
}

int FitPOD::podArrayMax(int *a, int n)
{
  int b = a[0];
  for (int i = 1; i < n; i++)
    if (a[i] > b) b = a[i];
  return b;
}

void FitPOD::podKron(double *C, double *A, double *B, double alpha, int M1, int M2)
{
  int M = M1 * M2;
  for (int idx = 0; idx < M; idx++) {
    int ib = idx % M2;
    int ia = (idx - ib) / M2;
    C[idx] += alpha * A[ia] * B[ib];
  }
}

void FitPOD::podCumsum(int *output, int *input, int length)
{
  output[0] = 0;
  for (int j = 1; j < length; ++j) output[j] = input[j - 1] + output[j - 1];
}

double FitPOD::podArrayNorm(double *a, int n)
{
  double e = a[0] * a[0];
  for (int i = 1; i < n; i++) e += a[i] * a[i];
  return sqrt(e);
}

double FitPOD::podArrayErrorNorm(double *a, double *b, int n)
{
  double e = (a[0] - b[0]) * (a[0] - b[0]);
  for (int i = 1; i < n; i++) e += (a[i] - b[i]) * (a[i] - b[i]);
  return sqrt(e);
}

void FitPOD::podArraySetValue(double *y, double a, int n)
{
  for (int i = 0; i < n; i++) y[i] = a;
}

void FitPOD::podArrayCopy(double *y, double *x, int n)
{
  for (int i = 0; i < n; i++) y[i] = x[i];
}

void FitPOD::rotation_matrix(double *Rmat, double alpha, double beta, double gamma)
{
  double ca = cos(alpha);
  double cb = cos(beta);
  double cg = cos(gamma);
  double sa = sin(alpha);
  double sb = sin(beta);
  double sg = sin(gamma);

  Rmat[0] = ca * cg * cb - sa * sg;
  Rmat[3] = -ca * cb * sg - sa * cg;
  Rmat[6] = ca * sb;

  Rmat[1] = sa * cg * cb + ca * sg;
  Rmat[4] = -sa * cb * sg + ca * cg;
  Rmat[7] = sa * sb;

  Rmat[2] = -sb * cg;
  Rmat[5] = sb * sg;
  Rmat[8] = cb;
}

void FitPOD::matrix33_multiplication(double *xrot, double *Rmat, double *x, int natom)
{
  double x1, x2, x3;
  for (int i = 0; i < natom; i++) {
    x1 = x[0 + 3 * i];
    x2 = x[1 + 3 * i];
    x3 = x[2 + 3 * i];
    xrot[0 + 3 * i] = Rmat[0] * x1 + Rmat[3] * x2 + Rmat[6] * x3;
    xrot[1 + 3 * i] = Rmat[1] * x1 + Rmat[4] * x2 + Rmat[7] * x3;
    xrot[2 + 3 * i] = Rmat[2] * x1 + Rmat[5] * x2 + Rmat[8] * x3;
  }
}

void FitPOD::matrix33_inverse(double *invA, double *A1, double *A2, double *A3)
{
  double a11 = A1[0];
  double a21 = A1[1];
  double a31 = A1[2];
  double a12 = A2[0];
  double a22 = A2[1];
  double a32 = A2[2];
  double a13 = A3[0];
  double a23 = A3[1];
  double a33 = A3[2];
  double detA = (a11 * a22 * a33 - a11 * a23 * a32 - a12 * a21 * a33 + a12 * a23 * a31 +
                 a13 * a21 * a32 - a13 * a22 * a31);

  invA[0] = (a22 * a33 - a23 * a32) / detA;
  invA[1] = (a23 * a31 - a21 * a33) / detA;
  invA[2] = (a21 * a32 - a22 * a31) / detA;
  invA[3] = (a13 * a32 - a12 * a33) / detA;
  invA[4] = (a11 * a33 - a13 * a31) / detA;
  invA[5] = (a12 * a31 - a11 * a32) / detA;
  invA[6] = (a12 * a23 - a13 * a22) / detA;
  invA[7] = (a13 * a21 - a11 * a23) / detA;
  invA[8] = (a11 * a22 - a12 * a21) / detA;
}

void FitPOD::triclinic_lattice_conversion(double *a, double *b, double *c, double *A, double *B,
                                          double *C)
{
  double Anorm = sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
  double Bnorm = sqrt(B[0] * B[0] + B[1] * B[1] + B[2] * B[2]);
  double Cnorm = sqrt(C[0] * C[0] + C[1] * C[1] + C[2] * C[2]);

  double Ahat[3];
  Ahat[0] = A[0] / Anorm;
  Ahat[1] = A[1] / Anorm;
  Ahat[2] = A[2] / Anorm;

  double ax = Anorm;
  double bx = B[0] * Ahat[0] + B[1] * Ahat[1] + B[2] * Ahat[2];    //dot(B,Ahat);
  double by = sqrt(Bnorm * Bnorm - bx * bx);    //sqrt(Bnorm^2 - bx^2);// #norm(cross(Ahat,B));
  double cx = C[0] * Ahat[0] + C[1] * Ahat[1] + C[2] * Ahat[2];    // dot(C,Ahat);
  double cy =
      (B[0] * C[0] + B[1] * C[1] + B[2] * C[2] - bx * cx) / by;    // (dot(B, C) - bx*cx)/by;
  double cz = sqrt(Cnorm * Cnorm - cx * cx - cy * cy);             // sqrt(Cnorm^2 - cx^2 - cy^2);

  a[0] = ax;
  a[1] = 0.0;
  a[2] = 0.0;
  b[0] = bx;
  b[1] = by;
  b[2] = 0.0;
  c[0] = cx;
  c[1] = cy;
  c[2] = cz;
}

// Function to calculate Euclidean distance between two points in N-dimensional space
double FitPOD::squareDistance(const double *a, const double *b, int DIMENSIONS)
{
  double sum = 0.0;
  for (int i = 0; i < DIMENSIONS; i++) { sum += (a[i] - b[i]) * (a[i] - b[i]); }
  return sum;
}

// Function to assign points to the nearest cluster
void FitPOD::assignPointsToClusters(double *points, double *centroids, int *assignments,
                                    int *clusterSizes, int NUM_POINTS, int NUM_CLUSTERS,
                                    int DIMENSIONS)
{
  // Initialize clusterSizes to zero
  for (int i = 0; i < NUM_CLUSTERS; i++) { clusterSizes[i] = 0; }

  for (int i = 0; i < NUM_POINTS; i++) {
    double minDist = squareDistance(&points[i * DIMENSIONS], &centroids[0], DIMENSIONS);
    int closestCluster = 0;
    for (int j = 1; j < NUM_CLUSTERS; j++) {
      double dist = squareDistance(&points[i * DIMENSIONS], &centroids[j * DIMENSIONS], DIMENSIONS);
      if (dist < minDist) {
        minDist = dist;
        closestCluster = j;
      }
    }
    assignments[i] = closestCluster;
    clusterSizes[closestCluster]++;
  }
}

// Function to update centroids based on point assignments
void FitPOD::updateCentroids(double *points, double *centroids, int *assignments, int *clusterSizes,
                             int NUM_POINTS, int NUM_CLUSTERS, int DIMENSIONS)
{
  // Reset centroids for recalculation
  for (int i = 0; i < NUM_CLUSTERS * DIMENSIONS; i++) { centroids[i] = 0.0; }

  // Accumulate sum of points in each cluster
  for (int i = 0; i < NUM_POINTS; i++) {
    int cluster = assignments[i];
    for (int j = 0; j < DIMENSIONS; j++) {
      centroids[cluster * DIMENSIONS + j] += points[i * DIMENSIONS + j];
    }
  }

  // Use MPI_Allreduce to sum up the local sums and cluster sizes across all processes
  MPI_Allreduce(MPI_IN_PLACE, centroids, NUM_CLUSTERS * DIMENSIONS, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, clusterSizes, NUM_CLUSTERS, MPI_INT, MPI_SUM, world);

  // Divide by number of points to get the mean (centroid)
  for (int i = 0; i < NUM_CLUSTERS; i++) {
    if (clusterSizes[i] != 0) {
      for (int j = 0; j < DIMENSIONS; j++) { centroids[i * DIMENSIONS + j] /= clusterSizes[i]; }
    }
  }
}

// Function for K-means clustering
void FitPOD::KmeansClustering(double *points, double *centroids, int *assignments,
                              int *clusterSizes, int NUM_POINTS, int NUM_CLUSTERS, int DIMENSIONS,
                              int MAX_ITER)
{
  for (int iter = 0; iter < MAX_ITER; iter++) {
    assignPointsToClusters(points, centroids, assignments, clusterSizes, NUM_POINTS, NUM_CLUSTERS,
                           DIMENSIONS);
    updateCentroids(points, centroids, assignments, clusterSizes, NUM_POINTS, NUM_CLUSTERS,
                    DIMENSIONS);
  }
}

void FitPOD::savematrix2binfile(const std::string &filename, double *A, int nrows, int ncols)
{
  SafeFilePtr fp = fopen(filename.c_str(), "wb");
  double sz[2];
  sz[0] = (double) nrows;
  sz[1] = (double) ncols;
  fwrite(reinterpret_cast<char *>(sz), sizeof(double) * 2, 1, fp);
  fwrite(reinterpret_cast<char *>(A), sizeof(double) * (nrows * ncols), 1, fp);
}

void FitPOD::saveintmatrix2binfile(const std::string &filename, int *A, int nrows, int ncols)
{
  SafeFilePtr fp = fopen(filename.c_str(), "wb");
  int sz[2];
  sz[0] = nrows;
  sz[1] = ncols;
  fwrite(reinterpret_cast<char *>(sz), sizeof(int) * 2, 1, fp);
  fwrite(reinterpret_cast<char *>(A), sizeof(int) * (nrows * ncols), 1, fp);
}

void FitPOD::savedata2textfile(const std::string &filename, const std::string &text, double *A,
                               int n, int m, int dim)
{
  if (comm->me == 0) {
    int precision = 15;
    SafeFilePtr fp = fopen(filename.c_str(), "w");
    if (dim == 1) {
      utils::print(fp, text, n);
      for (int i = 0; i < n; i++) utils::print(fp, "{:<10.{}f} \n", A[i], precision);
    } else if (dim == 2) {
      utils::print(fp, text, n);
      utils::print(fp, "{} \n", m);
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) utils::print(fp, "{:<10.{}f}     ", A[j + i * n], precision);
        utils::print(fp, "   \n");
      }
    }
  }
}
