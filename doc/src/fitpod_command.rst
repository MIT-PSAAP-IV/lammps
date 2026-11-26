.. index:: fitpod

fitpod command
==============

Syntax
""""""

.. code-block:: LAMMPS

   fitpod Ta_param.pod Ta_data.pod Ta_coefficients.pod

* fitpod = style name of this command
* Ta_param.pod = an input file that describes proper orthogonal descriptors (PODs)
* Ta_data.pod = an input file that specifies DFT data used to fit a POD potential
* Ta_coefficients.pod (optional) = an input file that specifies trainable coefficients of a POD potential

Examples
""""""""

.. code-block:: LAMMPS

   fitpod Ta_param.pod Ta_data.pod
   fitpod Ta_param.pod Ta_data.pod Ta_coefficients.pod

Description
"""""""""""
.. versionadded:: 22Dec2022

Fit a machine-learning interatomic potential (ML-IAP) based on proper
orthogonal descriptors (POD); please see :ref:`(Nguyen and Rohskopf)
<Nguyen20222a>`, :ref:`(Nguyen2023) <Nguyen20232a>`, :ref:`(Nguyen2024)
<Nguyen20242a>`, and :ref:`(Nguyen and Sema) <Nguyen20243a>` for details.
The fitted POD potential can be used to run MD simulations via
:doc:`pair_style pod <pair_pod>`.

Two input files are required for this command. The first input file
describes a POD potential parameter settings, while the second input
file specifies the DFT data used for the fitting procedure. All keywords
except *species* have default values. If a keyword is not set in the
input file, its default value is used. The table below has one-line
descriptions of all the keywords that can be used in the first input
file (i.e. ``Ta_param.pod``)

.. list-table::
   :header-rows: 1
   :widths: 40 9 10 41

   * - Keyword
     - Default
     - Type
     - Description
   * - species
     - (none)
     - STRING
     - Chemical symbols for all elements in the system and have to match XYZ training files.
   * - pbc
     - 1 1 1
     - INT
     - three integer constants specify boundary conditions
   * - rin
     - 0.5
     - REAL
     - a real number specifies the inner cut-off radius
   * - rcut
     - 5.0
     - REAL
     - a real number specifies the outer cut-off radius
   * - bessel_polynomial_degree
     - 4
     - INT
     - the maximum degree of Bessel polynomials
   * - inverse_polynomial_degree
     - 8
     - INT
     - the maximum degree of inverse radial basis functions
   * - number_of_environment_clusters
     - 1
     - INT
     - the number of clusters for environment-adaptive potentials
   * - number_of_principal_components
     - 2
     - INT
     - the number of principal components for dimensionality reduction
   * - onebody
     - 1
     - BOOL
     - turns on/off one-body potential
   * - twobody_number_radial_basis_functions
     - 8
     - INT
     - number of radial basis functions for two-body potential
   * - threebody_number_radial_basis_functions
     - 6
     - INT
     - number of radial basis functions for three-body potential
   * - threebody_angular_degree
     - 5
     - INT
     - angular degree for three-body potential
   * - fourbody_number_radial_basis_functions
     - 4
     - INT
     - number of radial basis functions for four-body potential
   * - fourbody_angular_degree
     - 3
     - INT
     - angular degree for four-body potential
   * - fivebody_number_radial_basis_functions
     - 0
     - INT
     - number of radial basis functions for five-body potential
   * - fivebody_angular_degree
     - 0
     - INT
     - angular degree for five-body potential
   * - sixbody_number_radial_basis_functions
     - 0
     - INT
     - number of radial basis functions for six-body potential
   * - sixbody_angular_degree
     - 0
     - INT
     - angular degree for six-body potential
   * - sevenbody_number_radial_basis_functions
     - 0
     - INT
     - number of radial basis functions for seven-body potential
   * - sevenbody_angular_degree
     - 0
     - INT
     - angular degree for seven-body potential

Note that both the number of radial basis functions and angular degree
must decrease as the body order increases. The next table describes all
keywords that can be used in the second input file (i.e. ``Ta_data.pod``
in the example above):


.. list-table::
   :header-rows: 1
   :widths: 38 9 10 43

   * - Keyword
     - Default
     - Type
     - Description
   * - file_format
     - extxyz
     - STRING
     - only the extended xyz format (extxyz) is currently supported
   * - file_extension
     - xyz
     - STRING
     - extension of the data files
   * - path_to_training_data_set
     - (none)
     - STRING
     - specifies the path to training data files in double quotes
   * - path_to_test_data_set
     - ""
     - STRING
     - specifies the path to test data files in double quotes
   * - path_to_environment_configuration_set
     - ""
     - STRING
     - specifies the path to environment configuration files in double quotes
   * - fraction_training_data_set
     - 1.0
     - REAL
     - a real number (<= 1.0) specifies the fraction of the training set used to fit POD
   * - randomize_training_data_set
     - 0
     - BOOL
     - turns on/off randomization of the training set
   * - fraction_test_data_set
     - 1.0
     - REAL
     - a real number (<= 1.0) specifies the fraction of the test set used to validate POD
   * - randomize_test_data_set
     - 0
     - BOOL
     - turns on/off randomization of the test set
   * - fitting_weight_energy
     - 100.0
     - REAL
     - a real constant specifies the weight for energy in the least-squares fit
   * - fitting_weight_force
     - 1.0
     - REAL
     - a real constant specifies the weight for force in the least-squares fit
   * - fitting_regularization_parameter
     - 1.0e-10
     - REAL
     - a real constant specifies the regularization parameter in the least-squares fit
   * - cluster_smoothing_parameter
     - 0.0
     - REAL
     - couples the coefficients of neighboring environment clusters (see below)
   * - kmeans_initialization
     - sequential
     - STRING
     - ``sequential`` or ``quantile`` starting centroids for the k-means clustering
   * - kmeans_max_iterations
     - 100
     - INT
     - maximum number of k-means iterations
   * - cluster_min_training_configs
     - 0
     - INT
     - minimum number of training configurations for each environment cluster (see below)
   * - error_analysis_for_training_data_set
     - 0
     - BOOL
     - turns on/off error analysis for the training data set
   * - error_analysis_for_test_data_set
     - 0
     - BOOL
     - turns on/off error analysis for the test data set
   * - basename_for_output_files
     - pod
     - STRING
     - a basename string added to the output files
   * - precision_for_pod_coefficients
     - 8
     - INT
     - number of digits after the decimal points for numbers in the coefficient file
   * - group_weights
     - global
     - STRING
     - ``table`` uses group weights defined for each group named by filename

All keywords except *path_to_training_data_set* have default values. If
a keyword is not set in the input file, its default value is used.  After
successful training, a number of output files are produced, if enabled:

* ``<basename>_training_errors.pod``  reports the errors in energy and forces for the training data set
* ``<basename>_training_analysis.pod`` reports detailed errors for all training configurations
* ``<basename>_test_errors.pod`` reports errors for the test data set
* ``<basename>_test_analysis.pod`` reports detailed errors for all test configurations
* ``<basename>_coefficients.pod`` contains the coefficients of the POD potential
* ``<basename>_cluster_training_support.pod`` reports the number of training configurations for each environment cluster

After training the POD potential, ``Ta_param.pod`` and
``<basename>_coefficients.pod`` are the two files needed to use the POD
potential in LAMMPS.  See :doc:`pair_style pod <pair_pod>` for using the
POD potential. Examples about training and using POD potentials are
found in the directory lammps/examples/PACKAGES/pod and the Github repo
https://github.com/cesmix-mit/pod-examples.

Loss Function Group Weights
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *group_weights* keyword in the ``data.pod`` file is responsible for
weighting certain groups of configurations in the loss function. For
example:

.. code-block:: LAMMPS

    group_weights table
    Displaced_A15 100.0 1.0
    Displaced_BCC 100.0 1.0
    Displaced_FCC 100.0 1.0
    Elastic_BCC   100.0 1.0
    Elastic_FCC   100.0 1.0
    GSF_110       100.0 1.0
    GSF_112       100.0 1.0
    Liquid        100.0 1.0
    Surface       100.0 1.0
    Volume_A15    100.0 1.0
    Volume_BCC    100.0 1.0
    Volume_FCC    100.0 1.0

This will apply an energy weight of ``100.0`` and a force weight of
``1.0`` for all groups in the ``Ta`` example. The groups are named by
their respective filename. If certain groups are left out of this table,
then the globally defined weights from the ``fitting_weight_energy`` and
``fitting_weight_force`` keywords will be used.

POD Potential
"""""""""""""

We consider a multi-element system of *N* atoms with :math:`N_\mathrm{e}`
unique elements.  We denote by :math:`\boldsymbol r_n` and :math:`Z_n`
position vector and type of an atom *n* in the system,
respectively. Note that we have :math:`Z_n \in \{1, \ldots, N_\mathrm{e}
\}`, :math:`\boldsymbol R = (\boldsymbol r_1, \boldsymbol r_2, \ldots,
\boldsymbol r_N) \in \mathbb{R}^{3N}`, and :math:`\boldsymbol Z = (Z_1,
Z_2, \ldots, Z_N) \in \mathbb{N}^{N}`. The total energy of the
POD potential is expressed as :math:`E(\boldsymbol R, \boldsymbol Z) =
\sum_{i=1}^N E_i(\boldsymbol R_i, \boldsymbol Z_i)`, where

.. math::

    E_i(\boldsymbol R_i, \boldsymbol Z_i) \ = \ \sum_{m=1}^M c_m \mathcal{D}_{im}(\boldsymbol R_i, \boldsymbol Z_i)


Here :math:`c_m` are trainable coefficients and
:math:`\mathcal{D}_{im}(\boldsymbol R_i, \boldsymbol Z_i)` are per-atom
POD descriptors. Summing the per-atom descriptors over :math:`i` yields
the global descriptors :math:`d_m(\boldsymbol R, \boldsymbol Z) =
\sum_{i=1}^N \mathcal{D}_{im}(\boldsymbol R_i, \boldsymbol Z_i)`.  It
thus follows that :math:`E(\boldsymbol R, \boldsymbol Z) = \sum_{m=1}^M
c_m d_m(\boldsymbol R, \boldsymbol Z)`.

The per-atom POD descriptors include one, two, three, four, five, six,
and seven-body descriptors, which can be specified in the first input
file. Furthermore, the per-atom POD descriptors also depend on the
number of environment clusters specified in the first input file.
Please see :ref:`(Nguyen2024) <Nguyen20242a>` and :ref:`(Nguyen and Sema)
<Nguyen20243a>` for the detailed description of the per-atom POD
descriptors.

Training
""""""""

A POD potential is trained using the least-squares regression against
density functional theory (DFT) data.  Let :math:`J` be the number of
training configurations, with :math:`N_j` being the number of atoms in
the j-th configuration. The training configurations are extracted from
the extended XYZ files located in a directory (i.e.,
path_to_training_data_set in the second input file).  Let
:math:`\{E^{\star}_j\}_{j=1}^{J}` and :math:`\{\boldsymbol
F^{\star}_j\}_{j=1}^{J}` be the DFT energies and forces for :math:`J`
configurations. Next, we calculate the global descriptors and their
derivatives for all training configurations. Let :math:`d_{jm}, 1 \le m
\le M`, be the global descriptors associated with the j-th
configuration, where :math:`M` is the number of global descriptors. We
then form a matrix :math:`\boldsymbol A \in \mathbb{R}^{J \times M}`
with entries :math:`A_{jm} = d_{jm}/ N_j` for :math:`j=1,\ldots,J` and
:math:`m=1,\ldots,M`.  Moreover, we form a matrix :math:`\boldsymbol B
\in \mathbb{R}^{\mathcal{N} \times M}` by stacking the derivatives of
the global descriptors for all training configurations from top to
bottom, where :math:`\mathcal{N} = 3\sum_{j=1}^{J} N_j`.

The coefficient vector :math:`\boldsymbol c` of the POD potential is
found by solving the following least-squares problem

.. math::

    {\min}_{\boldsymbol c \in \mathbb{R}^{M}} \ w_E \|\boldsymbol A \boldsymbol c - \bar{\boldsymbol E}^{\star} \|^2 + w_F \|\boldsymbol B \boldsymbol c + \boldsymbol F^{\star} \|^2 + w_R \|\boldsymbol c \|^2,

where :math:`w_E` and :math:`w_F` are weights for the energy
(*fitting_weight_energy*) and force (*fitting_weight_force*),
respectively; and :math:`w_R` is the regularization parameter
(*fitting_regularization_parameter*).  Here :math:`\bar{\boldsymbol
E}^{\star} \in \mathbb{R}^{J}` is a vector of with entries
:math:`\bar{E}^{\star}_j = E^{\star}_j/N_j` and :math:`\boldsymbol
F^{\star}` is a vector of :math:`\mathcal{N}` entries obtained by
stacking :math:`\{\boldsymbol F^{\star}_j\}_{j=1}^{J}` from top to
bottom.

Environment clusters
""""""""""""""""""""

.. versionadded:: TBD

For environment-adaptive POD potentials with more than one environment
cluster, every cluster has its own set of coefficients.  The cluster
centroids are obtained by k-means clustering of the latent (principal
component) descriptors of the environment configuration set.  Real data
sets sample the latent space unevenly: the clusters at the edges of the
latent space, which often describe high-energy atomic environments, can
contain atoms from only a few training configurations.  The coefficients
of such clusters are then poorly determined, and adding more clusters
can lower the training errors while raising the test errors.  Four
keywords in the second input file control this behavior.

The *cluster_smoothing_parameter* :math:`w_S` adds the penalty

.. math::

    w_S \sum_{m=1}^{M_d} \bar{A}_{m} \sum_{k=1}^{K-1} \left(c_{k,m} - c_{k+1,m}\right)^2

to the least-squares problem, where :math:`c_{k,m}` is the coefficient
of base descriptor :math:`m` in cluster :math:`k`, the :math:`K`
clusters are ordered along the latent axis, and :math:`\bar{A}_{m}` is
the average diagonal entry of the least-squares matrix for descriptor
:math:`m` over all clusters, which makes :math:`w_S` independent of the
descriptor scale.  Clusters with many training configurations are
hardly affected, while the coefficients of clusters with few training
configurations are pulled toward those of their neighbors rather than
toward zero.  This also reduces spurious forces caused by large
differences between the energies predicted by neighboring clusters.
Values between 1.0e-4 and 1.0e-1 are a reasonable starting range; a
value of zero turns the penalty off.  This keyword requires a
one-dimensional latent space.

The *kmeans_initialization* keyword selects the starting centroids of
the k-means clustering.  With *sequential*, the atoms are split into
groups of equal size in the order in which they are read.  With
*quantile*, the atoms are sorted along the first latent component
before they are split, so the starting centroids already follow the
distribution of the data.  The clustering stops as soon as no atom
changes its cluster, or after *kmeans_max_iterations* iterations.

After the clustering, the number of training configurations that have
at least one atom nearest to each cluster centroid is written to the
file ``<basename>_cluster_training_support.pod``.  This number can be
much smaller than the number of atoms in a cluster, in particular when
the environment configuration set contains configurations that are not
part of the training set.  If *cluster_min_training_configs* is larger
than zero, clusters with fewer training configurations are merged into
their neighbor with fewer training configurations, and the clusters
freed this way are used to split the clusters with the most training
configurations, so the total number of clusters stays the same.  The
cluster occupancy file is then rewritten from the training atoms.  This
keyword requires a one-dimensional latent space.

Validation
""""""""""

POD potential can be validated on a test dataset in a directory
specified by setting path_to_test_data_set in the second input file.  It
is possible to validate the POD potential after the training is
complete.  This is done by providing the coefficient file as an input to
:doc:`fitpod <fitpod_command>`, for example,

.. code-block:: LAMMPS

   fitpod Ta_param.pod Ta_data.pod Ta_coefficients.pod

Restrictions
""""""""""""

This command is part of the ML-POD package.  It is only enabled if
LAMMPS was built with that package. See the :doc:`Build package
<Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`pair_style pod <pair_pod>`,
:doc:`compute pod/atom <compute_pod_atom>`,
:doc:`compute podd/atom <compute_pod_atom>`,
:doc:`compute pod/local <compute_pod_atom>`,
:doc:`compute pod/global <compute_pod_atom>`

Default
"""""""

The keyword defaults are also given in the description of the input files.

----------

.. _Nguyen20222a:

**(Nguyen and Rohskopf)** Nguyen and Rohskopf,  Journal of Computational Physics, 480, 112030, (2023).

.. _Nguyen20232a:

**(Nguyen2023)** Nguyen, Physical Review B, 107(14), 144103, (2023).

.. _Nguyen20242a:

**(Nguyen2024)** Nguyen, Journal of Computational Physics, 113102, (2024).

.. _Nguyen20243a:

**(Nguyen and Sema)** Nguyen and Sema, https://arxiv.org/abs/2405.00306, (2024).


