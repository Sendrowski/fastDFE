.. _reference.config:

Configuration files
===================

Basic usage
-----------

Inference objects can be created directly from configuration files, which facilitates the reproducibility and consistency of analyses, and simplifies the management of hyperparameters and other settings:

.. tab-set::
   :sync-group: language
   :class: code-tabs

   .. tab-item:: :fab:`python` Python
      :sync: python

      .. code-block:: python

          import fastdfe as fd

          config = fd.Config.from_file("config.yaml")

          inf = fd.BaseInference.from_config(config)

   .. tab-item:: :fab:`r-project` R
      :sync: r

      .. code-block:: r

          library(fastdfe)
          fd <- load_fastdfe()

          config <- fd$Config$from_file("config.yaml")

          inf <- fd$BaseInference$from_config(config)

A configuration file typically looks as follows, and reflects the parameters of :class:`~fastdfe.base_inference.BaseInference`, :class:`~fastdfe.joint_inference.JointInference` and :class:`~fastdfe.config.Config`.

.. code-block:: yaml

    # The DFE parametrization to be used.
    # GammaExpParametrization is parametrized by S_b, S_d, p_b, and b
    model: GammaExpParametrization
    # The parameters held fixed. Note that we need to
    # specify a type, even when only one type is present.
    # For BaseInference we thus resort to the 'all' type
    fixed_params:
      all:
        eps: 0
    # The initial values. This is not very important
    # as fastDFE is fast and usually runs the numerical
    # optimization multiple times for different initial values
    x0:
      all:
        S_b: 2
        S_d: -10000
        p_b: 0.1
        b: 0.2
        eps: 0
    # The number of independent runs
    n_runs: 10
    # The parameter bounds. Note that we do not specify the
    # type here. This property should be left untouched in
    # most cases.
    bounds: { }
    # The scales over which the parameters are optimized,
    # i.e. 'lin', 'log', or 'symlog'.
    scales: { }
    # Whether to perform bootstraps directly when invoking
    # Inference.run()
    do_bootstrap: true
    # The number of bootstraps
    n_bootstraps: 100
    # Whether to parallelize the computations across all
    # available cores
    parallelize: true
    # The seed for the random number generator which is
    # used for sampling the bootstraps and changing the
    # initial values among other this. Fixing the seed
    # guarantees deterministic behaviour
    seed: 0
    # The neutral SFS for the different types. Here we just
    # have the 'all' type.
    sfs_neut:
      all:
        - 173705
        - 3294
        - 1115
        - 534
        - 326
        - 239
        - 225
        - 214
        - 231
        - 176
        - 73
    # The selected SFS for the different types.
    sfs_sel:
      all:
        - 792884
        - 4762
        - 1397
        - 720
        - 423
        - 259
        - 247
        - 256
        - 265
        - 209
        - 90

All available configuration options are described in :class:`~fastdfe.config.Config`.

A configuration file can also be created from an existing inference object:

.. tab-set::
   :sync-group: language
   :class: code-tabs

   .. tab-item:: :fab:`python` Python
      :sync: python

      .. code-block:: python

          inf.create_config().to_file("config.yaml")

   .. tab-item:: :fab:`r-project` R
      :sync: r

      .. code-block:: r

          inf$create_config()$to_file("config.yaml")

JSON files
----------

Configuration files can also be written in JSON and loaded with :meth:`Config.from_json() <fastdfe.config.Config.from_json>`. A JSON configuration file typically looks as follows:

.. code-block:: json

    {
      "model": "GammaExpParametrization",
      "fixed_params": {
        "all": {
          "eps": 0
        }
      },
      "x0": {
        "all": {
          "S_b": 2,
          "S_d": -10000,
          "p_b": 0.1,
          "b": 0.2,
          "eps": 0
        }
      },
      "n_runs": 10,
      "bounds": {},
      "scales": {},
      "do_bootstrap": true,
      "n_bootstraps": 100,
      "parallelize": true,
      "seed": 0,
      "sfs_neut": {
        "all": [
          173705,
          3294,
          1115,
          534,
          326,
          239,
          225,
          214,
          231,
          176,
          73
        ]
      },
      "sfs_sel": {
        "all": [
          792884,
          4762,
          1397,
          720,
          423,
          259,
          247,
          256,
          265,
          209,
          90
        ]
      }
    }

Joint inference example
-----------------------

A more involved configuration file, configuring a joint inference with a number of fixed and shared parameters as well as covariates, might look as follows:

.. code-block:: yaml

    model: GammaExpParametrization
    # parameter b is shared among all types
    shared_params:
      - types: all
        params:
          - b
    fixed_params:
      # eps is fixed for all types
      all:
        eps: 0
      # S_b is fixed for pubescens
      pubescens:
        S_b: 90
    # parameter S_d has covariates
    covariates:
      - param: S_d
        values:
          pendula: -32623.595481483513
          pubescens: -426.59080558648185
    bounds: { }
    scales: { }
    x0: { }
    do_bootstrap: true
    linearized: true
    loss_type: likelihood
    n_bootstraps: 100
    n_runs: 10
    opts_mle: { }
    parallelize: true
    seed: 0
    sfs_neut:
      pendula:
        - 177130
        - 997
        - 441
        - 228
        - 156
        - 117
        - 114
        - 83
        - 105
        - 109
        - 652
      pubescens:
        - 172528
        - 3612
        - 1359
        - 790
        - 584
        - 427
        - 325
        - 234
        - 166
        - 76
        - 31
    sfs_sel:
      pendula:
        - 797939
        - 1329
        - 499
        - 265
        - 162
        - 104
        - 117
        - 90
        - 94
        - 119
        - 794
      pubescens:
        - 791106
        - 5326
        - 1741
        - 1005
        - 756
        - 546
        - 416
        - 294
        - 177
        - 104
        - 41
