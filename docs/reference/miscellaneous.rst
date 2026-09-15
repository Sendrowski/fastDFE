.. _reference.miscellaneous:

Miscellaneous
=============

Logging
-------

``fastdfe`` uses the standard Python :mod:`logging` module for logging. By default, ``fastdfe`` logs to the console at the ``INFO`` level. The logging level can be changed, for example to ``DEBUG``, as follows:

.. tab-set::
   :sync-group: language
   :class: code-tabs

   .. tab-item:: :fab:`python` Python
      :sync: python

      .. code-block:: python

          import fastdfe as fd

          fd.logger.setLevel("DEBUG")

   .. tab-item:: :fab:`r-project` R
      :sync: r

      .. code-block:: r

          library(fastdfe)
          fd <- load_fastdfe()

          fd$logger$setLevel("DEBUG")

The progress bars are disabled as follows:

.. tab-set::
   :sync-group: language
   :class: code-tabs

   .. tab-item:: :fab:`python` Python
      :sync: python

      .. code-block:: python

          fd.Settings.disable_pbar = True

   .. tab-item:: :fab:`r-project` R
      :sync: r

      .. code-block:: r

          fd$Settings$disable_pbar <- TRUE

Debugging
---------

When an unexpected error occurs, disabling parallelization yields a more descriptive stack trace (see ``parallelize`` in :class:`~fastdfe.base_inference.BaseInference` and :class:`~fastdfe.joint_inference.JointInference`).

Seeding
-------

``fastdfe`` is seeded by default to ensure reproducibility (see ``seed`` in :class:`~fastdfe.base_inference.BaseInference` and :class:`~sfsutils.parser.Parser`). Randomness is required for various computational tasks, such as bootstrapping, choosing initial values for different optimization runs, and taking subsamples during VCF parsing.

Object-oriented design
----------------------

``fastdfe`` follows an object-oriented design. Objects such as :class:`~fastdfe.base_inference.BaseInference` take their configuration on construction, and expose their results through properties and methods.
