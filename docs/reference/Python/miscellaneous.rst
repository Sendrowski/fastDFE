.. _reference.python.miscellaneous:

Miscellaneous
=============

Logging
-------

fastDFE uses the standard Python :mod:`logging` module for logging. By default, fastDFE logs to the console at the ``INFO`` level. You can change the logging level, to for example ``DEBUG`` as follows::

    import fastdfe

    fastdfe.logger.setLevel("DEBUG")

You can also disable the progress bar like this::

    fastdfe.Settings.disable_pbar = True


Debugging
---------

If you encounter an unexpected error, you might want to disable parallelization to obtain a more descriptive stack trace (see ``parallelize`` in :class:`~fastdfe.base_inference.BaseInference`, :class:`~fastdfe.joint_inference.JointInference`, etc.).

Seeding
-------

fastDFE is seeded by default to ensure reproducibility (see ``seed`` in :class:`~fastdfe.base_inference.BaseInference`, :class:`~fastdfe.parser.Parser`, etc.). Randomness is required for various computational tasks, such as bootstrapping, choosing initial values for different optimization runs, and taking subsamples during VCF parsing.