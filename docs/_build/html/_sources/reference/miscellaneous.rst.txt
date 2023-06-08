.. _reference.miscellaneous:

Miscellaneous
=============

Logging
-------

fastDFE uses the standard Python :mod:`logging` module for logging. By default, fastDFE logs to the console at the ``INFO`` level. You can change the logging level, to ``DEBUG`` for example, **after** importing fastdfe by doing as follows::

    import fastdfe, logging

    logging.getLogger('fastdfe').setLevel(logging.DEBUG)

You can also disable the progress bar like this::

    fastdfe.disable_pbar = True


Debugging
---------

If you encounter an unexpected error, you can disable parallelization to obtain a more descriptive stack trace (see ``parallelize`` in :class:`~fastdfe.base_inference.BaseInference`).

    import fastdfe

    fastdfe.parallelize = False

    # ... run your code here ...

    fastdfe.parallelize = True

Seeding
-------

fastDFE is seeded by default to ensure reproducibility (see ``seed`` in :class:`~fastdfe.base_inference.BaseInference`, :class:`~fastdfe.parser.Parser`, etc.). Randomness is required when bootstrapping, choosing initial values for different optimization runs, and when parsing the SFS from VCF files which requires taking random subsamples.