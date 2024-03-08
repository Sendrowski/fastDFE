.. _reference.r.miscellaneous:

Miscellaneous
=============

Logging
-------

fastDFE uses the standard Python :mod:`logging` module for logging. By default, fastDFE logs to the console at the ``INFO`` level. You can change the logging level, to for example ``DEBUG`` as follows

.. code-block:: r

   fastdfe <- load_fastdfe()

   fastdfe$logger$setLevel("DEBUG")

You can also disable the progress bar like this

.. code-block:: r

   fastdfe$Settings$disable_pbar <- TRUE


Debugging
---------

If you encounter an unexpected error, you might want to disable parallelization to obtain a more descriptive stack trace (see ``parallelize`` in :class:`~fastdfe.base_inference.BaseInference`, :class:`~fastdfe.joint_inference.JointInference`, etc.). Unfortunately, this being a Python package, the stack trace will be in Python.

Seeding
-------

fastDFE is seeded by default to ensure reproducibility (see ``seed`` in :class:`~fastdfe.base_inference.BaseInference`, :class:`~fastdfe.parser.Parser`, etc.). Randomness is required for various computational tasks, such as bootstrapping, choosing initial values for different optimization runs, and taking subsamples during VCF parsing.

Object-Oriented Design
----------------------
The fastDFE Python library is implemented using an object-oriented design which carries over to the R interface. This paradigm may be less familiar to R users. To aid comprehension, our documentation often demonstrates how methods are called on the class itself rather than on an instance. For example:

.. code-block:: r

   inference <- fastdfe$BaseInference(...)

   fastdfe$BaseInference$run(inference)

In this style, the :meth:`~fastdfe.base_inference.BaseInference.run` method is called on the :class:`~fastdfe.base_inference.BaseInference` class, and its instance is explicitly passed as an argument.

Alternatively, :meth:`~fastdfe.base_inference.BaseInference.run` can also be called directly on an instance of a class:

.. code-block:: r

   inference$run()

This is more in line with traditional object-oriented programming. Both approaches are functionally equivalent, but the first method is more verbose.