.. _reference.r.installation:

Installation
============

To install the ``fastdfe`` package in R, execute the following command:

.. code-block:: r

   devtools::install_github("Sendrowski/fastDFE")

Once the installation is successfully completed, initiate the package within your R session using:

.. code-block:: r

   library(fastdfe)

The ``fastdfe`` R package serves as a wrapper around the Python library but re-implements visualization through ggplot2. Loading the R package declares the Python requirement, which reticulate resolves into a suitable environment the first time the module is loaded:

.. code-block:: r

   fd <- load_fastdfe()

``fastdfe`` is compatible with Python 3.11 through 3.13.

.. note::

   The input backends are optional extras: ``vcf`` for VCF files, ``zarr`` for VCF-Zarr stores and
   ``arg`` for tree sequences. Only ``vcf`` is declared by default. Additional backends are declared by calling
   ``install_fastdfe()`` before the module is loaded:

   .. code-block:: r

      install_fastdfe(extras = c("vcf", "zarr", "arg"))
      fd <- load_fastdfe()

To use an existing Python installation instead, follow the `Python installation guide <../Python/installation.html>`_ and select the environment before loading the module:

.. code-block:: r

   reticulate::use_condaenv("~/miniforge3/envs/fastdfe", required = TRUE)
   fd <- load_fastdfe()

See the R package documentation for more information on the available functions.