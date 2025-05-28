.. _reference.r.installation:

Installation
============

To install the ``fastdfe`` package in R, execute the following command:

.. code-block:: r

   devtools::install_github("Sendrowski/fastDFE")

Once the installation is successfully completed, initiate the package within your R session using:

.. code-block:: r

   library(fastdfe)

The ``fastdfe`` R package serves as a wrapper around the Python library but re-implements visualization through ggplot2. Because of this, the Python package must be installed separately. This can be accomplished with:

.. code-block:: r

   install_fastdfe()

``fastdfe`` is compatible with Python 3.10 through 3.12.

.. note::

   As of ``fastdfe`` version 1.1.12, the ``cyvcf2`` dependency, which is required for VCF handling, is optional.
   To enable VCF support, run the following command in R **before** calling ``install_fastdfe()``:

   .. code-block:: r

      reticulate::py_install("fastdfe[vcf]", pip = TRUE)

Alternatively, you can also follow the instructions in the `Python installation guide <../Python/installation.html>`_ to install the Python package.

After installing the Python package, the ``fastdfe`` wrapper module can be loaded into your R environment using the following command:

.. code-block:: r

   fastdfe <- load_fastdfe()

See the R package documentation for more information on the available functions.