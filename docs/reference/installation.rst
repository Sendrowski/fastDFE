.. _reference.installation:

Installation
============

.. tab-set::
   :sync-group: language
   :class: code-tabs

   .. tab-item:: :fab:`python` Python
      :sync: python

      .. rubric:: PyPI

      ``fastdfe`` can be installed with ``pip``:

      .. code-block:: bash

         pip install fastdfe

      ``fastdfe`` is compatible with Python 3.11 through 3.13.

      The SFS parsing is provided by the :mod:`sfsutils` package, whose input backends are optional extras that ``fastdfe`` exposes under the same names: ``vcf`` (the :mod:`cyvcf2 <cyvcf2.cyvcf2>` dependency, for VCF files), ``zarr`` (the :mod:`zarr` dependency, for VCF-Zarr stores) and ``arg`` (the :mod:`tskit` dependency, for tree sequences). All of them are installed with:

      .. code-block:: bash

         pip install fastdfe[vcf,zarr,arg]

      .. rubric:: Conda

      To avoid potential conflicts with other packages, it is recommended to install ``fastdfe`` in an isolated environment. The easiest way to do this is with ``conda`` or ``mamba``:

      .. code-block:: bash

         mamba create -n fastdfe -c conda-forge fastdfe
         mamba activate fastdfe

      The optional input backends are not installed automatically with conda. :mod:`zarr` and :mod:`tskit` are available on conda-forge, while ``cyvcf2`` is available on bioconda, so both channels are required:

      .. code-block:: bash

         mamba create -n fastdfe -c conda-forge -c bioconda fastdfe cyvcf2 zarr tskit

      Alternatively, for reproducibility, the environment can be defined in a file ``environment.yml``:

      .. code-block:: yaml

        name: fastdfe
        channels:
          - conda-forge
          - bioconda
        dependencies:
          - fastdfe
          - cyvcf2
          - zarr
          - tskit

      The environment is then created and activated with:

      .. code-block:: bash

        mamba env create -f environment.yml
        mamba activate fastdfe

      ``fastdfe`` is then imported with:

      .. code-block:: python

          import fastdfe as fd

   .. tab-item:: :fab:`r-project` R
      :sync: r

      The ``fastdfe`` R package is installed from GitHub with:

      .. code-block:: r

         devtools::install_github("Sendrowski/fastDFE")

      Once the installation has completed, the package is loaded in an R session with:

      .. code-block:: r

         library(fastdfe)

      The ``fastdfe`` R package serves as a wrapper around the Python library, and draws its figures with ``ggplot2``. Loading the R package declares the Python requirement, which ``reticulate`` resolves into a suitable environment the first time the module is loaded:

      .. code-block:: r

         fd <- load_fastdfe()

      ``fastdfe`` is compatible with Python 3.11 through 3.13.

      The input backends are optional extras: ``vcf`` for VCF files, ``zarr`` for VCF-Zarr stores and ``arg`` for tree sequences. Only ``vcf`` is declared by default. Additional backends are declared by calling ``install_fastdfe()`` before the module is loaded:

      .. code-block:: r

         install_fastdfe(extras = c("vcf", "zarr", "arg"))
         fd <- load_fastdfe()

      An existing Python installation can be used instead by installing ``fastdfe`` as described under the Python tab and selecting its environment before loading the module:

      .. code-block:: r

         reticulate::use_condaenv("~/miniforge3/envs/fastdfe", required = TRUE)
         fd <- load_fastdfe()

      The R package documentation describes the available functions in more detail.
