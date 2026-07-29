.. _reference.python.installation:

Installation
============

PyPI
^^^^
To install the ``fastdfe`` package via pip:

.. code-block:: bash

   pip install fastdfe

``fastdfe`` is compatible with Python 3.11 through 3.13.

.. note::

   The SFS parsing is provided by the :mod:`sfsutils` package, whose input backends are optional
   extras that ``fastdfe`` exposes under the same names: ``vcf`` (the :mod:`cyvcf2 <cyvcf2.cyvcf2>`
   dependency, for VCF files), ``zarr`` (the :mod:`zarr` dependency, for VCF-Zarr stores) and ``arg``
   (the :mod:`tskit` dependency, for tree sequences / ARGs). Install the ones you need, for example
   all of them:

   .. code-block:: bash

      pip install fastdfe[vcf,zarr,arg]

Conda
^^^^^
As of version 1.1.12, ``fastdfe`` is also available on **conda-forge**. To install it:

.. code-block:: bash

   mamba create -n fastdfe -c conda-forge fastdfe
   mamba activate fastdfe

.. note::

   The optional input backends are not pulled in automatically via conda. :mod:`zarr` and
   :mod:`tskit` are on **conda-forge**, while ``cyvcf2`` (for VCF handling) is on **bioconda**,
   so add both channels:

   .. code-block:: bash

      mamba create -n fastdfe -c conda-forge -c bioconda fastdfe cyvcf2 zarr tskit

Alternatively, to ensure reproducibility, you can create a file ``environment.yml``:

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

Then run the following commands to create and activate the environment:

.. code-block:: bash

  mamba env create -f environment.yml
  mamba activate fastdfe

You are now ready to use ``fastdfe``:

.. code-block:: python

    import fastdfe as fd