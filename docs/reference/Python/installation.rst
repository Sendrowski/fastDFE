.. _reference.python.installation:

Installation
============

PyPI
^^^^
To install the ``fastdfe`` package via pip:

.. code-block:: bash

   pip install fastdfe

``fastdfe`` is compatible with Python 3.10 through 3.12.

.. note::

   As of ``fastdfe`` version 1.1.12, the ``cyvcf2`` dependency, which is required for VCF handling, is optional.
   To enable VCF support, install with the ``vcf`` extra:

   .. code-block:: bash

      pip install fastdfe[vcf]

Conda
^^^^^
As of version 1.1.12, ``fastdfe`` is also available on **conda-forge**. To install it:

.. code-block:: bash

   mamba create -n fastdfe -c conda-forge fastdfe
   mamba activate fastdfe

.. note::

   If you want to use the VCF utilities in ``fastdfe`` via **conda**, you also need to install ``cyvcf2``, which is hosted on **bioconda**.
   Be sure to add the required channels:

   .. code-block:: bash

      mamba create -n fastdfe -c conda-forge -c bioconda fastdfe cyvcf2

Alternatively, to ensure reproducibility, you can create a file ``environment.yml``:

.. code-block:: yaml

  name: fastdfe
  channels:
    - conda-forge
    - bioconda
  dependencies:
    - fastdfe
    - cyvcf2

Then run the following commands to create and activate the environment:

.. code-block:: bash

  mamba env create -f environment.yml
  mamba activate fastdfe

You are now ready to use ``fastdfe``:

.. code-block:: python

    import fastdfe as fd