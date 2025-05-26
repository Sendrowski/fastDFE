.. _reference.python.installation:

Installation
============

PyPi
^^^^
To install the ``fastdfe`` package, you can use pip:

.. code-block:: bash

   pip install fastdfe

``fastdfe`` is compatible with Python 3.10 through 3.12.

.. note::

   Support for VCF input (e.g., reading ``.vcf.gz`` files) in ``fastdfe`` requires the optional ``cyvcf2`` dependency.
   To enable this functionality, install with the ``vcf`` extra:

   .. code-block:: bash

      pip install fastdfe[vcf]

Conda
^^^^^
To avoid potential conflicts with other packages, it is recommended to install ``fastdfe`` in an isolated environment. The easiest way to do this is to use `conda` (or `mamba`):

To do this, run:

.. code-block:: bash

    mamba create -n fastdfe 'python>=3.10,<3.13' pip
    mamba activate fastdfe
    pip install fastdfe

Alternatively, create a new file called ``environment.yml`` with the following content:

.. code-block:: yaml

  name: fastdfe
  channels:
    - defaults
  dependencies:
    - python>=3.10,<3.13
    - pip
    - pip:
        - fastdfe

Then run the following command to create the environment:

.. code-block:: bash

  mamba env create -f environment.yml

Activate the newly created environment:

.. code-block:: bash

  mamba activate fastdfe

You are now ready to use the ``fastdfe`` package within the isolated conda environment.

.. code-block:: python

    import fastdfe as fd