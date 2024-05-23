.. _reference.python.installation:

Installation
============

PyPi
^^^^
To install the ``fastdfe``, you can use pip:

.. code-block:: bash

   pip install fastdfe

fastdfe is compatible with Python 3.10, 3.11 and 3.12.

Conda
^^^^^
However, to avoid potential conflicts with other packages, it is recommended to install ``fastdfe`` in an isolated environment. The easiest way to do this is to use `conda` (or `mamba`):

To do this, you can run

.. code-block:: bash

    mamba create -n fastdfe 'python>=3.10,<3.13' pip
    mamba activate fastdfe
    pip install fastdfe

Alternative, create a new file called ``environment.yml`` with the following content:

.. code-block:: yaml

  name: fastdfe
  channels:
    - defaults
  dependencies:
    - python>=3.10,<3.13
    - pip
    - pip:
        - fastdfe

Run the following command to create a new `conda` environment using the ``environment.yml`` file:

.. code-block:: bash

  mamba env create -f environment.yml

Activate the newly created conda environment:

.. code-block:: bash

  mamba activate fastdfe

You are now ready to use the ``fastdfe`` package within the isolated conda environment.

.. code-block:: python

    import fastdfe as fd