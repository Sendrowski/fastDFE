.. _installation:

Installation
============

PyPi
^^^^
To install the `fastdfe`, you can use pip:

.. code-block:: bash

   pip install fastdfe

Conda
^^^^^
However, to avoid potential conflicts with other packages, it is recommended to install `fastdfe` in an isolated environment. The easiest way to do this is to use `conda` (or `mamba`):

Create a new file called ``environment.yml`` with the following content:

   .. code-block:: yaml

      name: fastdfe-env
      channels:
        - defaults
      dependencies:
        - python
        - pip
        - pip:
            - fastdfe

Run the following command to create a new `conda` environment using the ``environment.yml`` file:

   .. code-block:: bash

      conda env create -f environment.yml

Activate the newly created conda environment:

   .. code-block:: bash

      conda activate fastdfe-env

You are now ready to use the ``fastdfe`` package within the isolated conda environment.

   .. code-block:: python

        from fastdfe import ...