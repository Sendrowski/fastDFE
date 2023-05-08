.. _installation:

Installation
------------
To install the `fastdfe`, you can use pip, the standard Python package manager:

.. code-block:: bash

   pip install fastdfe

Although it is possible to use the package directly after this step, it is highly recommended to use conda for a better management of dependencies and to avoid potential conflicts:

.. code-block:: bash

   conda create -n fastdfe-env

Next, activate the newly created conda environment:

.. code-block:: bash

   conda activate fastdfe-env

Now, install pip within the conda environment:

.. code-block:: bash

   conda install pip

Finally, install the `fastdfe` package using the pip installed within the conda environment:

.. code-block:: bash

   pip install fastdfe

You are now ready to use `fastdfe` in your code within the isolated conda environment.
