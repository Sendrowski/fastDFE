.. _installation_r:

Installation
============

R
^
To install the `fastdfe` package in R, we first install and load the reticulate package in R, which allows us to interface with Python. We then use reticulate's :meth:`py_install` function to install the `fastdfe` Python package, and subsequently import it into our `R environment for further use.

.. code-block:: r

   install.packages("reticulate")

   library(reticulate)

   py_install("fastdfe")

   fastdfe <- import("fastdfe")