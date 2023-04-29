.. _reference.logging:

Logging
=======

fastDFE uses the standard Python :mod:`logging` module for logging. By default, fastDFE logs to the console at the ``INFO`` level. To change the logging level, to ``DEBUG`` for example, use the following code::

    import logging

    logging.getLogger('fastdfe').setLevel(logging.DEBUG)

Additionally, you can disable the progress bar like this::

    fastdfe.disable_pbar = True
