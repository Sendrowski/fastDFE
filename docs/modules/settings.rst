.. _modules.settings:

Settings
--------

:class:`~sfsutils.settings.Settings` holds package-wide defaults, set on the class itself.
``fastdfe`` re-exports the very same class, so ``fastdfe.Settings`` and ``sfsutils.settings.Settings``
are one and the same object::

    import fastdfe

    fastdfe.Settings.disable_pbar = True

.. autoclass:: sfsutils.settings.Settings
   :members:
