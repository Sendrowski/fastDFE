"""
Backward-compatibility shim: this module moved to :mod:`sfsutils.annotation`.

Importing from ``fastdfe.annotation`` still works, and jsonpickle can restore objects that
were serialized with the old ``fastdfe.annotation.*`` paths, because the classes are the very
same objects re-exported from ``sfsutils.annotation``.
"""

from sfsutils.annotation import *  # noqa: F401,F403
from sfsutils import annotation as _module


def __getattr__(name):
    # delegate any remaining attribute (incl. private classes referenced in old
    # serialized data) to the real sfsutils module
    return getattr(_module, name)
