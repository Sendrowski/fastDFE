"""
Backward-compatibility shim: this module moved to :mod:`sfsutils.spectrum`.

Importing from ``fastdfe.spectrum`` still works, and jsonpickle can restore objects that
were serialized with the old ``fastdfe.spectrum.*`` paths, because the classes are the very
same objects re-exported from ``sfsutils.spectrum``.
"""

from sfsutils.spectrum import *  # noqa: F401,F403
from sfsutils import spectrum as _module


def __getattr__(name):
    # delegate any remaining attribute (incl. private classes referenced in old
    # serialized data) to the real sfsutils module
    return getattr(_module, name)
