"""
Backward-compatibility shim: this module moved to :mod:`sfsutils.settings`.

Importing from ``fastdfe.settings`` still works, and jsonpickle can restore objects that
were serialized with the old ``fastdfe.settings.*`` paths, because the classes are the very
same objects re-exported from ``sfsutils.settings``.
"""

from sfsutils.settings import *  # noqa: F401,F403
from sfsutils import settings as _module


def __getattr__(name):
    # delegate any remaining attribute (incl. private classes referenced in old
    # serialized data) to the real sfsutils module
    return getattr(_module, name)
