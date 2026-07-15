"""
Backward-compatibility shim: this module moved to :mod:`sfsutils.io_handlers`.

Importing from ``fastdfe.io_handlers`` still works, and jsonpickle can restore objects that
were serialized with the old ``fastdfe.io_handlers.*`` paths, because the classes are the very
same objects re-exported from ``sfsutils.io_handlers``.
"""

from sfsutils.io_handlers import *  # noqa: F401,F403
from sfsutils import io_handlers as _module


def __getattr__(name):
    # delegate any remaining attribute (incl. private classes referenced in old
    # serialized data) to the real sfsutils module
    return getattr(_module, name)
