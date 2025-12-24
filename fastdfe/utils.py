"""
Utilities for fastDFE.
"""

__author__ = "Janek Sendrowski"
__contact__ = "sendrowski.janek@gmail.com"
__date__ = "2025-12-19"

from abc import ABC
from typing_extensions import Self

import jsonpickle


class Serializable(ABC):
    """
    Mixin class for serializable objects.
    """

    def to_json(self) -> str:
        """
        Serialize object.

        :return: JSON string
        """
        return jsonpickle.encode(self, indent=4, warn=True)

    def to_file(self, file: str):
        """
        Save object to file.

        :param file: File to save to
        """
        with open(file, 'w') as fh:
            fh.write(self.to_json())

    @classmethod
    def from_json(cls, json: str, classes=None) -> Self:
        """
        Unserialize object.

        :param classes: Classes to be used for unserialization
        :param json: JSON string
        """
        return jsonpickle.decode(json, classes=classes)

    @classmethod
    def from_file(cls, file: str, classes=None) -> Self:
        """
        Load object from file.

        :param classes: Classes to be used for unserialization.
        :param file: File to load from
        """
        with open(file, 'r') as fh:
            return cls.from_json(fh.read(), classes)
