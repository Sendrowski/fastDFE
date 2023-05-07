import logging
import sys
from pathlib import Path

logging.getLogger('fastdfe').setLevel(logging.DEBUG)


def prioritize_installed_packages():
    """
    This function prioritizes installed packages over local packages.
    """
    # Get the current working directory
    cwd = str(Path().resolve())

    print('test')

    # Check if the current working directory is in sys.path
    if cwd in sys.path:
        # Remove the current working directory from sys.path
        sys.path.remove(cwd)
        # Append the current working directory to the end of sys.path
        sys.path.append(cwd)


prioritize_installed_packages()

