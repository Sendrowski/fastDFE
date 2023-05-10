import logging
import sys, os
from pathlib import Path

from dotenv import load_dotenv

# load environment variables
load_dotenv()

# set the logging level
logging.getLogger('fastdfe').setLevel(int(os.getenv('TESTING_LOG_LEVEL', logging.INFO)))


def prioritize_installed_packages():
    """
    This function prioritizes installed packages over local packages.
    """
    # Get the current working directory
    cwd = str(Path().resolve())

    # Check if the current working directory is in sys.path
    if cwd in sys.path:
        # Remove the current working directory from sys.path
        sys.path.remove(cwd)
        # Append the current working directory to the end of sys.path
        sys.path.append(cwd)
