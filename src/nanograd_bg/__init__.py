from importlib.metadata import version

__author__ = "Braden Griebel"
__version__ = version("nanograd_bg")
__all__ = ["engine", "Value"]

# Package Imports
from nanograd_bg._core import engine
from nanograd_bg._core.engine import Value
