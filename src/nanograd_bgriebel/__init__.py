from importlib.metadata import version

__author__ = "Braden Griebel"
__version__ = version("nanograd_bgriebel")
__all__ = ["engine", "Value"]

# Package Imports
from nanograd_bgriebel._core import engine
from nanograd_bgriebel._core.engine import Value
