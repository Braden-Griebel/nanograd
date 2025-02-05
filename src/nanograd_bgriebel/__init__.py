from importlib.metadata import version

__author__ = "Braden Griebel"
__version__ = version("nanograd_bgriebel")
__all__ = ["engine", "nn", "Value", "Module", "Neuron", "Layer", "MultiLayerPerceptron"]

# Package Imports
from nanograd_bgriebel._core import engine, nn
from nanograd_bgriebel._core.engine import Value
from nanograd_bgriebel._core.nn import Module, Neuron, Layer, MultiLayerPerceptron
