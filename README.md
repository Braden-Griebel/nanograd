# Nanograd

A small scalar valued automatic differentiation engine written in C++ with bindings in python. Also includes a small neural network api with basic classes like Neuron, Layer, MultiLayerPerceptron for creating neural networks. Based on [micrograd](https://github.com/karpathy/micrograd) from Andrej Karpathy. 

## Installation
Nanograd is available on pypi (the python package index), and so can be installed with uv or pip. It's recommended to create a virtual environment when installing.   

With pip:
```{shell}
# Create a virtual environment
python -m venv .venv # Creates a virtual environment called .venv
# Activate (Linux)
source .venv/bin/activate
# Activate (Windows)
.\.venv\env\scripts\activate
# Then you can install using pip
pip install nanograd-bgriebel
```
With uv:
```{shell}
# Create a virtual environment
uv venv
# Activate (Linux)
source .venv/bin/activate
# Activate (Windows)
.\.venv\env\scripts\activate
# Install using uv
uv pip install nanograd-bgriebel
```

## Usage

Nanograd is built around the Value class, which is essentially a wrapped float which keeps track of calculations it is used in so that the gradients can be calculated. 
  
Creating a Value is easy, it just needs a float as input:

```{python}
import nanograd_bgriebel as ng
my_value = ng.Value(1.0)
```

Then Values can be combined with simple arithmetic operations, and gradients can be calculated.  

```{python}
import nanograd_bgriebel as ng
# Create some Values
x = ng.Value(2.0)
y = ng.Value(3.0)
z = ng.Value(4.0)

# Can add, and multiply the different Values
a = x+y
b = x*z

# And also raise them to different powers
c = x**3

# This can be chained to build up more complex expressions
d = a/(b*c) # equivalent to (x+y) / (x*z*(x**3))

# Then the result of the calculations can be obtained through the data property
print(d.data) # Output: 0.078125

# And the value of derivatives can be found using the backwards function
d.backwards() # Calculate the derivatives of x*z
print(a.grad) # The gradient of d with respect to a
print(b.grad) # The gradient of b with respect to b
```

## Examples

The examples directory contains example notebooks (one for using the automatic differentiation engine
one for using the neural network api). These can be run by installing the requirements in the requirements.txt file
(again, it is suggested to use a virtual environment). 
  
From inside the examples directory:
```{shell}
# Create a virtual environment
python -m venv .venv # Creates a virtual environment called .venv
# Activate (Linux)
source .venv/bin/activate
# Activate (Windows)
.\.venv\env\scripts\activate
# Then you can install using pip
pip install -r ./requirements.txt
```
or

```{shell}
# Create a virtual environment
uv venv
# Activate (Linux)
source .venv/bin/activate
# Activate (Windows)
.\.venv\env\scripts\activate
# Install using uv
uv pip install -r requirements.txt
```
  
The notebooks can then be opened in jupyter lab or jupyter notebooks   

```{shell}
jupyter lab /path/to/jupyter/notebook
```

or 

```{shell}
jupyter notebook jupyter lab /path/to/jupyter/notebook
```