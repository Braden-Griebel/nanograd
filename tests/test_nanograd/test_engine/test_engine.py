# Standard Library Imports

# External Imports
import pytest

# Local imports
import nanograd_bgriebel as ng


class TestEngine:
    @pytest.fixture
    def values(self):
        x = ng.Value(5)
        y = ng.Value(3)
        return x, y

    @pytest.fixture
    def negative_values(self):
        x = ng.Value(-5.0)
        y = ng.Value(-3.0)
        return x, y

    def test_value_creation(self, values):
        x, y = values
        assert x.grad == pytest.approx(0.0), "Gradient wasn't initially 0."
        assert y.grad == pytest.approx(0.0), "Gradient wasn't initially 0."
        assert x.data == pytest.approx(5.0), "Value of x wasn't set to 5"
        assert y.data == pytest.approx(3.0), "Value of y wasn't set to 3"

    def test_addition(self, values):
        x, y = values
        z = x + y
        # Calculate the gradients of x and y
        print("not error yet")
        z.backwards()
        assert z.grad == pytest.approx(1.0), "Gradient of z wasn't 1."
        assert x.grad == pytest.approx(1.0), "Gradient of x+y wrt x wasn't 1."
        assert y.grad == pytest.approx(1.0), "Gradient of x+y wrt y wasn't 1."

    def test_multiplication(self, values):
        x, y = values
        z = x * y
        # Calculate the gradients of x and y
        z.backwards()
        assert z.grad == pytest.approx(1.0), "Gradient wrt to self wasn't 1."
        assert y.grad == pytest.approx(5.0), "Gradient of x*y wrt y when x=5 wasn't 5"
        assert x.grad == pytest.approx(3.0), "Gradient of x*y wrt x when y=3 wasn't 3"

    def test_exponentiation(self, values):
        x, _ = values
        z = x**3
        # Calculate the gradient of x
        z.backwards()
        assert z.grad == pytest.approx(1.0), "Gradient wrt to self wasn't 1."
        assert x.grad == pytest.approx(75.0), "Gradient of x**3 wrt x at 5 wasn't 75"

    def test_relu_positive(self, values):
        x, _ = values
        z = x.relu()
        # Calculate the gradient of x
        z.backwards()
        assert z.grad == pytest.approx(1.0), "Gradient wrt to self wasn't 1."
        assert x.grad == pytest.approx(1.0), "Gradient of ReLU(x) wrt x wasn't 1"

    def test_relu_negative(self, negative_values):
        x, _ = negative_values
        z = x.relu()
        # Calculate the gradient of x
        z.backwards()
        assert z.grad == pytest.approx(1.0), "Gradient wrt to self wasn't 1."
        assert x.grad == pytest.approx(0.0), "Gradient of ReLU(x), x<0, wrt x wasn't 0"
