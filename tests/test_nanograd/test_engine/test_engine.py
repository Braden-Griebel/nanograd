# Standard Library Imports

# External Imports
import pytest
import torch

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

    def test_complex_calc(self):
        a = ng.Value(-4.0)
        b = ng.Value(2.0)
        c = a + b
        d = a * b + b**3
        c += c + 1
        c += 1 + c + (-a)
        d += d * 2 + (b + a).relu()
        d += 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g += 10.0 / f
        g.backwards()
        amg, bmg, gmg = a, b, g

        a = torch.Tensor([-4.0]).double()
        b = torch.Tensor([2.0]).double()
        a.requires_grad = True
        b.requires_grad = True
        c = a + b
        d = a * b + b**3
        c = c + c + 1
        c = c + 1 + c + (-a)
        d = d + d * 2 + (b + a).relu()
        d = d + 3 * d + (b - a).relu()
        e = c - d
        f = e**2
        g = f / 2.0
        g = g + 10.0 / f
        g.backward()
        apt, bpt, gpt = a, b, g

        tol = 1e-6
        # forward pass went well
        assert abs(gmg.data - gpt.data.item()) < tol
        # backward pass went well
        assert abs(amg.grad - apt.grad.item()) < tol
        assert abs(bmg.grad - bpt.grad.item()) < tol
