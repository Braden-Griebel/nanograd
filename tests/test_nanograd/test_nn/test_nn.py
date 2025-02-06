# Standard Library Imports

# External Imports
import pytest

# Local Imports
import nanograd_bgriebel as ng


class TestNeuron:
    def test_creation(self):
        test_neuron = ng.Neuron(5, True)
        assert len(test_neuron.get_parameters()) == 6

    def test_calling(self):
        test_neuron = ng.Neuron(5, True)
        input_ = [ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1)]
        output = test_neuron(input_)
        assert isinstance(output, ng.Value)
        assert output.data >= 0

    def test_grads(self):
        test_neuron = ng.Neuron(5, True)
        input_ = [ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1)]
        output = test_neuron(input_)
        output.backwards()

    def test_zeroing(self):
        test_neuron = ng.Neuron(5, True)
        input_ = [ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1)]
        output = test_neuron(input_)
        output.backwards()
        test_neuron.zero_grad()
        for param in test_neuron.get_parameters():
            assert param.grad == 0


class TestLayer:
    def test_creation(self):
        test_layer = ng.Layer(4,5,True)
        assert len(test_layer.get_parameters()) == 25

    def test_calling(self):
        test_layer = ng.Layer(4,5,True)
        input_ = [ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1)]
        outputs = test_layer(input_)
        for output in outputs:
            assert isinstance(output, ng.Value)
            assert output.data >= 0

    def test_grads(self):
        test_layer = ng.Layer(4,5,True)
        input_ = [ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1)]
        outputs = test_layer(input_)
        for output in outputs:
            output.backwards()

    def test_zeroing(self):
        test_layer = ng.Layer(4,5,True)
        input_ = [ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1)]
        outputs = test_layer(input_)
        for output in outputs:
            output.backwards()
        non_zero_grads = False
        for param in test_layer.get_parameters():
            if abs(param.grad) > 0.0001:
                non_zero_grads = True
        assert non_zero_grads
        test_layer.zero_grad()
        for param in test_layer.get_parameters():
            assert param.grad == 0

class TestMultiLayerPerceptron:
    def test_creation(self):
        test_mlp = ng.MultiLayerPerceptron(4,[5,5,4,3])
        assert len(test_mlp.get_parameters()) == 94

    def test_calling(self):
        test_mlp = ng.MultiLayerPerceptron(4,[5,5,4,3])
        input_ = [ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1)]
        outputs = test_mlp(input_)

    def test_grads(self):
        test_mlp = ng.MultiLayerPerceptron(4,[5,5,4,3])
        input_ = [ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1)]
        outputs = test_mlp(input_)
        for output in outputs:
            output.backwards()

    def test_zeroing(self):
        test_mlp = ng.MultiLayerPerceptron(4,[5,5,4,3])
        input_ = [ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1)]
        outputs = test_mlp(input_)
        for output in outputs:
            output.backwards()
        test_mlp.zero_grad()
        for param in test_mlp.get_parameters():
            assert param.grad == 0




