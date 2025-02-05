# Standard Library Imports

# External Imports
import pytest

# Local Imports
import nanograd_bgriebel as ng


class TestNeuron:
    def test_creation(self):
        test_neuron = ng.Neuron(5, True)
        assert len(test_neuron.get_parameters()) == 6
        assert isinstance(test_neuron, ng.Module)

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
        for param in test_neuron.get_parameters():
            assert param.grad > 0

    def test_zeroing(self):
        test_neuron = ng.Neuron(5, True)
        input_ = [ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1), ng.Value(1)]
        output = test_neuron(input_)
        output.backwards()
        for param in test_neuron.get_parameters():
            assert param.grad > 0
        test_neuron.zero_grad()
        for param in test_neuron.get_parameters():
            assert param.grad == 0
