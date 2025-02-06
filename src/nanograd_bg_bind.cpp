// Standard Library Dependencies

// Local Dependencies
#include "engine.h"
#include "nn.h"

// External Dependencies
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Create a python trampoline class for Module
class PyModule : public Module {
    using Module::Module;

public:
    void zero_grad() override { PYBIND11_OVERRIDE(void, Module, zero_grad,); }

    std::vector<Value> get_parameters() override {
        PYBIND11_OVERRIDE(std::vector<Value>, Module, get_parameters,);
    }
};

void add_engine(py::module_ &m) {
    const py::module_ engine =
            m.def_submodule("engine", "Automatic differentiation engine");

    // Add Value class to submodule
    py::class_<Value>(engine, "Value")
            .def(py::init<const double>())
            .def("__repr__", &Value::as_string, R"pbdoc(
            Create a string representation of the Value. 

            Returns:
                str: A representation of the Value.
            )pbdoc")
            .def_property("grad", &Value::get_grad, &Value::set_grad, R"pbdoc(
                float: The gradient associated with the Value
            )pbdoc")
            .def_property("data", &Value::get_data, &Value::set_data, R"pbdoc(
                float: Data wrapped by the Value.
            )pbdoc")
            .def("zero_grad", &Value::zero_grad, R"pbdoc(
                Set the value of grad to 0.0
            )pbdoc")
            .def("backwards", &Value::backwards, R"pbdoc(
                Compute the gradients of a Value. 

                Uses backpropagation to calculate the dertivative of this Value with
                respect to any Vaues which were used to generate it. 

                Examples:
                    >>> # Create some Values
                    >>> x = Value(3)
                    >>> y = Value(4)
                    >>> # Perform a computation with the Values
                    >>> z = x*y
                    >>> # Calculate the gradient of z wrt x and y
                    >>> z.backwards()
                    >>> print(x.grad)
                    4
                    >>> print(y.grad)
                    3
            )pbdoc")
            .def(py::self + py::self)
            .def(double() + py::self)
            .def(py::self + double())
            .def(py::self - py::self)
            .def(double() - py::self)
            .def(py::self - double())
            .def(py::self * py::self)
            .def(double() * py::self)
            .def(py::self * double())
            .def(py::self / py::self)
            .def(double() / py::self)
            .def(py::self / double())
            .def("__pow__", [](const Value &a, const double b) { return a.pow(b); })
            .def("__neg__", [](const Value &a) { return -a; })
            .def("relu", &Value::relu, R"pbdoc(
                Calculate the output of a ReLU on the Value.

                Returns:
                    Value: The output of the ReLU operation wrapped into a Value.
            )pbdoc")
            .doc() = R"pbdoc(
                A wrapped float which can be used in computing gradients.

                Args:
                   data (float): Data to wrap in the Value.
            )pbdoc";
}

void add_nn(py::module_ &m) {
    const auto nn = m.def_submodule("nn", "Neural Network Classes");
    // Add Module class to the submodule
    py::class_<Module, PyModule>(nn, "Module")
            .def(py::init<>())
            .def("zero_grad", &Module::zero_grad, R"pbdoc(
                Zero the gradients of all parameters associated with Module.
            )pbdoc")
            .def("get_parameters", &Module::get_parameters, R"pbdoc(
                Get a list of all parameters associated with Module. 
            )pbdoc")
            .doc() = R"pbdoc(
                Base class for neural network classes.
            )pbdoc";

    // Add the Neuron class to the submodule
    py::class_<Neuron>(nn, "Neuron")
            .def(py::init<int, bool>())
            .def("get_parameters", &Neuron::get_parameters, R"pbdoc(
                Get a list of all parameters associated with the Neuron. 

                Returns:
                    list[Value]: The parameters of the Neuron, specifically the weights and bias.
            )pbdoc")
            .def("zero_grad", &Neuron::zero_grad, R"pbdoc(
                Set the gradient of all Neuron parameters to 0.
            )pbdoc")
            .def("__call__", &Neuron::operator())
            .doc() = R"pbdoc(
                A single neuron, with randomly initialized weights and bias, as well as an activation function.

                Args:
                    nin (int): Number of inputs to the Neuron.
                    nonlinear (bool): Whether the activation function should be non-linear 
                        (fed through a ReLU).
            )pbdoc";

    // Add the Layer class to the submodule
    py::class_<Layer>(nn, "Layer")
            .def(py::init<int, int, bool>())
            .def("get_parameters", &Layer::get_parameters, R"pbdoc(
                Get a list of all parameters associated with the Layer. 

                Returns:
                    list[Value]: The parameters of the Layer, specifically 
                        the weights and biases of all associated Neurons.
            )pbdoc")
            .def("zero_grad", &Layer::zero_grad, R"pbdoc(
                Set the gradient of all Layer parameters to 0.
            )pbdoc")
            .def("__call__", &Layer::operator())
            .doc() = R"pbdoc(
                A Layer of Neurons in a neural network.

                Args:
                    nin (int): Number of inputs to the Layer.
                    nouts (int): Number of outputs from the Layer.
                    nonlinear (bool): Whether the output of the Layer should be nonlinear 
                        (fed through a ReLU).
            )pbdoc";

    // Add the multilayer perceptron class to the submodule
    py::class_<MultiLayerPerceptron>(nn, "MultiLayerPerceptron")
            .def(py::init<int, std::vector<int> >())
            .def("get_parameters", &MultiLayerPerceptron::get_parameters, R"pbdoc(
                Get a list of all parameters associated with the MultiLayerPerceptron. 

                Returns:
                    list[Value]: The parameters of the MultiLayerPerceptron, specifically 
                        the weights and biases of all associated Neurons in each Layer.
            )pbdoc")
            .def("zero_grad", &MultiLayerPerceptron::zero_grad, R"pbdoc(
                Set the gradient of all MultiLayerPerceptron parameters to 0.
            )pbdoc")
            .def("__call__", &MultiLayerPerceptron::operator())
            .doc() = R"pbdoc(
                A Multi-Layer Perceptron.

                Args:
                    nin (int): Number of inputs to the MultiLayerPerceptron.
                    nouts (list[int]): Sizes of the Layers in the MultiLayerPerceptron, 
                        the last of which is the number of outputs form the 
                        MultiLayerPerceptron.
            )pbdoc";
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Small scalar valued automatic differentiation library";

    // Add the engine submodule
    add_engine(m);

    // Add the nn submodule
    add_nn(m);
}
