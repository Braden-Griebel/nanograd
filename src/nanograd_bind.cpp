// Standard Library Dependencies

// Local Dependencies
#include "engine.h"
#include "nn.h"

// External Dependencies
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>


namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Small scalar valued automatic differentiation library";

    // Add the engine submodule
    auto engine = m.def_submodule("engine");
    engine.doc() = "Automatic differentiation engine for nanograd";
    py::class_<Value>(m, "Value")
        .def(py::init<const double>())
        .def("__repr__", &Value::as_string)
    .def_property("grad", &Value::get_grad, &Value::set_grad)
    .def_property("data", &Value::get_data, &Value::set_data)
    .def("zero_grad", &Value::zero_grad, R"pbdoc(
        Set the value of grad to 0.0
        )pbdoc")
    .def("backward", &Value::backwards)
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
    .def("__pow__", [](const Value &a, double b) {
        return a.pow(b);
    })
    .def("relu", &Value::relu);
}
