// Standard Library Dependencies

// Local Dependencies
#include "engine.h"
#include "nn.h"

// External Dependencies
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>


namespace py = pybind11;

void add_engine(py::module_ &m) {
    py::module_ engine = m.def_submodule("engine", "Automatic differentiation engine");

    // Add Value class to submodule
    py::class_<Value>(engine, "Value")
    .def(py::init<const double>())
        .def("__repr__", &Value::as_string)
    .def_property("grad", &Value::get_grad, &Value::set_grad)
    .def_property("data", &Value::get_data, &Value::set_data)
    .def("zero_grad", &Value::zero_grad, R"pbdoc(
        Set the value of grad to 0.0
        )pbdoc")
    .def("backwards", &Value::backwards)
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
    .def("__pow__", [](const Value &a, const double b) {
        return a.pow(b);
    })
    .def("relu", &Value::relu);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Small scalar valued automatic differentiation library";

    // Add the engine submodule
    add_engine(m);

}
