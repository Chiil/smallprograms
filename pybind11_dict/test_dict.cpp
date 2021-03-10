#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

py::dict read_dict(const py::dict input_dict)
{
    int itot;
    int ktot;

    itot = input_dict["itot"].cast<int>();
    ktot = input_dict["ktot"].cast<int>();

    std::vector<double> v(itot*ktot);
    for (int i=0; i<itot*ktot; ++i)
        v[i] = i;

    py::dict output_dict;
    output_dict["heating_rate"] = py::array({ktot, itot}, v.data());

    return output_dict;
}

PYBIND11_MODULE(test_dict, m)
{
    m.doc() = R"pbdoc(
        Reading a Python dictionary in C++
        -----------------------

        .. currentmodule:: read_dict

        .. autosummary::
           :toctree: _generate

           read_dict
    )pbdoc";

    m.def("read_dict", &read_dict, py::return_value_policy::move, R"pbdoc(
        Read a dictionary in C++

        Some other explanation about the read function.
    )pbdoc");

    m.attr("__version__") = "v0.0.1";
}
