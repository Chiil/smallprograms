#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>

namespace py = pybind11;

class Array
{
    public:
        Array(const int itot, const int ktot) : v_(itot*ktot)
        {
            std::cout << "Constructing Array..." << std::endl;
            for (int n=0; n<itot*ktot; ++n)
                v_[n] = n;
        }

        ~Array()
        { std::cout << "Destructing Array..." << std::endl; }

        double* data() { return v_.data(); }

        std::vector<double>& v() { return v_; }

    private:
        std::vector<double> v_;
};

std::unique_ptr<Array> field;

py::dict read_dict(const py::dict input_dict)
{
    int itot;
    int ktot;

    itot = input_dict["itot"].cast<int>();
    ktot = input_dict["ktot"].cast<int>();

    field = std::make_unique<Array>(itot, ktot);

    py::dict output_dict;
    output_dict["field"] = py::array_t<double>({ktot, itot}, field->data(), py::none());

    return output_dict;
}

void print_data()
{
    for (double d : field->v())
        std::cout << d << ", ";
    std::cout << std::endl;
}

PYBIND11_MODULE(test_dict, m)
{
    m.doc() = "Creating an output dictionary based on an input dictionary.";

    m.def("read_dict", &read_dict, py::return_value_policy::reference, "Read a dictionary in C++ and return numpy output");
    m.def("print_data", &print_data, "Print the output array on the C++ side");

    m.attr("__version__") = "v0.0.1";
}
