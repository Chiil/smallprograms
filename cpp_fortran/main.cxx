#include <iostream>
#include <vector>

namespace
{
    extern "C" void square(double*, size_t);
    extern "C" void set_array(int*, int, int);
    extern "C" void increment_int(int*, int);
    extern "C" void increment_double(double*, double);
    extern "C" void reverse_bool(bool*);

    void increment(int& a, int b) { increment_int(&a, b); }
    void increment(double& a, double b) { increment_double(&a, b); }
}

int main()
{
    std::vector<double> a = {1., 2., 3.};

    square(a.data(), a.size());

    for (const double d : a)
        std::cout << d << "\n";

    const int itot = 4;
    const int jtot = 6;
    std::vector<int> b(itot*jtot);

    for (int j=0; j<jtot; ++j)
        for (int i=0; i<itot; ++i)
        {
            const int ij = i + j*itot;
            b[ij] = 10*j + i;
        }

    std::vector<int> c(itot*jtot);

    set_array(c.data(), itot, jtot);

    for (int j=0; j<jtot; ++j)
        for (int i=0; i<itot; ++i)
        {
            const int ij = i + j*itot;
            std::cout << i << "," << j << ": " << b[ij] << "," << c[ij] << "\n";
        }

    int d = 6;
    const int e = 2;

    increment(d, e);

    std::cout << "d = " << d << std::endl;
    std::cout << "e = " << e << std::endl;

    double f = 6;
    const double g = 2;

    increment(f, g);

    std::cout << "f = " << f << std::endl;
    std::cout << "g = " << g << std::endl;

    bool i_am_true = false;
    reverse_bool(&i_am_true);
    std::cout << "i_am_true = " << i_am_true << std::endl;

    return 0;
}
