#include <iostream>
#include <vector>

namespace
{
    extern "C" void square(double*, size_t);
    extern "C" void set_array(int*, int, int);
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

    return 0;
}
