#include <iostream>
#include <vector>

namespace
{
    extern "C" void square(double*, size_t);
}

int main()
{
    std::vector<double> a = {1., 2., 3.};

    square(a.data(), a.size());

    for (const double d : a)
        std::cout << d << "\n";

    return 0;
}
