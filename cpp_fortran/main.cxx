#include <iostream>
#include <vector>

namespace
{
    extern "C" void square(double*, int*);
}

int main()
{
    std::vector<double> a = {1., 2., 3.};

    int size = a.size();
    square(a.data(), &size);

    for (const double d : a)
        std::cout << d << "\n";

    return 0;
}
