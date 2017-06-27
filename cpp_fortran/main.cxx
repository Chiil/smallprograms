#include <iostream>
#include <vector>

extern "C" void square_(double*, int*);

int main()
{
    std::vector<double> a = {1., 2., 3.};

    int size = a.size();
    square_(a.data(), &size);

    for (const double d : a)
        std::cout << d << "\n";

    return 0;
}
