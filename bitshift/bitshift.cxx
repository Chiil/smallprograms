#include <random>
#include <vector>
#include <iostream>

namespace
{
    const int fieldsize = 12;
}

int main()
{
    int min = 0;
    int max = 1+2+4+8; // center, u, v, w

    std::random_device rd;     // only used once to initialise (seed) engine
    std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
    std::uniform_int_distribution<int> uni(min,max); // guaranteed unbiased

    std::vector<int> field;
    for (int i=0; i<fieldsize; ++i)
        field.push_back(uni(rng));

    std::cout << "field: " << std::endl;
    for (int i : field)
        std::cout << i << ", "
                  << ((i & 0x1) >> 0) << ", " 
                  << ((i & 0x2) >> 1) << ", " 
                  << ((i & 0x4) >> 2) << ", " 
                  << ((i & 0x8) >> 3) << ", " 
                  << std::endl;
    std::cout << std::endl;

    // This function is expensive at generation, but fast at runtime
    std::vector<short> mask(fieldsize/4);
    for (int i=0; i<fieldsize/4; ++i)
    {
        const int index = 4*i;
        mask[i] = (field[index  ] & 0x1)
                + (field[index+1] & 0x2)
                + (field[index+2] & 0x3)
                + (field[index+3] & 0x4);
    }

    return 0;
}
