#include "Array.hpp"

int main()
{
    const int itot = 10;
    Array_1d a(itot);
    Array_1d b(itot);

    a.print();

    a = a + b;

    a.print();
}
