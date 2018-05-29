#include "Array.hpp"

int main()
{
    const int itot = 10;
    Array_1d a(itot);
    Array_1d b(itot);

    a = 3.;
    b = 5.;

    a.print();

    a = a*(a+b);
    a.print();

    a(0, 3) = 666.;
    a.print();

    b(7,10) = a(0,3);
    b.print();
}
