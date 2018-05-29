#include "Array.hpp"

int main()
{
    const int itot = 10;
    Array_1d a(itot);
    Array_1d b(itot);
    Array_1d c(itot);

    a = 3.;
    b = 5.;
    c = 0.;

    a.print();

    a = a*(a+b);
    a.print();

    a(0, 3) = 666.;
    a.print();

    b(7,10) = a(0,3);
    b.print();

    c(4,7) = a(1,4)*a(2,5)*b(3,6);
    c.print();
}
