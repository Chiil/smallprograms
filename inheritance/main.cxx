#include "base.h"
#include "derived.h"

int main()
{
    Base* base;
    base = new Base();

    Base* base2;
    base2 = new Derived();

    return 0;
}
