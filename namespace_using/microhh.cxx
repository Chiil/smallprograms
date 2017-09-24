#include "Advec.h"
#include "Fields.h"

int main()
{
    std::string swadvec;
    std::cin >> swadvec;

    Fields::init();
    Advec::init(swadvec);

    Advec::exec();
} 
