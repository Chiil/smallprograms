#include <iostream>
#include <exception>
#include "Advec.h"
#include "Fields.h"

int main()
{
    try
    {
        std::string swadvec;
        std::cin >> swadvec;

        Fields::init();
        Advec::init(swadvec);

        Advec::exec();
    }
    catch (std::exception& e)
    {
        std::cout << "EXCEPTION CAUGHT: " << e.what() << std::endl;
    }
} 
