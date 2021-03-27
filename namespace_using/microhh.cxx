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

        Grid::init();
        Fields::init();
        Advec::init(swadvec);

        Advec::exec();

        for (int i=0; i<Grid::itot; ++i)
        {
            std::cout << i << ": " << Fields::all_3d.at("u")[i] 
                           << ", " << Fields::all_3d.at("v")[i] 
                           << ", " << Fields::all_3d.at("w")[i]
                           << std::endl;
        }
    }
    catch (std::exception& e)
    {
        std::cout << "EXCEPTION CAUGHT: " << e.what() << std::endl;
    }
} 
