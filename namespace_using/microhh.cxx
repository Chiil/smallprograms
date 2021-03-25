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

        for (int i=0; i<Grid::grid_data.itot; ++i)
        {
            std::cout << i << ": " << Fields::ap.at("u")[i] 
                           << ", " << Fields::ap.at("v")[i] 
                           << ", " << Fields::ap.at("w")[i]
                           << std::endl;
        }
    }
    catch (std::exception& e)
    {
        std::cout << "EXCEPTION CAUGHT: " << e.what() << std::endl;
    }
} 
