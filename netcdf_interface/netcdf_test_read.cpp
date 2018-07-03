#include <exception>
#include <iostream>

#include "netcdf_interface.h"

int main()
{
    try
    {
        Netcdf_file nc_file("test.nc", Netcdf_mode::Read);
        Netcdf_group nc_group = nc_file.get_group("test_group");

        std::vector<double> zh(33);
        nc_group.get_variable(zh, "zh", {0}, {zh.size()});

        for (size_t i=0; i<zh.size(); ++i)
            std::cout << i << " = " << zh[i] << std::endl;

        std::vector<double> vflux(33*2);
        nc_group.get_variable(vflux, "vflux", {0, 0}, {2, zh.size()});

        for (size_t n=0; n<2; ++n)
        {
            std::cout << "STEP: n = " << n << std::endl;
            for (size_t i=0; i<zh.size(); ++i)
            {
                const size_t index = i + n*zh.size();
                std::cout << i << " = " << vflux[index] << std::endl;
            }
        }
    }

    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
