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

        // Read out a block of a 3D variable.
        const size_t read_size = 2;
        std::vector<double> s(read_size*read_size*read_size);
        Netcdf_group nc_group_3d = nc_file.get_group("test_3d");
        nc_group_3d.get_variable(s, "s", {1,1,1}, {read_size,read_size,read_size});

        for (size_t k=0; k<read_size; ++k)
        {
            for (size_t j=0; j<read_size; ++j)
            {
                for (size_t i=0; i<read_size; ++i)
                {
                    const size_t ijk = i + j*read_size + k*read_size*read_size;
                    std::cout << s[ijk] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }

    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
