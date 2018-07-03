#include <exception>
#include <iostream>

#include "netcdf_interface.h"

int main()
{
    try
    {
        Netcdf_file nc_file("test.nc", Netcdf_mode::Create);
        nc_file.add_dimension("t");
        nc_file.add_dimension("zh", 33);

        auto nc_uflux = nc_file.add_variable("uflux", {"t", "zh"});
        auto nc_vflux = nc_file.add_variable("vflux", {"t", "zh"});

        std::vector<double> uflux(33);
        std::vector<double> vflux(33);

        std::fill(uflux.begin(), uflux.end(), 1.);
        std::fill(vflux.begin(), vflux.end(), 2.);

        nc_uflux.insert(uflux, {0,0}, {1,33});
        nc_vflux.insert(vflux, {0,0}, {1,33});

        std::fill(uflux.begin(), uflux.end(), 11.);
        std::fill(vflux.begin(), vflux.end(), 22.);

        nc_uflux.insert(uflux, {1,0}, {1,33});
        nc_vflux.insert(vflux, {1,0}, {1,33});
    }

    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
