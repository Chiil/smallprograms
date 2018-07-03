#include <exception>
#include <iostream>

#include "netcdf_interface.h"

int main()
{
    try
    {
        Netcdf_file nc_file("test.nc", Netcdf_mode::Create);
        Netcdf_group nc_group(nc_file, "test_group");

        nc_group.add_dimension("time");
        nc_group.add_dimension("zh", 33);

        auto nc_time = nc_group.add_variable("time", {"time"});
        auto nc_uflux = nc_group.add_variable("uflux", {"time", "zh"});
        auto nc_vflux = nc_group.add_variable("vflux", {"time", "zh"});

        std::vector<double> uflux(33);
        std::vector<double> vflux(33);

        std::fill(uflux.begin(), uflux.end(), 1.);
        std::fill(vflux.begin(), vflux.end(), 2.);

        double time = 3600;
        nc_time.insert(time, {0});
        nc_uflux.insert(uflux, {0,0});
        nc_vflux.insert(vflux, {0,0});

        std::fill(uflux.begin(), uflux.end(), 11.);
        std::fill(vflux.begin(), vflux.end(), 22.);

        time += 3600;
        nc_time.insert(time, {1});
        nc_uflux.insert(uflux, {1,0});
        nc_vflux.insert(vflux, {1,0});
    }

    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
