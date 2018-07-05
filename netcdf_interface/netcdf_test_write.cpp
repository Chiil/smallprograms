#include <exception>
#include <iostream>

#include "netcdf_interface.h"

int main()
{
    try
    {
        Netcdf_file nc_file("test.nc", Netcdf_mode::Create);
        Netcdf_group nc_group = nc_file.add_group("test_group");

        nc_group.add_dimension("time");
        nc_group.add_dimension("zh", 33);

        auto nc_time = nc_group.add_variable("time", {"time"});
        auto nc_ustar = nc_group.add_variable("ustar", {"time"});
        auto nc_zh = nc_group.add_variable("zh", {"zh"});
        auto nc_uflux = nc_group.add_variable("uflux", {"time", "zh"});
        auto nc_vflux = nc_group.add_variable("vflux", {"time", "zh"});

        std::vector<double> zh(33);
        std::vector<double> uflux(33);
        std::vector<double> vflux(33);

        for(int i=0; i<33; ++i)
            zh[i] = 2*i;

        nc_zh.insert(zh, {0});

        std::fill(uflux.begin(), uflux.end(), 1.);
        std::fill(vflux.begin(), vflux.end(), 2.);

        double time = 3600;
        double ustar = 0.3;
        nc_time.insert(time, {0});
        nc_ustar.insert(ustar, {0});
        nc_uflux.insert(uflux, {0,0});
        nc_vflux.insert(vflux, {0,0});

        std::fill(uflux.begin(), uflux.end(), 11.);
        std::fill(vflux.begin(), vflux.end(), 22.);

        time += 3600;
        ustar = 0.5;
        nc_time.insert(time, {1});
        nc_time.insert(ustar, {1});
        nc_uflux.insert(uflux, {1,0});
        nc_vflux.insert(vflux, {1,0});

        Netcdf_group nc_group_3d = nc_file.add_group("test_3d");
        nc_group_3d.add_dimension("x", 4);
        nc_group_3d.add_dimension("y", 4);
        nc_group_3d.add_dimension("z", 4);
        auto nc_s = nc_group_3d.add_variable("s", {"z", "y", "x"});

        std::vector<double> s(4*4*4);
        for (int k=0; k<4; ++k)
            for (int j=0; j<4; ++j)
                for (int i=0; i<4; ++i)
                {
                    const int ijk = i + j*4 + k*4*4;
                    s.at(ijk) = (i+1) + (j+1)*10 + (k+1)*100;
                }

        nc_s.insert(s, {0,0,0}, {4,4,4});
    }

    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    return 0;
}
