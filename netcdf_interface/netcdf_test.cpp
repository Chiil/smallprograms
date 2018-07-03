#include "netcdf_interface.h"

int main()
{
    Netcdf_file nc_file("test.nc", Netcdf_mode::Create);
    Netcdf_time_series u_lev1(nc_file, "u_lev1", "t");
    Netcdf_time_series v_lev1(nc_file, "v_lev1", "t");

    return 0;
}
