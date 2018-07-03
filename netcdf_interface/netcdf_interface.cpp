#include <iostream>
#include <map>
#include <vector>
#include <netcdf.h>

#include "netcdf_interface.h"

namespace
{
    void nc_throw(const int return_value)
    {
        std::string error(nc_strerror(return_value));
        throw std::runtime_error(error);
    }

    void nc_check(const int return_value)
    {
        if (return_value != NC_NOERR)
            nc_throw(return_value);
    }
}

Netcdf_file::Netcdf_file(const std::string& name, Netcdf_mode mode)
{
    if (mode == Netcdf_mode::Create)
        nc_check( nc_create(name.c_str(), NC_NOCLOBBER | NC_NETCDF4, &ncid) );
    else if (mode == Netcdf_mode::Write)
        nc_check( nc_open(name.c_str(), NC_WRITE | NC_NETCDF4, &ncid) );
    else if (mode == Netcdf_mode::Read)
        nc_check( nc_open(name.c_str(), NC_NOWRITE | NC_NETCDF4, &ncid) );

    nc_check( nc_enddef(ncid) );
}

Netcdf_file::~Netcdf_file()
{
    nc_check( nc_close(ncid) );
}

void Netcdf_file::add_dimension(const std::string& dim_name, const size_t dim_size)
{
    nc_check( nc_redef(ncid) );

    int dim_id;
    int def_out = nc_def_dim(ncid, dim_name.c_str(), dim_size, &dim_id);

    // Dimension is written or already exists.
    if (def_out == NC_NOERR)
        dims.emplace(dim_name, dim_id);
    else if (def_out == NC_ENAMEINUSE)
    {}
    // Error.
    else
        nc_throw(def_out);

    nc_check( nc_enddef(ncid) );
}

Netcdf_variable Netcdf_file::add_variable(
        const std::string& var_name,
        const std::vector<std::string> dim_names)
{
    nc_check ( nc_redef(ncid) );

    int ndims = dim_names.size();
    std::vector<int> dim_ids;
    for (const std::string& dim_name : dim_names)
        dim_ids.push_back(dims.at(dim_name));

    int var_id;
    nc_check( nc_def_var(ncid, var_name.c_str(), NC_DOUBLE, ndims, dim_ids.data(), &var_id) );
    nc_check( nc_enddef(ncid) );

    std::vector<size_t> dim_sizes;
    for (const int dim_id : dim_ids)
    {
        size_t dim_len;
        nc_check( nc_inq_dimlen(ncid, dim_id, &dim_len) );

        if (dim_len == NC_UNLIMITED)
            dim_sizes.push_back(1);
        else
            dim_sizes.push_back(dim_len);
    }

    return Netcdf_variable(*this, var_id, dim_sizes);
}

void Netcdf_file::insert(
        const std::vector<double>& values,
        const int var_id,
        const std::vector<size_t>& i_start,
        const std::vector<size_t>& i_count)
{
    nc_check( nc_put_vara_double(ncid, var_id, i_start.data(), i_count.data(), values.data()) );
}

void Netcdf_file::insert(
        const double value,
        const int var_id,
        const std::vector<size_t>& i_start,
        const std::vector<size_t>& i_count)
{
    nc_check( nc_put_vara_double(ncid, var_id, i_start.data(), i_count.data(), &value) );
}

Netcdf_variable::Netcdf_variable(Netcdf_file& nc_file, const int var_id, const std::vector<size_t>& dim_sizes) :
    nc_file(nc_file), var_id(var_id), dim_sizes(dim_sizes)
{}

void Netcdf_variable::insert(const std::vector<double>& values, const std::vector<size_t> i_start)
{
    // CvH: Add proper size checking.
    nc_file.insert(values, var_id, i_start, dim_sizes);
}

void Netcdf_variable::insert(const double value, const std::vector<size_t> i_start)
{
    // CvH: Add proper size checking.
    nc_file.insert(value, var_id, i_start, dim_sizes);
}
