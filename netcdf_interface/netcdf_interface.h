#ifndef NETCDF_INTERFACE_H
#define NETCDF_INTERFACE_H

#include <iostream>
#include <map>
#include <vector>
#include <netcdf.h>

enum class Netcdf_mode { Create, Read, Write };

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

class Netcdf_file
{
    public:
        Netcdf_file(const std::string& name, Netcdf_mode mode)
        {
            if (mode == Netcdf_mode::Create)
                nc_check( nc_create(name.c_str(), NC_NOCLOBBER | NC_NETCDF4, &ncid) );
            else if (mode == Netcdf_mode::Write)
                nc_check( nc_open(name.c_str(), NC_WRITE | NC_NETCDF4, &ncid) );
            else if (mode == Netcdf_mode::Read)
                nc_check( nc_open(name.c_str(), NC_NOWRITE | NC_NETCDF4, &ncid) );
        }

        ~Netcdf_file()
        {
            nc_check( nc_close(ncid) );
        }

        void add_dimension(const std::string& dim_name, const size_t dim_size = NC_UNLIMITED)
        {
            int dim_id;
            int def_out = nc_def_dim(ncid, dim_name.c_str(), dim_size, &dim_id);

            // Dimension is written or already exists.
            if (def_out == NC_NOERR)
            {
                dims.emplace(dim_name, def_out);
                std::cout << "CvH: Created new dimension: " << dim_name << std::endl;
                return;
            }
            else if (def_out == NC_ENAMEINUSE)
            {
                std::cout << "CvH: Use existing dimension: " << dim_name << std::endl;
                return;
            }
            // Error.
            else
                nc_throw(def_out);
        }

        void add_variable(
                const std::string& var_name,
                const std::vector<std::string> dim_names)
        {
            int ndims = dim_names.size();
            std::vector<int> dim_ids;
            for (const std::string& dim_name : dim_names)
                dim_ids.push_back(dims.at(dim_name));

            int var_id;
            nc_check( nc_def_var(ncid, var_name.c_str(), NC_DOUBLE, ndims, dim_ids.data(), &var_id) );
            vars.emplace(var_name, var_id);
            std::cout << "CvH: Added variable: " << var_name << ", with dims: ";
            for (auto& s : dim_names)
                std::cout << s << " ";
            std::cout << std::endl;
        }

    private:
        int ncid;
        std::map<std::string, int> dims;
        std::map<std::string, int> vars;
};

class Netcdf_time_series
{
    public:
        Netcdf_time_series(
                Netcdf_file& nc_file,
                const std::string name,
                const std::string time_dim_name,
                const size_t dim_size = NC_UNLIMITED) :
            nc_file(nc_file), name(name)
        {
            // Create the time dimension.
            nc_file.add_dimension(time_dim_name, dim_size);

            // Create the variable.
            nc_file.add_variable(name, { time_dim_name } );
        }

    private:
        Netcdf_file& nc_file;
        const std::string name;

};
#endif
