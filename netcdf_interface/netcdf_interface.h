#ifndef NETCDF_INTERFACE_H
#define NETCDF_INTERFACE_H

#include <iostream>
#include <map>
#include <vector>
#include <netcdf.h>

enum class Netcdf_mode { Create, Read, Write };

class Netcdf_file;

class Netcdf_variable
{
    public:
        Netcdf_variable(Netcdf_file& nc_file, const int var_id);
        void insert(std::vector<double>& values, std::vector<size_t> i_start, std::vector<size_t> i_length);

    private:
        Netcdf_file& nc_file;
        const int var_id;
};

class Netcdf_file
{
    public:
        Netcdf_file(const std::string& name, Netcdf_mode mode);
        ~Netcdf_file();

        void add_dimension(const std::string& dim_name, const size_t dim_size = NC_UNLIMITED);

        Netcdf_variable add_variable(
                const std::string& var_name,
                const std::vector<std::string> dim_names);

        void insert(
                std::vector<double>& values,
                const int var_id,
                const std::vector<size_t> i_start,
                const std::vector<size_t> i_length);

    private:
        int ncid;
        std::map<std::string, int> dims;
};
#endif
