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
        Netcdf_variable(Netcdf_file&, const int, const std::vector<size_t>&);
        void insert(const std::vector<double>&, const std::vector<size_t>);
        void insert(const double, const std::vector<size_t>);

    private:
        Netcdf_file& nc_file;
        const int var_id;
        const std::vector<size_t> dim_sizes;
};

class Netcdf_file
{
    public:
        Netcdf_file(const std::string&, Netcdf_mode);
        ~Netcdf_file();

        void add_dimension(const std::string&, const size_t dim_size = NC_UNLIMITED);

        Netcdf_variable add_variable(
                const std::string&,
                const std::vector<std::string>);

        void insert(
                const std::vector<double>&,
                const int var_id,
                const std::vector<size_t>&,
                const std::vector<size_t>&);

        void insert(
                const double,
                const int var_id,
                const std::vector<size_t>&,
                const std::vector<size_t>&);

    private:
        int ncid;
        std::map<std::string, int> dims;
};
#endif
