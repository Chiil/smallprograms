#ifndef NETCDF_INTERFACE_H
#define NETCDF_INTERFACE_H

#include <iostream>
#include <map>
#include <vector>
#include <netcdf.h>

enum class Netcdf_mode { Create, Read, Write };

class Netcdf_handle;
class Netcdf_group;

class Netcdf_variable
{
    public:
        Netcdf_variable(Netcdf_handle&, const int, const std::vector<size_t>&);
        void insert(const std::vector<double>&, const std::vector<size_t>);
        void insert(const double, const std::vector<size_t>);

    private:
        Netcdf_handle& nc_file;
        const int var_id;
        const std::vector<size_t> dim_sizes;
};

class Netcdf_handle
{
    public:
        Netcdf_handle();
        void add_dimension(const std::string&, const size_t dim_size = NC_UNLIMITED);

        Netcdf_group add_group(const std::string&);

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

    protected:
        int ncid;
        int root_ncid;
        std::map<std::string, int> dims;
        int record_counter;
};

class Netcdf_file : public Netcdf_handle
{
    public:
        Netcdf_file(const std::string&, Netcdf_mode);
        ~Netcdf_file();
};

class Netcdf_group : public Netcdf_handle
{
    public:
        Netcdf_group(const int, const int);
};
#endif
