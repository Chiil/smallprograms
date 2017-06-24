#include <vector>
#include <iostream>
#include <stdexcept>

#include "Input.h"
#include "Data_block.h"

template <typename T>
void print_vector(const std::vector<T>& v, const std::string name)
{
    std::cout << "vector "<< name << "(" << v.size() << "):" << std::endl;
    for (const T& i : v)
        std::cout << i << " ";
    std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    try
    {
        if (argc != 2)
            throw std::runtime_error("Illegal number or arguments");

        std::string case_name = std::string(argv[1]);
        std::string ini_file_name = case_name + ".ini";
        std::string data_file_name = case_name + ".data";

        Input input(ini_file_name);
        // input.printItemList();

        int itot = input.get_item<int>("grid", "itot");
        double xsize = input.get_item<double>("grid", "xsize");
        double zsize = input.get_item<double>("grid", "zsize");
        std::string swthermo = input.get_item<std::string>("thermo", "swthermo");
        std::vector<std::string> crosslist = input.get_list<std::string>("cross", "crosslist");
        std::vector<double> xy = input.get_list<double>("cross", "xy");
        double rndamp = input.get_item<double>("fields", "rndamp");
        double rndampb = input.get_item<double>("fields", "rndamp", "b");

        std::cout << "itot = " << itot  << std::endl;
        std::cout << "xsize = " << xsize << std::endl;
        std::cout << "zsize = " << zsize << std::endl;
        std::cout << "swthermo = " << swthermo << std::endl;
        std::cout << "crosslist = ";
        for (std::string &s : crosslist)
            std::cout << "\"" << s << "\"" << " ";
        std::cout << std::endl;

        std::cout << "xy = ";
        for (const double &i : xy)
            std::cout << i << " ";
        std::cout << std::endl;

        std::cout << "rndamp = " << rndamp << std::endl;
        std::cout << "rndamp[b] = " << rndampb << std::endl;

        // Read data block.
        Data_block data_block(data_file_name);

        std::vector<std::string> a = data_block.get_vector<std::string>("a", 4);
        std::vector<double> b = data_block.get_vector<double>("b", 1);
        std::vector<int> c = data_block.get_vector<int>("c", 4);
        std::vector<double> d;
        d = data_block.get_vector<double>("d", 2);
        std::vector<double> e;
        d = data_block.get_vector<double>("e", 2, 2);

        print_vector(a, "a");
        print_vector(b, "b");
        print_vector(c, "c");
        print_vector(d, "d");
        print_vector(e, "e");
    }
    catch (std::exception &e)
    {
        std::cout << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
