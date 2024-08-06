#include <iostream>
#include <map>
#include "toml.hpp"


int main()
{
    std::map<std::string, std::vector<int>> cross_xy_map;

    try
    {
        toml::table tbl = toml::parse_file("test.toml");

        auto cross_xy = tbl["cross"]["cross_xy"].as_array();

        if (cross_xy == nullptr)
            throw std::runtime_error("Item not found");
        else
        {
            for (auto& c : *cross_xy)
            {
                
                std::string name(c[0]);
                std::cout << name << std::endl;
            }
        }
    }

    catch (const toml::parse_error& e)
    {
        std::cout << "Caught exception: " << e.description() << std::endl;
        return 1;
    }

    catch (const std::exception& e)
    {
        std::cout << "Caught exception: " << e.what() << std::endl;
        return 1;
    }

    catch (...)
    {
        std::cout << "Uncaught exception, debugging needed!" << std::endl;
        return 1;
    }

    return 0;
}

