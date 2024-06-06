#include <iostream>
#include <map>
#include "toml.hpp"


int main()
{
    toml::table tbl = toml::parse_file("test.toml");
    std::cout << tbl << std::endl;

    auto cross_xy = tbl["cross"]["cross_xy"];

    std::cout << cross_xy << std::endl;

    std::map<std::string, std::vector<int>> cross_xy_map;

    for (const auto& [name, values_ptr] : *cross_xy.as_table())
    {
        auto values = *values_ptr.as_array();

        std::string map_name(name.str());

        std::vector<int> map_values;
        for (const auto& value : values)
        {
            int value_int = value.as_integer()->get();
            map_values.push_back(value_int);
        }

        cross_xy_map.emplace(map_name, map_values);
    }

    for (const auto& [name, values] : cross_xy_map)
    {
        for (const auto& value : values)
            std::cout << name << ": " << value << std::endl;
    }
}

