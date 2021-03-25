#ifndef FIELDS_H
#define FIELDS_H

#include <map>
#include <vector>

namespace Fields
{
    template<typename TF>
    struct Fields_data
    {
        std::map<std::string, std::vector<TF>> all_3d;
    };

    Fields_data<FLOAT_TYPE> data;

    // Functions.
    void init()
    {
        int i = 0;
        data.all_3d.emplace("u", std::vector<double>(3, ++i));
        data.all_3d.emplace("v", std::vector<double>(3, ++i));
        data.all_3d.emplace("w", std::vector<double>(3, ++i));
    }
}
#endif
