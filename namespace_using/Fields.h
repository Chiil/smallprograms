#ifndef FIELDS_H
#define FIELDS_H

#include <map>
#include <vector>
#include "Types.h"

namespace Fields
{
    struct Fields_data
    {
        std::map<std::string, std::vector<Real>> all_3d;
    };

    Fields_data data;

    // Functions.
    void init()
    {
        int i = 0;
        data.all_3d.emplace("u", std::vector<Real>(3, ++i));
        data.all_3d.emplace("v", std::vector<Real>(3, ++i));
        data.all_3d.emplace("w", std::vector<Real>(3, ++i));
    }
}
#endif
