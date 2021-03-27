#ifndef FIELDS_H
#define FIELDS_H

#include <map>
#include <vector>
#include "Types.h"

namespace Fields
{
    std::map<std::string, std::vector<Real>> all_3d;

    // Functions.
    void init()
    {
        int i = 0;
        all_3d.emplace("u", std::vector<Real>(3, ++i));
        all_3d.emplace("v", std::vector<Real>(3, ++i));
        all_3d.emplace("w", std::vector<Real>(3, ++i));
    }
}
#endif
