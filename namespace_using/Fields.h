#ifndef FIELDS
#define FIELDS

#include <map>
#include <vector>

namespace Fields
{
    std::map<std::string, std::vector<double>> ap;

    void init()
    {
        int i = 0;
        ap["u"] = std::vector<double>(3, ++i);
        ap["v"] = std::vector<double>(3, ++i);
        ap["w"] = std::vector<double>(3, ++i);
    }
}
#endif
