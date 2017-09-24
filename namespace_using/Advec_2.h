#ifndef ADVEC_2
#define ADVEC_2

#include "Grid.h"
#include "Fields.h"

namespace Advec_2
{
    void kernel(double* u, const double* v, const double* w, const int itot)
    {
        for (int i=0; i<itot; ++i)
            u[i] += v[i] + w[i];
    }

    void exec()
    {
        using Grid::itot;
        using Fields::ap;

        kernel(ap.at("u").data(), ap.at("v").data(), ap.at("w").data(), itot);

        for (int i=0; i<itot; ++i)
            std::cout << i << ": " << ap.at("u")[i] 
                           << ", " << ap.at("v")[i] 
                           << ", " << ap.at("w")[i]
                           << std::endl;
    }
}
#endif
