#ifndef ADVEC_2
#define ADVEC_2

#include "Grid.h"
#include "Fields.h"

namespace Advec_2
{
    void kernel(
            double* u, const double* v, const double* w,
            const double dx,
            const int itot)
    {
        std::cout << dx << std::endl;
        for (int i=0; i<itot; ++i)
            u[i] += v[i] + w[i]*dx;
    }

    void exec()
    {
        using Grid::grid_data;
        using Fields::ap;

        kernel(ap.at("u").data(), ap.at("v").data(), ap.at("w").data(),
                grid_data.dx, grid_data.itot);

        for (int i=0; i<grid_data.itot; ++i)
        {
            std::cout << i << ": " << ap.at("u")[i] 
                           << ", " << ap.at("v")[i] 
                           << ", " << ap.at("w")[i]
                           << std::endl;
        }
    }
}
#endif
