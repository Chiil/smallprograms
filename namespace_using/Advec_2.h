#ifndef ADVEC_2
#define ADVEC_2

#include "Grid.h"
#include "Fields.h"

namespace Advec_2
{
    void kernel(
            Real_ptr u, const Real_ptr v, const Real_ptr w,
            const Real dx,
            const int itot)
    {
        for (int i=0; i<itot; ++i)
            u[i] += v[i] + w[i]*dx;
    }

    void exec()
    {
        using Grid::dx, Grid::itot;
        using Fields::all_3d;

        kernel(
                all_3d.at("u").data(), all_3d.at("v").data(), all_3d.at("w").data(),
                dx, itot);
    }
}
#endif
