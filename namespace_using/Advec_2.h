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
        auto& gd = Grid::data;
        auto& fd = Fields::data;

        kernel(fd.all_3d.at("u").data(), fd.all_3d.at("v").data(), fd.all_3d.at("w").data(),
                gd.dx, gd.itot);
    }
}
#endif
