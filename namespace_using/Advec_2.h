#ifndef ADVEC_2
#define ADVEC_2

#include "Grid.h"
#include "Fields.h"

namespace Advec_2
{
    template<typename TF>
    void kernel(
            TF* u, const TF* v, const TF* w,
            const TF dx,
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
