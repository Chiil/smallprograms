#ifndef GRID_H
#define GRID_H

#include "Types.h"

namespace Grid
{
    struct Grid_data
    {
        int itot;
        Real dx;
    };

    Grid_data data;

    void init()
    {
        data.itot = 3;
        data.dx = 10;
    }
}
#endif
