#ifndef GRID_H
#define GRID_H

namespace Grid
{
    template<typename TF>
    struct Grid_data
    {
        int itot;
        TF dx;
    };

    Grid_data<FLOAT_TYPE> grid_data;

    void init()
    {
        grid_data.itot = 3;
        grid_data.dx = 10;
    }
}
#endif
