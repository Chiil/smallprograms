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

    Grid_data<FLOAT_TYPE> data;

    void init()
    {
        data.itot = 3;
        data.dx = 10;
    }
}
#endif
