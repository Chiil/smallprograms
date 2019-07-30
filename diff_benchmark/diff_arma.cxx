#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <stdlib.h>
#include <cstdio>
#include <ctime>
#include "math.h"
#include <armadillo>

using namespace arma;

void init(double* const __restrict__ a, double* const __restrict__ at, const int ncells)
{
    for (int i=0; i<ncells; ++i)
    {
        a[i]  = pow(i,2)/pow(i+1,2);
        at[i] = 0.;
    }
}

typedef subview_cube<double> scd;

void diff(
        scd& at_mid, const scd& a_mid, const scd& a_west, const scd& a_east,
        const scd& a_south, const scd& a_north, const scd& a_bot, const scd& a_top,
        const double visc, 
        const double dxidxi, const double dyidyi, const double dzidzi, 
        const int itot, const int jtot, const int ktot)
{
    at_mid += visc * ( (a_east  - 2.*a_mid + a_west )*dxidxi
                     + (a_north - 2.*a_mid + a_south)*dyidyi
                     + (a_top   - 2.*a_mid + a_bot  )*dzidzi );
}

int main()
{
    const int nloop = 10;
    const int itot = 384;
    const int jtot = 384;
    const int ktot = 384;
    const int ncells = itot*jtot*ktot;

    cube at = cube(itot, jtot, ktot);
    cube a = cube(itot, jtot, ktot);

    init(a.memptr(), at.memptr(), ncells);

    subview_cube<double> at_mid  = at.subcube(1, 1, 1, itot-2, jtot-2, ktot-2);
    subview_cube<double> a_mid   = a .subcube(1, 1, 1, itot-2, jtot-2, ktot-2);
    subview_cube<double> a_west  = a .subcube(0, 1, 1, itot-3, jtot-2, ktot-2);
    subview_cube<double> a_east  = a .subcube(2, 1, 1, itot-1, jtot-2, ktot-2);
    subview_cube<double> a_south = a .subcube(1, 0, 1, itot-2, jtot-3, ktot-2);
    subview_cube<double> a_north = a .subcube(1, 2, 1, itot-2, jtot-1, ktot-2);
    subview_cube<double> a_bot   = a .subcube(1, 1, 0, itot-2, jtot-2, ktot-3);
    subview_cube<double> a_top   = a .subcube(1, 1, 2, itot-2, jtot-2, ktot-1);

    // Check results
    diff(at_mid, a_mid, a_west, a_east, a_south, a_north, a_bot, a_top,
            0.1, 0.1, 0.1, 0.1, itot, jtot, ktot);

    printf("at=%.20f\n",at[itot*jtot+itot+itot/2]);
 
    // Time performance 
    std::clock_t start = std::clock(); 
   
    for (int i=0; i<nloop; ++i)
        diff(at_mid, a_mid, a_west, a_east, a_south, a_north, a_bot, a_top,
                0.1, 0.1, 0.1, 0.1, itot, jtot, ktot);
  
    double duration = (std::clock() - start ) / (double)CLOCKS_PER_SEC;
   
    printf("time/iter = %f s (%i iters)\n",duration/(double)nloop, nloop);
    
    return 0;
}
