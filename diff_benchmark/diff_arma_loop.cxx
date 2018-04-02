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

void diff(cube& at, const cube& a, const double visc, 
          const double dxidxi, const double dyidyi, const double dzidzi, 
          const int itot, const int jtot, const int ktot)
{
    for (int k=1; k<ktot-1; k++)
        for (int j=1; j<jtot-1; j++)
            for (int i=1; i<itot-1; i++)
            {
                at(i, j, k) += visc * (
                        + ( (a(i+1, j, k) - a(i  , j, k))
                          - (a(i  , j, k) - a(i-1, j, k)) ) * dxidxi
                        + ( (a(i, j+1, k) - a(i, j  , k))
                          - (a(i, j  , k) - a(i, j-1, k)) ) * dyidyi
                        + ( (a(i, j, k+1) - a(i, j, k  ))
                          - (a(i, j, k  ) - a(i, j, k-1)) ) * dzidzi
                        );
            }
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

    // Check results
    diff(at, a, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot); 
    printf("at=%.20f\n",at[itot*jtot+itot+itot/2]);
 
    // Time performance 
    std::clock_t start = std::clock(); 
   
    for (int i=0; i<nloop; ++i)
        diff(at, a, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot); 
  
    double duration = (std::clock() - start ) / (double)CLOCKS_PER_SEC;
   
    printf("time/iter = %f s (%i iters)\n",duration/(double)nloop, nloop);
    
    return 0;
}
