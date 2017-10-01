#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <stdlib.h>
#include <cstdio>
#include <ctime>
#include <vector>
#include "math.h"

void init(std::vector<double>& a, std::vector<double>& at, const int ncells)
{
    for (int i=0; i<ncells; ++i)
    {
        a[i]  = pow(i,2)/pow(i+1,2);
        at[i] = 0.;
    }
}

void diff(std::vector<double>& at, const std::vector<double>& a, const double visc, 
          const double dxidxi, const double dyidyi, const double dzidzi, 
          const int itot, const int jtot, const int ktot)
{
    const int ii = 1;
    const int jj = itot;
    const int kk = itot*jtot;

    for (int k=1; k<ktot-1; k++)
        for (int j=1; j<jtot-1; j++)
        #pragma ivdep
            for (int i=1; i<itot-1; i++)
            {
                const int ijk = i + j*jj + k*kk;
                at[ijk] += visc * (
                        + ( (a[ijk+ii] - a[ijk   ]) 
                          - (a[ijk   ] - a[ijk-ii]) ) * dxidxi 
                        + ( (a[ijk+jj] - a[ijk   ]) 
                          - (a[ijk   ] - a[ijk-jj]) ) * dyidyi
                        + ( (a[ijk+kk] - a[ijk   ]) 
                          - (a[ijk   ] - a[ijk-kk]) ) * dzidzi
                        );
            }
}

int main()
{
    const int nloop = 100;
    const int itot = 384;
    const int jtot = 384;
    const int ktot = 384;
    const int ncells = itot*jtot*ktot;

    std::vector<double> a (ncells);
    std::vector<double> at(ncells);
   
    init(a, at, ncells);

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
