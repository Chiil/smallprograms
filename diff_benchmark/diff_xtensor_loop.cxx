#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <xtensor/xarray.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>

void init(double* const __restrict__ a, double* const __restrict__ at, const size_t ncells)
{
    for (size_t i=0; i<ncells; ++i)
    {
        a [i] = pow(i,2)/pow(i+1,2);
        at[i] = 0.;
    }
}

template<typename T>
void diff_xtensor(
        T& at, const T& a,
        const double visc, 
        const double dxidxi, const double dyidyi, const double dzidzi, 
        const int itot, const int jtot, const int ktot)
{
    for (int k=1; k<ktot-1; k++)
        for (int j=1; j<jtot-1; j++)
            #pragma GCC ivdep
            for (int i=1; i<itot-1; i++)
            {
                at(k, j, i) += visc * (
                        + ( (a(k  , j  , i+1) - a(k  , j  , i  )) 
                          - (a(k  , j  , i  ) - a(k  , j  , i-1)) ) * dxidxi 
                        + ( (a(k  , j+1, i  ) - a(k  , j  , i  )) 
                          - (a(k  , j  , i  ) - a(k  , j-1, i  )) ) * dyidyi
                        + ( (a(k+1, j  , i  ) - a(k  , j  , i  )) 
                          - (a(k  , j  , i  ) - a(k-1, j  , i  )) ) * dzidzi
                        );
            }
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Add the grid size as an argument!" << std::endl;
        return 1;
    }

    constexpr int nloop = 10;
    const size_t itot = std::stoi(argv[1]);
    const size_t jtot = std::stoi(argv[1]);
    const size_t ktot = std::stoi(argv[1]);
    const size_t ncells = itot*jtot*ktot;

    constexpr double visc = 0.1;
    constexpr double dxidxi = 0.1;
    constexpr double dyidyi = 0.1;
    constexpr double dzidzi = 0.1;

    xt::xtensor<double, 3> a ({ktot, jtot, itot});
    xt::xtensor<double, 3> at({ktot, jtot, itot});

    init(a.data(), at.data(), ncells);

    // Check the results.
    diff_xtensor(
            at, a,
            visc,
            dxidxi, dyidyi, dzidzi,
            itot, jtot, ktot);

    printf("at=%.20f\n", at.data()[itot*jtot+itot+itot/2]);

    // Time performance.
    std::clock_t start = std::clock(); 
   
    for (int i=0; i<nloop; ++i)
    {
        diff_xtensor(
                at, a,
                visc,
                dxidxi, dyidyi, dzidzi,
                itot, jtot, ktot);
    }
  
    double duration = (std::clock() - start ) / (double)CLOCKS_PER_SEC;
   
    printf("time/iter = %f s (%i iters)\n", duration/(double)nloop, nloop);
    
    return 0;
}
