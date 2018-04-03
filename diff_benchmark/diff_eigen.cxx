#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <unsupported/Eigen/CXX11/Tensor>

typedef Eigen::Tensor<double, 3> Field3d;

void init(double* const __restrict__ a, double* const __restrict__ at, const int ncells)
{
    for (int i=0; i<ncells; ++i)
    {
        a[i]  = pow(i,2)/pow(i+1,2);
        at[i] = 0.;
    }
}

void diff_eigen(
        Field3d& at, const Field3d& a, const double visc, 
        const double dxidxi, const double dyidyi, const double dzidzi, 
        const int itot, const int jtot, const int ktot)
{
    Eigen::array<int, 3> mid = {{1, 1, 1}};
    Eigen::array<int, 3> east = {{2, 1, 1}};
    Eigen::array<int, 3> west = {{0, 1, 1}};
    Eigen::array<int, 3> north = {{1, 2, 1}};
    Eigen::array<int, 3> south = {{1, 0, 1}};
    Eigen::array<int, 3> top = {{1, 1, 2}};
    Eigen::array<int, 3> bot = {{1, 1, 0}};
    Eigen::array<int, 3> ext = {{itot-2, jtot-2, ktot-2}};

    at.slice(mid, ext)
        += visc * ( ( a.slice( east, ext) - 2.*a.slice(mid, ext) + a.slice( west, ext) )*dxidxi
                  + ( a.slice(north, ext) - 2.*a.slice(mid, ext) + a.slice(south, ext) )*dyidyi
                  + ( a.slice(  top, ext) - 2.*a.slice(mid, ext) + a.slice(  bot, ext) )*dzidzi );
}

void diff_basic(
        double* const __restrict__ at, const double* const __restrict__ a, const double visc, 
        const double dxidxi, const double dyidyi, const double dzidzi, 
        const int itot, const int jtot, const int ktot)
{
    const int ii = 1;
    const int jj = itot;
    const int kk = itot*jtot;

    for (int k=1; k<ktot-1; k++)
        for (int j=1; j<jtot-1; j++)
#pragma GCC ivdep
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
    const int nloop = 10;
    const int itot = 384;
    const int jtot = 384;
    const int ktot = 384;
    const int ncells = itot*jtot*ktot;

    Field3d a (itot, jtot, ktot);
    Field3d at(itot, jtot, ktot);

    init(a.data(), at.data(), ncells);

    // Time performance 
    std::clock_t start = std::clock(); 

    for (int i=0; i<nloop; ++i)
        diff_eigen(at, a, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot); 

    double duration = (std::clock() - start ) / (double)CLOCKS_PER_SEC;

    printf("time/iter = %f s (%i iters)\n",duration/(double)nloop, nloop);

    // Time performance 
    start = std::clock(); 

    for (int i=0; i<nloop; ++i)
        diff_basic(at.data(), a.data(), 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot); 

    duration = (std::clock() - start ) / (double)CLOCKS_PER_SEC;

    printf("NO EIGEN: time/iter = %f s (%i iters)\n",duration/(double)nloop, nloop);

    return 0;
}
