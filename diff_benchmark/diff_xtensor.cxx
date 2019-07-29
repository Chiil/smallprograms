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

void diff_ref(
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

    auto at_c = xt::view(at, xt::range(1, ktot-1), xt::range(1, jtot-1), xt::range(1, itot-1));

    const auto a_c = xt::view(a, xt::range(1, ktot-1), xt::range(1, jtot-1), xt::range(1, itot-1));
    const auto a_w = xt::view(a, xt::range(1, ktot-1), xt::range(1, jtot-1), xt::range(0, itot-2));
    const auto a_e = xt::view(a, xt::range(1, ktot-1), xt::range(1, jtot-1), xt::range(2, itot  ));
    const auto a_s = xt::view(a, xt::range(1, ktot-1), xt::range(0, jtot-2), xt::range(1, itot-1));
    const auto a_n = xt::view(a, xt::range(1, ktot-1), xt::range(2, jtot  ), xt::range(1, itot-1));
    const auto a_b = xt::view(a, xt::range(0, ktot-2), xt::range(1, jtot-1), xt::range(1, itot-1));
    const auto a_t = xt::view(a, xt::range(2, ktot  ), xt::range(1, jtot-1), xt::range(1, itot-1));

    std::cout << typeid(a_c).name() << std::endl;

    init(a.data(), at.data(), ncells);

    // Check the results.
    xt::noalias(at_c) += visc * ( (a_e - 2*a_c + a_w) * dxidxi
                                + (a_n - 2*a_c + a_s) * dyidyi
                                + (a_t - 2*a_c + a_b) * dzidzi );

    printf("at=%.20f\n", at.data()[itot*jtot+itot+itot/2]);

    // Time performance.
    std::clock_t start = std::clock(); 
   
    for (int i=0; i<nloop; ++i)
    {
        xt::noalias(at_c) += visc * ( (a_e - 2*a_c + a_w) * dxidxi
                                    + (a_n - 2*a_c + a_s) * dyidyi
                                    + (a_t - 2*a_c + a_b) * dzidzi );

        // Handwritten kernel 10 times faster than code above.
        // diff_ref(
        //         at.data(), a.data(),
        //         dxidxi, dyidyi, dzidzi, visc,
        //         itot, jtot, ktot);
    }
  
    double duration = (std::clock() - start ) / (double)CLOCKS_PER_SEC;
   
    printf("time/iter = %f s (%i iters)\n", duration/(double)nloop, nloop);
    
    return 0;
}
