#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <Kokkos_core.hpp>


namespace
{
    using Array_3d = Kokkos::View<double***, Kokkos::LayoutRight>;
    using Array_3d_inner = Kokkos::View<double***, Kokkos::LayoutStride>;
    using size_type = Array_3d::size_type;

    using Range_3d = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    void init(double* const __restrict__ a, double* const __restrict__ at, const size_t ncells)
    {
        for (size_t i=0; i<ncells; ++i)
        {
            a [i] = pow(i,2)/pow(i+1,2);
            at[i] = 0.;
        }
    }
    
    struct diff
    {
        Array_3d at;
        const Array_3d a;
        const double visc;
        const double dxidxi;
        const double dyidyi;
        const double dzidzi;
    
        diff(
                Array_3d at_, const Array_3d a_,
                const double visc_,
                const double dxidxi_, const double dyidyi_, const double dzidzi_) :
            at(at_), a(a_), visc(visc_), dxidxi(dxidxi_), dyidyi(dyidyi_), dzidzi(dzidzi_) {};
    
        KOKKOS_INLINE_FUNCTION
        void operator()(Array_3d::size_type k, Array_3d::size_type j, Array_3d::size_type i) const
        {
            at(k, j, i) += visc * (
                    + ( (a(k+1, j  , i  ) - a(k  , j  , i  ))
                      - (a(k  , j  , i  ) - a(k-1, j  , i  )) ) * dxidxi
                    + ( (a(k  , j+1, i  ) - a(k  , j  , i  ))
                      - (a(k  , j  , i  ) - a(k  , j-1, i  )) ) * dyidyi
                    + ( (a(k  , j  , i+1) - a(k  , j  , i  ))
                      - (a(k  , j  , i  ) - a(k  , j  , i-1)) ) * dzidzi
                    );
        }
    };
    
    void diff_ref(
            double* const __restrict__ at, const double* const __restrict__ a,
            const double visc,
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
}

int main(int argc, char* argv[])
{
    Kokkos::initialize(argc, argv);
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

        Array_3d a ("a" , ktot, jtot, itot);
        Array_3d at("at", ktot, jtot, itot);

        Array_3d_inner at_inner = Kokkos::subview(
                at,
                Kokkos::pair<size_type, size_type>(1, ktot-2),
                Kokkos::pair<size_type, size_type>(1, jtot-2),
                Kokkos::pair<size_type, size_type>(1, itot-2));

        Range_3d range_3d({1, 1, 1}, {ktot-2, jtot-2, itot-2}, {1, 1, 1});

        init(a.data(), at.data(), ncells);

        // Check the results.
        Kokkos::parallel_for(range_3d, diff(at, a, visc, dxidxi, dyidyi, dzidzi));

        printf("at=%.20f\n", at.data()[itot*jtot+itot+itot/2]);

        // Time performance.
        std::clock_t start = std::clock(); 
   
        for (int i=0; i<nloop; ++i)
        {
            Kokkos::parallel_for(range_3d, diff(at, a, visc, dxidxi, dyidyi, dzidzi));

            // diff_ref(
            //         at.data(), a.data(),
            //         dxidxi, dyidyi, dzidzi, visc,
            //         itot, jtot, ktot);
        }

        Kokkos::fence();
  
        double duration = (std::clock() - start ) / (double)CLOCKS_PER_SEC;
   
        printf("time/iter = %f s (%i iters)\n", duration/(double)nloop, nloop);
    }

    Kokkos::finalize();
    
    return 0;
}
