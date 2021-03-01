#include <fstream>
#include <iostream>
#include <cstdio>
#include <Kokkos_Core.hpp>

namespace
{
    using Array_3d_cpu = Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace>;
    using Array_3d_gpu = Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::CudaSpace>;

    using Range_3d_cpu = Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<3>>;
    using Range_3d_gpu = Kokkos::MDRangePolicy<
        Kokkos::Cuda,
        Kokkos::Rank<3, Kokkos::Iterate::Left, Kokkos::Iterate::Right>>;

    void init(double* const __restrict__ a, double* const __restrict__ at, const size_t ncells)
    {
        for (size_t i=0; i<ncells; ++i)
        {
            a [i] = pow(i,2)/pow(i+1,2);
            at[i] = 0.;
        }
    }

    template<typename Array_3d>
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

        constexpr int nloop = 30;
        const size_t itot = std::stoi(argv[1]);
        const size_t jtot = std::stoi(argv[1]);
        const size_t ktot = std::stoi(argv[1]);
        const size_t ncells = itot*jtot*ktot;

        constexpr double visc = 0.1;
        constexpr double dxidxi = 0.1;
        constexpr double dyidyi = 0.1;
        constexpr double dzidzi = 0.1;

        Array_3d_gpu a_gpu ("a_gpu" , ktot, jtot, itot);
        Array_3d_gpu at_gpu("at_gpu", ktot, jtot, itot);

        Range_3d range_3d({1, 1, 1}, {ktot-1, jtot-1, itot-1}, {1, 1, 64});

        Array_3d_gpu::HostMirror a_cpu = Kokkos::create_mirror_view(a_gpu);
        Array_3d_gpu::HostMirror at_cpu = Kokkos::create_mirror_view(at_gpu);

        init(a.data(), at.data(), ncells);

        Kokkos::deep_copy(a_gpu, a);
        Kokkos::deep_copy(at_gpu, at);

        // Time performance.
        Kokkos::Timer timer;

        for (int i=0; i<nloop; ++i)
        {
            Kokkos::parallel_for(
                    range_3d_gpu,
                    diff<Array_3d_gpu>(at_gpu, a_gpu, visc, dxidxi, dyidyi, dzidzi));
        }

        Kokkos::fence();

        double duration = timer.seconds();

        printf("time/iter = %E s (%i iters)\n", duration/(double)nloop, nloop);

        // Time performance.
        timer.reset();

        for (int i=0; i<nloop; ++i)
        {
            Kokkos::parallel_for(
                    range_3d_cpu,
                    diff<Array_3d_cpu>(at_cpu, at_cpu, visc, dxidxi, dyidyi, dzidzi));
        }

        Kokkos::fence();

        double duration = timer.seconds();

        printf("time/iter = %E s (%i iters)\n", duration/(double)nloop, nloop);

        // Kokkos::deep_copy(at, at_gpu);

        // printf("at=%.20f\n", at.data()[itot*jtot+itot+itot/4]);

    }

    Kokkos::finalize();

    return 0;
}
