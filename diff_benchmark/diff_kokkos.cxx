#include <cstdio>
#include <Kokkos_Core.hpp>

namespace
{
    using Array_3d = Kokkos::View<float***, Kokkos::LayoutRight>;
    using Range_3d = Kokkos::MDRangePolicy<Kokkos::Rank<3>>;

    void init(float* const __restrict__ a, float* const __restrict__ at, const size_t ncells)
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
        const float visc;
        const float dxidxi;
        const float dyidyi;
        const float dzidzi;

        diff(
                Array_3d at_, const Array_3d a_,
                const float visc_,
                const float dxidxi_, const float dyidyi_, const float dzidzi_) :
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

        constexpr float visc = 0.1;
        constexpr float dxidxi = 0.1;
        constexpr float dyidyi = 0.1;
        constexpr float dzidzi = 0.1;

        Array_3d a ("a" , ktot, jtot, itot);
        Array_3d at("at", ktot, jtot, itot);

        Range_3d range_3d({1, 1, 1}, {ktot-1, jtot-1, itot-1}, {8, 0, 0});

        init(a.data(), at.data(), ncells);

        // Check the results.
        Kokkos::parallel_for(
                "diffusion",
                range_3d,
                diff(at, a, visc, dxidxi, dyidyi, dzidzi));

        printf("at=%.20f\n", at.data()[itot*jtot+itot+itot/2]);

        // Time performance.
        Kokkos::Timer timer;

        for (int i=0; i<nloop; ++i)
        {
            Kokkos::parallel_for(
                    range_3d,
                    diff(at, a, visc, dxidxi, dyidyi, dzidzi));
        }

        Kokkos::fence();

        double duration = timer.seconds();

        printf("time/iter = %f s (%i iters)\n", duration/(double)nloop, nloop);

        printf("at=%.20f\n", at.data()[itot*jtot+itot+itot/4]);
    }

    Kokkos::finalize();

    return 0;
}
