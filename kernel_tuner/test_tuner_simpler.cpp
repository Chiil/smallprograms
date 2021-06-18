#include <iostream>
#include <functional>
#include <chrono>


struct Empty_settings {};


template<class Func, class... Args, class Settings = Empty_settings>
void measure(Func&& f, Args&&... args)
{
    auto start = std::chrono::high_resolution_clock::now();
    for (int i=0; i<10; ++i)
        f(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    std::cout << "Duration: " << duration/10 << " (s)" << std::endl;
}


template<int k_block>
void diff(double* const __restrict__ at, const double* const __restrict__ a, const double visc,
          const double dxidxi, const double dyidyi, const double dzidzi,
          const int itot, const int jtot, const int ktot)
{
    const int ii = 1;
    const int jj = itot;
    const int kk = itot*jtot;

    #pragma omp parallel for simd
    for (int kb=1; kb<ktot-1; kb+=k_block)
        for (int k=kb; k<std::min(ktot-1, kb+k_block); ++k)
            for (int j=1; j<jtot-1; ++j)
                #pragma GCC ivdep
                for (int i=1; i<itot-1; ++i)
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
    int itot{384};
    int jtot{384};
    int ktot{384};

    std::vector<double> at(itot*jtot*ktot);
    std::vector<double> a(itot*jtot*ktot);

    double dxidxi{1.};
    double dyidyi{1.};
    double dzidzi{1.};
    double visc{1.};

    auto lambda = [&]<int... Ks>(const std::integer_sequence<int, Ks...>)
    {
        ((measure(diff<Ks>, at.data(), a.data(), visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot)), ...);
    };

    constexpr std::integer_sequence<int, 1, 2, 4, 8, 16, 32, 64> k_blocks{};
    lambda(k_blocks);

    return 0;
}
