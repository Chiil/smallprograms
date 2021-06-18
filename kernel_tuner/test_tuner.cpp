#include <iostream>
#include <functional>
#include <chrono>


struct Empty_settings {};


template<class Func>
class Tuner
{
    public:
        Tuner(Func&& f) : f(f) {}

        template<class... Args>
        typename std::function<Func>::result_type run(Args&&... args)
        {
            return f(std::forward<Args>(args)...);
        }

        template<class... Args, class Settings = Empty_settings>
        typename std::function<Func>::result_type tune(Args&&... args)
        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i=0; i<10; ++i)
                f(std::forward<Args>(args)...);
            auto end = std::chrono::high_resolution_clock::now();
            double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

            std::cout << "Duration: " << duration/10 << " (s)" << std::endl;
        }

    private:
        std::function<Func> f;
};


template<class Settings>
void diff(double* const __restrict__ at, const double* const __restrict__ a, const double visc,
          const double dxidxi, const double dyidyi, const double dzidzi,
          const int itot, const int jtot, const int ktot)
{
    const int ii = 1;
    const int jj = itot;
    const int kk = itot*jtot;

    constexpr int k_block = Settings::k_block;

    #pragma omp parallel for simd schedule(static)
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


struct Diff_settings { static constexpr int k_block = 8; };

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

    Tuner tuner{diff<Diff_settings>};

    tuner.run(at.data(), a.data(), visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot);
    tuner.tune(at.data(), a.data(), visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot);

    return 0;
}
