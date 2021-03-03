#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>

void init(double* const __restrict__ a, double* const __restrict__ at, const int ncells)
{
    for (int i=0; i<ncells; ++i)
    {
        a[i]  = pow(i,2)/pow(i+1,2);
        at[i] = 0.;
    }
}

void diff(double* const __restrict__ at, const double* const __restrict__ a, const double visc,
          const double dxidxi, const double dyidyi, const double dzidzi,
          const int itot, const int jtot, const int ktot)
{
    const int ii = 1;
    const int jj = itot;
    const int kk = itot*jtot;

    constexpr int k_block = 8;

    #pragma omp parallel for
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

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Add the grid size as an argument!" << std::endl;
        return 1;
    }

    const int nloop = 30;
    const int itot = std::stoi(argv[1]);
    const int jtot = std::stoi(argv[1]);
    const int ktot = std::stoi(argv[1]);
    const int ncells = itot*jtot*ktot;

    double *a  = new double[ncells];
    double *at = new double[ncells];

    init(a, at, ncells);

    // Check results
    // diff(at, a, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot);
    // printf("at=%.20f\n",at[itot*jtot+itot+itot/2]);

    // Time performance
    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<nloop; ++i)
        diff(at, a, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    printf("time/iter = %f s (%i iters)\n",duration/(double)nloop, nloop);

    printf("at=%.20f\n", at[itot*jtot+itot+itot/4]);

    /*
    std::ofstream binary_file("at_ref.bin", std::ios::out | std::ios::trunc | std::ios::binary);

    if (binary_file)
        binary_file.write(reinterpret_cast<const char*>(at), ncells*sizeof(double));
    else
    {
        std::string error = "Cannot write file \"at_cuda.bin\"";
        throw std::runtime_error(error);
    }
    */

    return 0;
}
