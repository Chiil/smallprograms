#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>

void init(
        float* const __restrict__ a,
        float* const __restrict__ u,
        float* const __restrict__ v,
        float* const __restrict__ w,
        float* const __restrict__ at,
        const int ncells)
{
    for (int i=0; i<ncells; ++i)
    {
        a[i] = pow(i,2)/pow(i+1,2);
        u[i] = pow(i,2)/pow(i+1,2);
        v[i] = pow(i,2)/pow(i+1,2);
        w[i] = pow(i,2)/pow(i+1,2);
        at[i] = 0.f;
    }
}


float interp2(const float a, const float b) { return 0.5f*(a + b); }


void diff(float* const __restrict__ at, const float* const __restrict__ a,
          const float* const __restrict__ u,
          const float* const __restrict__ v,
          const float* const __restrict__ w,
          const float dxi, const float dyi, const float dzi,
          const int itot, const int jtot, const int ktot)
{
    const int ii = 1;
    const int jj = itot;
    const int kk = itot*jtot;

    constexpr int k_block = 8;

    for (int kb=1; kb<ktot-1; kb+=k_block)
        for (int k=kb; k<std::min(ktot-1, kb+k_block); ++k)
            for (int j=1; j<jtot-1; ++j)
                #pragma GCC ivdep
                for (int i=1; i<itot-1; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    at[ijk] += (
                            - (  u[ijk+ii] * interp2(a[ijk   ], a[ijk+ii])
                               - u[ijk   ] * interp2(a[ijk-ii], a[ijk   ]) ) * dxi

                            - (  v[ijk+jj] * interp2(a[ijk   ], a[ijk+jj])
                               - v[ijk   ] * interp2(a[ijk-jj], a[ijk   ]) ) * dyi

                            - (  w[ijk+kk] * interp2(a[ijk   ], a[ijk+kk])
                               - w[ijk   ] * interp2(a[ijk-kk], a[ijk   ]) ) * dzi
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

    float *a = new float[ncells];
    float *u = new float[ncells];
    float *v = new float[ncells];
    float *w = new float[ncells];
    float *at = new float[ncells];

    init(a, u, v, w, at, ncells);

    // Check results
    diff(at, a, u, v, w, 0.1f, 0.1f, 0.1f, itot, jtot, ktot);
    printf("at=%.20f\n",at[itot*jtot+itot+itot/2]);

    // Time performance
    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<nloop; ++i)
        diff(at, a, u, v, w, 0.1f, 0.1f, 0.1f, itot, jtot, ktot);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    printf("time/iter = %f s (%i iters)\n",duration/(double)nloop, nloop);

    printf("at=%.20f\n", at[itot*jtot+itot+itot/4]);

    return 0;
}
