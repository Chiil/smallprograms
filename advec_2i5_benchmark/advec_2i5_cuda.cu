#include <openacc.h>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <cmath>


namespace
{
    template<typename T> T* acc_to_cuda(T* ptr) { return static_cast<T*>(acc_deviceptr(ptr)); }
}

#define CUDA_MACRO __device__

CUDA_MACRO inline double interp2(const double a, const double b)
{
    return double(0.5) * (a + b);
}


CUDA_MACRO inline double interp4_ws(const double a, const double b, const double c, const double d)
{
    constexpr double c0 = double(7./12.);
    constexpr double c1 = double(1./12.);
    return c0*(b+c) - c1*(a+d);
}


CUDA_MACRO inline double interp3_ws(const double a, const double b, const double c, const double d)
{
    constexpr double c0 = double(3./12.);
    constexpr double c1 = double(1./12.);
    return c0*(c-b) - c1*(d-a);
}


CUDA_MACRO inline double interp6_ws(
        const double a, const double b, const double c, const double d, const double e, const double f)
{
    constexpr double c0 = double(37./60.);
    constexpr double c1 = double(8./60.);
    constexpr double c2 = double(1./60.);

    return c0*(c+d) - c1*(b+e) + c2*(a+f);
}


CUDA_MACRO inline double interp5_ws(
        const double a, const double b, const double c, const double d, const double e, const double f)
{
    constexpr double c0 = double(10./60.);
    constexpr double c1 = double(5./60.);
    constexpr double c2 = double(1./60.);

    return c0*(d-c) - c1*(e-b) + c2*(f-a);
}


void init(double* const __restrict__ a, const int ncells)
{
    static int i = 0;
    for (int n=0; n<ncells; ++n)
    {
        a[n] = pow(i,2)/pow(i+1,2);
        ++i;
    }
}


__global__
void advec_2i5(
        double* const restrict ut,
        const double* const restrict u,
        const double* const restrict v,
        const double* const restrict w,
        const double* const restrict dzi,
        const double dx, const double dy,
        const double* const restrict rhoref,
        const double* const restrict rhorefh,
        const int istart, const int iend,
        const int jstart, const int jend,
        const int kstart, const int kend,
        const int jj, const int kk)
{
    const int ii1 = 1;
    const int ii2 = 2;
    const int ii3 = 3;

    const int jj1 = jj;
    const int jj2 = 2*jj;
    const int jj3 = 3*jj;

    const int kk1 = kk;
    const int kk2 = 2*kk;
    const int kk3 = 3*kk;

    const double dxi = double(1.)/dx;
    const double dyi = double(1.)/dy;

    const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;
    const int k = blockIdx.z + kstart;

    if (i < iend && j < jend && k < kend)
    {
        const int ijk = i + j*jj1 + k*kk1;
        ut[ijk] +=
                // u*du/dx
                - ( interp2(u[ijk        ], u[ijk+ii1]) * interp6_ws(u[ijk-ii2], u[ijk-ii1], u[ijk    ], u[ijk+ii1], u[ijk+ii2], u[ijk+ii3])
                  - interp2(u[ijk-ii1    ], u[ijk    ]) * interp6_ws(u[ijk-ii3], u[ijk-ii2], u[ijk-ii1], u[ijk    ], u[ijk+ii1], u[ijk+ii2]) ) * dxi

                + ( std::abs(interp2(u[ijk        ], u[ijk+ii1])) * interp5_ws(u[ijk-ii2], u[ijk-ii1], u[ijk    ], u[ijk+ii1], u[ijk+ii2], u[ijk+ii3])
                  - std::abs(interp2(u[ijk-ii1    ], u[ijk    ])) * interp5_ws(u[ijk-ii3], u[ijk-ii2], u[ijk-ii1], u[ijk    ], u[ijk+ii1], u[ijk+ii2]) ) * dxi

                // v*du/dy
                - ( interp2(v[ijk-ii1+jj1], v[ijk+jj1]) * interp6_ws(u[ijk-jj2], u[ijk-jj1], u[ijk    ], u[ijk+jj1], u[ijk+jj2], u[ijk+jj3])
                  - interp2(v[ijk-ii1    ], v[ijk    ]) * interp6_ws(u[ijk-jj3], u[ijk-jj2], u[ijk-jj1], u[ijk    ], u[ijk+jj1], u[ijk+jj2]) ) * dyi

                + ( std::abs(interp2(v[ijk-ii1+jj1], v[ijk+jj1])) * interp5_ws(u[ijk-jj2], u[ijk-jj1], u[ijk    ], u[ijk+jj1], u[ijk+jj2], u[ijk+jj3])
                  - std::abs(interp2(v[ijk-ii1    ], v[ijk    ])) * interp5_ws(u[ijk-jj3], u[ijk-jj2], u[ijk-jj1], u[ijk    ], u[ijk+jj1], u[ijk+jj2]) ) * dyi;

        if (k >= kstart+3 && k < kend-3)
        {
                    ut[ijk] +=
                            // w*du/dz
                            - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp6_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2], u[ijk+kk3])
                              - rhorefh[k  ] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp6_ws(u[ijk-kk3], u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2]) ) / rhoref[k] * dzi[k]

                            + ( rhorefh[k+1] * std::abs(interp2(w[ijk-ii1+kk1], w[ijk+kk1])) * interp5_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2], u[ijk+kk3])
                              - rhorefh[k  ] * std::abs(interp2(w[ijk-ii1    ], w[ijk    ])) * interp5_ws(u[ijk-kk3], u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2]) ) / rhoref[k] * dzi[k];
        }

        else if (k == kstart)
        {
                ut[ijk] +=
                        // w*du/dz -> second order interpolation for fluxtop, fluxbot = 0. as w=0
                        - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp2(u[ijk    ], u[ijk+kk1]) ) / rhoref[k] * dzi[k];
        }

        else if (k == kstart+1)
        {
            ut[ijk] +=
                    // w*du/dz -> second order interpolation for fluxbot, fourth order for fluxtop
                    - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp4_ws(u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2])
                      - rhorefh[k  ] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp2(   u[ijk-kk1], u[ijk    ]) ) / rhoref[k] * dzi[k]

                    + ( rhorefh[k+1] * std::abs(interp2(w[ijk-ii1+kk1], w[ijk+kk1])) * interp3_ws(u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2]) ) / rhoref[k] * dzi[k];
        }

        else if (k == kstart+2)
        {
            ut[ijk] +=
                    // w*du/dz -> fourth order interpolation for fluxbot
                    - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp6_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2], u[ijk+kk3])
                      - rhorefh[k  ] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp4_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1]) ) / rhoref[k] * dzi[k]

                    + ( rhorefh[k+1] * std::abs(interp2(w[ijk-ii1+kk1], w[ijk+kk1])) * interp5_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2], u[ijk+kk3])
                      - rhorefh[k  ] * std::abs(interp2(w[ijk-ii1    ], w[ijk    ])) * interp3_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1]) ) / rhoref[k] * dzi[k];
        }

        else if (k == kend-3)
        {
            ut[ijk] +=
                    // w*du/dz -> fourth order interpolation for fluxtop
                    - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp4_ws(u[ijk-kk1   ], u[ijk    ], u[ijk+kk1], u[ijk+kk2])
                      - rhorefh[k  ] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp6_ws(u[ijk-kk3], u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2]) ) / rhoref[k] * dzi[k]

                    + ( rhorefh[k+1] * std::abs(interp2(w[ijk-ii1+kk1], w[ijk+kk1])) * interp3_ws(u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2])
                      - rhorefh[k  ] * std::abs(interp2(w[ijk-ii1    ], w[ijk    ])) * interp5_ws(u[ijk-kk3], u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2]) ) / rhoref[k] * dzi[k];
        }

        else if (k == kend-2)
        {
            ut[ijk] +=
                    // w*du/dz -> second order interpolation for fluxtop, fourth order for fluxbot
                    - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp2(u[ijk    ], u[ijk+kk1])
                      - rhorefh[k  ] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp4_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1]) ) / rhoref[k] * dzi[k]

                    - ( rhorefh[k  ] * std::abs(interp2(w[ijk-ii1    ], w[ijk    ])) * interp3_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1]) ) / rhoref[k] * dzi[k];
        }

        else if (k == kend-1)
        {
            ut[ijk] +=
                    // w*du/dz -> second order interpolation for fluxbot, fluxtop=0 as w=0
                    - ( -rhorefh[k] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp2(u[ijk-kk1], u[ijk    ]) ) / rhoref[k] * dzi[k];
        }
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

    const int icells = itot + 2*3;
    const int jcells = jtot + 2*3;
    const int kcells = ktot + 2*3;
    const int ncells = icells*jcells*kcells;

    const int istart = 3;
    const int jstart = 3;
    const int kstart = 3;

    const int iend = itot + 3;
    const int jend = jtot + 3;
    const int kend = ktot + 3;

    const int jstride = icells;
    const int kstride = icells*jcells;

    double* ut  = new double[ncells];
    double* u = new double[ncells];
    double* v = new double[ncells];
    double* w = new double[ncells];
    double* dzi  = new double[kcells];

    double* rhoref = new double[kcells];
    double* rhorefh = new double[kcells];

    const double dxi = 0.1;
    const double dyi = 0.1;
   
    init(ut, ncells);

    init(u, ncells);
    init(v, ncells);
    init(w, ncells);

    init(dzi, kcells);
    init(rhoref, kcells);
    init(rhorefh, kcells);

    // Send data to the GPU.
    #pragma acc enter data copyin(ut[0:ncells], u[0:ncells], v[0:ncells], w[0:ncells], dzi[0:kcells], rhoref[0:kcells], rhorefh[0:kcells])

    const int blocki = TILE_I;
    const int blockj = TILE_J;
    const int blockk = TILE_K;
    const int gridi = itot/blocki + (itot%blocki > 0);
    const int gridj = jtot/blockj + (jtot%blockj > 0);
    const int gridk = ktot/blockk + (ktot%blockk > 0);

    dim3 grid(gridi, gridj, gridk);
    dim3 block(blocki, blockj, blockk);

    // Check results
    advec_2i5<<<grid, block>>>(
            acc_to_cuda(ut), acc_to_cuda(u), acc_to_cuda(v), acc_to_cuda(w),
            acc_to_cuda(dzi), dxi, dyi,
            acc_to_cuda(rhoref), acc_to_cuda(rhorefh),
            istart, iend, jstart, jend, kstart, kend,
            jstride, kstride);
    cudaDeviceSynchronize();

    // Update the data.
    // #pragma acc update self(ut[0:ncells])
    // printf("ut=%.20f\n", ut[itot*jtot+itot+itot/2]);

    // Time performance
    cudaDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<nloop; ++i)
        advec_2i5<<<grid, block>>>(
                acc_to_cuda(ut), acc_to_cuda(u), acc_to_cuda(v), acc_to_cuda(w),
                acc_to_cuda(dzi), dxi, dyi,
                acc_to_cuda(rhoref), acc_to_cuda(rhorefh),
                istart, iend, jstart, jend, kstart, kend,
                jstride, kstride);

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    printf("time/iter = %E s (%i iters)\n",duration/(double)nloop, nloop);

    // Remove data from the GPU.
    #pragma acc exit data copyout(ut[0:ncells])

    // printf("ut=%.20f\n", ut[itot*jtot+itot+itot/4]);

    std::ofstream binary_file("ut_cuda.bin", std::ios::out | std::ios::trunc | std::ios::binary);

    if (binary_file)
        binary_file.write(reinterpret_cast<const char*>(ut), ncells*sizeof(double));
    else
    {
        std::string error = "Cannot write file \"ut_cuda.bin\"";
        throw std::runtime_error(error);
    }

    return 0;
}
