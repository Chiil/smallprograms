#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>


#define CUDA_MACRO

inline float interp2(const float a, const float b)
{
    return float(0.5) * (a + b);
}


inline float interp4_ws(const float a, const float b, const float c, const float d)
{
    constexpr float c0 = float(7./12.);
    constexpr float c1 = float(1./12.);
    return c0*(b+c) - c1*(a+d);
}


inline float interp3_ws(const float a, const float b, const float c, const float d)
{
    constexpr float c0 = float(3./12.);
    constexpr float c1 = float(1./12.);
    return c0*(c-b) - c1*(d-a);
}


inline float interp6_ws(
        const float a, const float b, const float c, const float d, const float e, const float f)
{
    constexpr float c0 = float(37./60.);
    constexpr float c1 = float(8./60.);
    constexpr float c2 = float(1./60.);

    return c0*(c+d) - c1*(b+e) + c2*(a+f);
}


inline float interp5_ws(
        const float a, const float b, const float c, const float d, const float e, const float f)
{
    constexpr float c0 = float(10./60.);
    constexpr float c1 = float(5./60.);
    constexpr float c2 = float(1./60.);

    return c0*(d-c) - c1*(e-b) + c2*(f-a);
}


void init_zero(float* const __restrict__ a, const int ncells)
{
    for (int n=0; n<ncells; ++n)
        a[n] = float(0.);
}


void init_rand(float* const __restrict__ a, const int ncells)
{
    for (int n=0; n<ncells; ++n)
        a[n] = float(std::rand() % 1000) + float(0.001);
}


void advec_2i5(
        float* const __restrict__ ut,
        const float* const __restrict__ u,
        const float* const __restrict__ v,
        const float* const __restrict__ w,
        const float* const __restrict__ dzi,
        const float dx, const float dy,
        const float* const __restrict__ rhoref,
        const float* const __restrict__ rhorefh,
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

    const float dxi = float(1.)/dx;
    const float dyi = float(1.)/dy;

    #pragma acc parallel present(ut, u, v, w, dzi, rhoref, rhorefh)
    {
        #pragma acc loop independent tile(TILE_I, TILE_J, TILE_K)
        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
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
                }

        // Vertical terms interior with full 5/6th order vertical
        #pragma acc loop independent tile(TILE_I, TILE_J, TILE_K)
        for (int k=kstart+3; k<kend-3; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj1 + k*kk1;
                    ut[ijk] +=
                            // w*du/dz
                            - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp6_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2], u[ijk+kk3])
                              - rhorefh[k  ] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp6_ws(u[ijk-kk3], u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2]) ) / rhoref[k] * dzi[k]

                            + ( rhorefh[k+1] * std::abs(interp2(w[ijk-ii1+kk1], w[ijk+kk1])) * interp5_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2], u[ijk+kk3])
                              - rhorefh[k  ] * std::abs(interp2(w[ijk-ii1    ], w[ijk    ])) * interp5_ws(u[ijk-kk3], u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2]) ) / rhoref[k] * dzi[k];
                }

        // Calculate vertical terms with reduced order near boundaries
        int k = kstart;
        #pragma acc loop independent tile(TILE_I, TILE_J)
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ijk = i + j*jj1 + k*kk1;
                ut[ijk] +=
                        // w*du/dz -> second order interpolation for fluxtop, fluxbot = 0. as w=0
                        - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp2(u[ijk    ], u[ijk+kk1]) ) / rhoref[k] * dzi[k];
            }

        k = kstart+1;
        #pragma acc loop independent tile(TILE_I, TILE_J)
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ijk = i + j*jj1 + k*kk1;
                ut[ijk] +=
                        // w*du/dz -> second order interpolation for fluxbot, fourth order for fluxtop
                        - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp4_ws(u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2])
                          - rhorefh[k  ] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp2(   u[ijk-kk1], u[ijk    ]) ) / rhoref[k] * dzi[k]

                        + ( rhorefh[k+1] * std::abs(interp2(w[ijk-ii1+kk1], w[ijk+kk1])) * interp3_ws(u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2]) ) / rhoref[k] * dzi[k];
            }


        k = kstart+2;
        #pragma acc loop independent tile(TILE_I, TILE_J)
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ijk = i + j*jj1 + k*kk1;
                ut[ijk] +=
                        // w*du/dz -> fourth order interpolation for fluxbot
                        - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp6_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2], u[ijk+kk3])
                          - rhorefh[k  ] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp4_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1]) ) / rhoref[k] * dzi[k]

                        + ( rhorefh[k+1] * std::abs(interp2(w[ijk-ii1+kk1], w[ijk+kk1])) * interp5_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2], u[ijk+kk3])
                          - rhorefh[k  ] * std::abs(interp2(w[ijk-ii1    ], w[ijk    ])) * interp3_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1]) ) / rhoref[k] * dzi[k];
            }

        k = kend-3;
        #pragma acc loop independent tile(TILE_I, TILE_J)
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ijk = i + j*jj1 + k*kk1;
                ut[ijk] +=
                        // w*du/dz -> fourth order interpolation for fluxtop
                        - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp4_ws(u[ijk-kk1   ], u[ijk    ], u[ijk+kk1], u[ijk+kk2])
                          - rhorefh[k  ] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp6_ws(u[ijk-kk3], u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2]) ) / rhoref[k] * dzi[k]

                        + ( rhorefh[k+1] * std::abs(interp2(w[ijk-ii1+kk1], w[ijk+kk1])) * interp3_ws(u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2])
                          - rhorefh[k  ] * std::abs(interp2(w[ijk-ii1    ], w[ijk    ])) * interp5_ws(u[ijk-kk3], u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1], u[ijk+kk2]) ) / rhoref[k] * dzi[k];
            }

        k = kend-2;
        #pragma acc loop independent tile(TILE_I, TILE_J)
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ijk = i + j*jj1 + k*kk1;
                ut[ijk] +=
                        // w*du/dz -> second order interpolation for fluxtop, fourth order for fluxbot
                        - ( rhorefh[k+1] * interp2(w[ijk-ii1+kk1], w[ijk+kk1]) * interp2(u[ijk    ], u[ijk+kk1])
                          - rhorefh[k  ] * interp2(w[ijk-ii1    ], w[ijk    ]) * interp4_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1]) ) / rhoref[k] * dzi[k]

                        - ( rhorefh[k  ] * std::abs(interp2(w[ijk-ii1    ], w[ijk    ])) * interp3_ws(u[ijk-kk2], u[ijk-kk1], u[ijk    ], u[ijk+kk1]) ) / rhoref[k] * dzi[k];
            }

        k = kend-1;
        #pragma acc loop independent tile(TILE_I, TILE_J)
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ijk = i + j*jj1 + k*kk1;
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

    // const int ijk_check = (istart + itot/2) + (jstart + jtot/2)*jstride + (kstart + ktot/2)*kstride;

    float* ut  = new float[ncells];
    float* u = new float[ncells];
    float* v = new float[ncells];
    float* w = new float[ncells];
    float* dzi  = new float[kcells];

    float* rhoref = new float[kcells];
    float* rhorefh = new float[kcells];

    const float dxi = 0.1;
    const float dyi = 0.1;
   
    init_zero(ut, ncells);

    std::srand(123);

    init_rand(u, ncells);
    init_rand(v, ncells);
    init_rand(w, ncells);

    init_rand(dzi, kcells);
    init_rand(rhoref, kcells);
    init_rand(rhorefh, kcells);

    // Send data to the GPU.
    #pragma acc enter data copyin(ut[0:ncells], u[0:ncells], v[0:ncells], w[0:ncells], dzi[0:kcells], rhoref[0:kcells], rhorefh[0:kcells])

    // Check results
    advec_2i5(
            ut, u, v, w,
            dzi, dxi, dyi,
            rhoref, rhorefh,
            istart, iend, jstart, jend, kstart, kend,
            jstride, kstride);

    // Update the data.
    // #pragma acc update self(ut[0:ncells])
    // printf("ut=%.20f\n", ut[ijk_check]);

    // Time performance
    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<nloop; ++i)
        advec_2i5(
                ut, u, v, w,
                dzi, dxi, dyi,
                rhoref, rhorefh,
                istart, iend, jstart, jend, kstart, kend,
                jstride, kstride);

    auto end = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration_cast<std::chrono::duration<float>>(end - start).count();

    printf("time/iter = %E s (%i iters)\n",duration/(float)nloop, nloop);

    // Remove data from the GPU.
    #pragma acc exit data copyout(ut[0:ncells]) delete(u[0:ncells], v[0:ncells], w[0:ncells], dzi[0:kcells], rhoref[0:kcells], rhorefh[0:kcells])

    std::ofstream binary_file("ut_acc.bin", std::ios::out | std::ios::trunc | std::ios::binary);

    if (binary_file)
        binary_file.write(reinterpret_cast<const char*>(ut), ncells*sizeof(float));
    else
    {
        std::string error = "Cannot write file \"at_acc.bin\"";
        throw std::runtime_error(error);
    }

    return 0;
}
