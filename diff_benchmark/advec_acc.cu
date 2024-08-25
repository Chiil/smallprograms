#include <iostream>
#include <fstream>
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


void diff(
        float* const __restrict__ at, const float* const __restrict__ a,
        const float* const __restrict__ u, const float* const __restrict__ v, const float* const __restrict__ w,
        const float dxi, const float dyi, const float dzi, 
        const int itot, const int jtot, const int ktot)
{
    const int ii = 1;
    const int jj = itot;
    const int kk = itot*jtot;

    #pragma acc parallel loop present(a, at, u, v, w) collapse(3)
    for (int k=1; k<ktot-1; ++k)
        for (int j=1; j<jtot-1; ++j)
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

    float *a_cuda;
    float *u_cuda;
    float *v_cuda;
    float *w_cuda;
    float *at_cuda;

    cudaMalloc(&a_cuda, ncells*sizeof(float));
    cudaMalloc(&u_cuda, ncells*sizeof(float));
    cudaMalloc(&v_cuda, ncells*sizeof(float));
    cudaMalloc(&w_cuda, ncells*sizeof(float));
    cudaMalloc(&at_cuda, ncells*sizeof(float));

    cudaMemcpy(a_cuda, a, ncells*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(u_cuda, u, ncells*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(v_cuda, v, ncells*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(w_cuda, w, ncells*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(at_cuda, at, ncells*sizeof(float), cudaMemcpyHostToDevice);

    // Check results
    diff(
            at_cuda, a_cuda,
            u_cuda, v_cuda, w_cuda,
            0.1f, 0.1f, 0.1f,
            itot, jtot, ktot);
 
    cudaMemcpy(at, at_cuda, ncells*sizeof(float), cudaMemcpyDeviceToHost);

    printf("at=%.20f\n",at[itot*jtot+itot+itot/2]);
 
    // Time performance 
    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<nloop; ++i)
        diff(
                at_cuda, a_cuda,
                u_cuda, v_cuda, w_cuda,
                0.1f, 0.1f, 0.1f,
                itot, jtot, ktot);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    printf("time/iter = %E s (%i iters)\n",duration/(double)nloop, nloop);

    cudaMemcpy(at, at_cuda, ncells*sizeof(float), cudaMemcpyDeviceToHost);

    printf("at=%.20f\n", at[itot*jtot+itot+itot/4]);

    return 0;
}
