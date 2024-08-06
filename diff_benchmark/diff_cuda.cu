#include <iostream>
#include <fstream>
#include <cstdio>
#include <chrono>
#include <cmath>

void init(float* const __restrict__ a, float* const __restrict__ at, const int ncells)
{
    for (int i=0; i<ncells; ++i)
    {
        a[i] = pow(i,2)/pow(i+1,2);
        at[i] = 0.;
    }
}

__global__ void diff(
        float* const __restrict__ at, const float* const __restrict__ a,
        const float visc, const float dxidxi, const float dyidyi, const float dzidzi, 
        const int itot, const int jtot, const int ktot)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x + 1;
    const int j = blockIdx.y*blockDim.y + threadIdx.y + 1;
    const int k = blockIdx.z + 1;

    const int ii = 1;
    const int jj = itot;
    const int kk = itot*jtot;

    if (i < itot-1 && j < jtot-1 && k < ktot-1)
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

    float *a  = new float[ncells];
    float *at = new float[ncells];
   
    init(a, at, ncells);

    float *a_cuda;
    float *at_cuda;

    cudaMalloc(&a_cuda, ncells*sizeof(float));
    cudaMalloc(&at_cuda, ncells*sizeof(float));

    cudaMemcpy(a_cuda, a, ncells*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(at_cuda, at, ncells*sizeof(float), cudaMemcpyHostToDevice);

    const int blocki = 64;
    const int blockj = 1;
    const int gridi = (itot-2)/blocki + ((itot-2)%blocki > 0);
    const int gridj = (jtot-2)/blockj + ((jtot-2)%blockj > 0);

    dim3 grid_gpu(gridi, gridj, ktot);
    dim3 block_gpu(blocki, blockj, 1);

    // Check results
    diff<<<grid_gpu, block_gpu>>>(
            at_cuda, a_cuda,
            0.1, 0.1, 0.1, 0.1,
            itot, jtot, ktot);
 
    cudaMemcpy(at, at_cuda, ncells*sizeof(float), cudaMemcpyDeviceToHost);

    printf("at=%.20f\n",at[itot*jtot+itot+itot/2]);
 
    // Time performance 
    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<nloop; ++i)
        diff<<<grid_gpu, block_gpu>>>(
                at_cuda, a_cuda,
                0.1, 0.1, 0.1, 0.1,
                itot, jtot, ktot);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    printf("time/iter = %E s (%i iters)\n",duration/(double)nloop, nloop);

    cudaMemcpy(at, at_cuda, ncells*sizeof(float), cudaMemcpyDeviceToHost);

    printf("at=%.20f\n", at[itot*jtot+itot+itot/4]);

    /*
    std::ofstream binary_file("at_cuda.bin", std::ios::out | std::ios::trunc | std::ios::binary);

    if (binary_file)
        binary_file.write(reinterpret_cast<const char*>(at), ncells*sizeof(float));
    else
    {
        std::string error = "Cannot write file \"at_cuda.bin\"";
        throw std::runtime_error(error);
    }
    */

    return 0;
}
