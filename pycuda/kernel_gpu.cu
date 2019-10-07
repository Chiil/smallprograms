template<typename TF>
__global__ void doublify(TF* a, const int itot, const int jtot)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    const int j = blockIdx.y*blockDim.y + threadIdx.y;
    const int k = blockIdx.z;

    int ijk = i + j*itot + k*itot*jtot;
    a[ijk] += TF(ijk);
}

template<typename TF>
void launch_doublify(TF* a, const int itot, const int jtot, const int ktot)
{
    dim3 grid_gpu (1, 1, ktot);
    dim3 block_gpu(itot, jtot, 1);

    doublify<<<grid_gpu, block_gpu>>>(a, itot, jtot);
}
