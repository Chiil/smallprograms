namespace nn
{
    template<typename TF>
    void inference_cpp(
            TF* __restrict__ ut, TF* __restrict__ vt, TF* __restrict__ wt,
            const TF* __restrict__ u, const TF* __restrict__ v, const TF* __restrict__ w,
            const int itot, const int jtot, const int ktot)
    {
        const int ii = 1;
        const int jj = itot;
        const int kk = itot*jtot;

        for (int k=1; k<ktot-1; k++)
            for (int j=1; j<jtot-1; j++)
                #pragma clang loop vectorize(enable)
                #pragma GCC ivdep
                for (int i=1; i<itot-1; i++)
                {
                    const int ijk = i + j*jj + k*kk;
                }
    }
}
