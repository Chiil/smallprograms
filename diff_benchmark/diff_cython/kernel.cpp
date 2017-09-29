namespace kernels
{
    void diff(double* const __restrict__ at, const double* const __restrict__ a, const double visc, 
              const double dxidxi, const double dyidyi, const double dzidzi, 
              const int itot, const int jtot, const int ktot)
    {
        const int ii = 1;
        const int jj = itot;
        const int kk = itot*jtot;

        for (int k=1; k<ktot-1; k++)
            for (int j=1; j<jtot-1; j++)
                #pragma ivdep GCC
                for (int i=1; i<itot-1; i++)
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


}
