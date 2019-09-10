#include <array>
#include <cblas.h>

namespace nn
{
    extern "C" void cblas_dgemv(
            const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE, const int, const int, const double, const double*,
            const int, const double*, const int, const double, double*, const int);

    constexpr int n0 = 375;
    constexpr int n1 = 80;
    constexpr int n2 = 18;

    template<typename TF>
    void add_bias_and_activate(TF* __restrict__ v, const TF* __restrict__ b, const int v_length)
    {
        #pragma GCC ivdep
        for (int n=0; n<v_length; ++n)
        {
            const TF v_tmp = v[n] + b[n];
            v[n] = std::max(TF(0.2)*v_tmp, v_tmp);
        }
    }

    template<typename TF>
    void inference_cpp(
            TF* __restrict__ ut, TF* __restrict__ vt, TF* __restrict__ wt,
            const TF* __restrict__ u, const TF* __restrict__ v, const TF* __restrict__ w,
            const TF* __restrict__ M0, const TF* __restrict__ b0,
            const TF* __restrict__ M1, const TF* __restrict__ b1,
            const int itot, const int jtot, const int ktot)
    {
        const int ii = 1;
        const int jj = itot;
        const int kk = itot*jtot;

        std::array<TF, n0> v0;
        std::array<TF, n1> v1;
        std::array<TF, n2> v2;

        for (int k=2; k<ktot-2; ++k)
            for (int j=2; j<jtot-2; ++j)
                #pragma GCC ivdep
                for (int i=2; i<itot-2; ++i)
                {
                    // Calculate the local index.
                    const int ijk = i + j*jj + k*kk;

                    // Step 1. Fill the block.
                    int iv = 0;
                    for (int kb=-2; kb<3; ++kb)
                        for (int jb=-2; jb<3; ++jb)
                            #pragma GCC ivdep
                            for (int ib=-2; ib<3; ++ib)
                            {
                                const int ijkb = (i+ib) + (j+jb)*jj + (k+kb)*kk;
                                v0[iv] = u[ijkb];
                                ++iv;

                                v0[iv] = v[ijkb];
                                ++iv;

                                v0[iv] = w[ijkb];
                                ++iv;
                            }

                    // Step 2. Execute the network.
                    // v1 = M0*v0 + b0;
                    cblas_dgemv(CblasRowMajor, CblasNoTrans, n1, n0, 1., M0, n0, v0.data(), 1, 0., v1.data(), 1);
                    add_bias_and_activate(v1.data(), b0, n1);

                    // v2 = M1*v1 + b1;
                    cblas_dgemv(CblasRowMajor, CblasNoTrans, n2, n1, 1., M1, n1, v1.data(), 1, 0., v2.data(), 1);
                    add_bias_and_activate(v2.data(), b1, n2);

                    // Step 3. Read out the vector.
                    ut[ijk   ] += v0[ 0];
                    ut[ijk+ii] += v0[ 1];
                    ut[ijk   ] += v0[ 2];
                    ut[ijk+jj] += v0[ 3];
                    ut[ijk   ] += v0[ 4];
                    ut[ijk+kk] += v0[ 5];

                    vt[ijk   ] += v0[ 6];
                    vt[ijk+ii] += v0[ 7];
                    vt[ijk   ] += v0[ 8];
                    vt[ijk+jj] += v0[ 9];
                    vt[ijk   ] += v0[10];
                    vt[ijk+kk] += v0[11];

                    wt[ijk   ] += v0[12];
                    wt[ijk+ii] += v0[13];
                    wt[ijk   ] += v0[14];
                    wt[ijk+jj] += v0[15];
                    wt[ijk   ] += v0[16];
                    wt[ijk+kk] += v0[17];
                }
    }
}
