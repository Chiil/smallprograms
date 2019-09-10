#include <array>
#include <Eigen/Dense>

namespace nn
{
    constexpr int n0 = 375;
    constexpr int n1 = 80;

    template<typename TF>
    void inference_cpp(
            TF* __restrict__ ut, TF* __restrict__ vt, TF* __restrict__ wt,
            const TF* __restrict__ u, const TF* __restrict__ v, const TF* __restrict__ w,
            const TF* __restrict__ M0_raw, const TF* __restrict__ b0_raw,
            const int itot, const int jtot, const int ktot)
    {
        const int ii = 1;
        const int jj = itot;
        const int kk = itot*jtot;

        Eigen::Matrix<TF, Eigen::Dynamic, 1> v0(n0);
        Eigen::Matrix<TF, Eigen::Dynamic, Eigen::Dynamic> M0(n0, n1);
        Eigen::Matrix<TF, Eigen::Dynamic, 1> b0(n1);

        Eigen::Matrix<TF, Eigen::Dynamic, 1> v1(n1);
        Eigen::Matrix<TF, Eigen::Dynamic, Eigen::Dynamic> M1(n1, 18);
        Eigen::Matrix<TF, Eigen::Dynamic, 1> b1(18);

        Eigen::Matrix<TF, Eigen::Dynamic, 1> v2(18);

        for (int k=2; k<ktot-2; ++k)
            for (int j=2; j<jtot-2; ++j)
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
                                v0[iv  ] = u[ijkb];
                                v0[iv+1] = v[ijkb];
                                v0[iv+2] = w[ijkb];
                                iv += 3;
                            }

                    // Step 2. Execute the network.
                    // v1 = M0*v0 + b0;
                    #pragma GCC ivdep
                    for (int n=0; n<v1.rows(); ++n)
                        v1[n] = std::max(TF(0.2)*v1[n], v1[n]);

                    // v2 = M1*v1 + b1;
                    #pragma GCC ivdep
                    for (int n=0; n<v2.rows(); ++n)
                        v2[n] = std::max(TF(0.2)*v2[n], v2[n]);

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
