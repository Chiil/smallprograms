#include <array>

namespace nn
{
    template<typename TF>
    void inference_cpp(
            TF* __restrict__ ut, TF* __restrict__ vt, TF* __restrict__ wt,
            const TF* __restrict__ u, const TF* __restrict__ v, const TF* __restrict__ w,
            const TF* __restrict__ M0, const TF* __restrict__ b0,
            const int itot, const int jtot, const int ktot)
    {
        const int ii = 1;
        const int jj = itot;
        const int kk = itot*jtot;

        std::array<TF, 375> v0;

        for (int k=2; k<ktot-2; ++k)
            for (int j=2; j<jtot-2; ++j)
                for (int i=2; i<itot-2; ++i)
                {
                    // Step 1. Fill the block.
                    int iv = 0;
                    for (int kb=-2; kb<3; ++kb)
                        for (int jb=-2; jb<3; ++jb)
                            #pragma GCC ivdep
                            for (int ib=-2; ib<3; ++ib)
                            {
                                const int ijk = (i+ib) + (j+jb)*jj + (k+kb)*kk;
                                v0[iv  ] = u[ijk];
                                v0[iv+1] = v[ijk];
                                v0[iv+2] = w[ijk];
                                iv += 3;
                            }

                    // Step 2. Execute the network.
                    #pragma GCC ivdep
                    for (int n=0; n<375; ++n)
                        v0[n] += b0[n];

                    // Step 3. Read out the block.
                    iv = 0;
                    for (int kb=-2; kb<3; ++kb)
                        for (int jb=-2; jb<3; ++jb)
                            #pragma GCC ivdep
                            for (int ib=-2; ib<3; ++ib)
                            {
                                const int ijk = (i+ib) + (j+jb)*jj + (k+kb)*kk;
                                ut[ijk] = v0[iv  ];
                                vt[ijk] = v0[iv+1];
                                wt[ijk] = v0[iv+2];
                                iv += 3;
                            }
                }
    }
}
