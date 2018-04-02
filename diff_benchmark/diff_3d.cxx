#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <vector>
#include "math.h"

class Field3d
{
    public:
        Field3d(const int itot, const int jtot, const int ktot) :
            itot_(itot), jtot_(jtot), ktot_(ktot), data_(itot*jtot*ktot) {}

        inline double& operator()(const int i, const int j, const int k)
        {
            return data_[i + j*itot_ + k*itot_*jtot_];
        }

        inline double operator()(const int i, const int j, const int k) const
        {
            return data_[i + j*itot_ + k*itot_*jtot_];
        }

        inline double& operator[](const int n)
        {
            return data_[n];
        }

        inline double* data() { return data_.data(); }

    private:
        const int itot_;
        const int jtot_;
        const int ktot_;
        std::vector<double> data_;
};

void init(double* const __restrict__ a, double* const __restrict__ at, const int ncells)
{
    for (int i=0; i<ncells; ++i)
    {
        a[i]  = pow(i,2)/pow(i+1,2);
        at[i] = 0.;
    }
}

void diff(Field3d& at, const Field3d& a, const double visc, 
          const double dxidxi, const double dyidyi, const double dzidzi, 
          const int itot, const int jtot, const int ktot)
{
    for (int k=1; k<ktot-1; k++)
        for (int j=1; j<jtot-1; j++)
            #pragma GCC ivdep
            for (int i=1; i<itot-1; i++)
            {
                at(i, j, k) += visc * (
                        + ( (a(i+1, j, k) - a(i  , j, k))
                          - (a(i  , j, k) - a(i-1, j, k)) ) * dxidxi
                        + ( (a(i, j+1, k) - a(i, j  , k))
                          - (a(i, j  , k) - a(i, j-1, k)) ) * dyidyi
                        + ( (a(i, j, k+1) - a(i, j, k  ))
                          - (a(i, j, k  ) - a(i, j, k-1)) ) * dzidzi
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

    const int nloop = 10;
    const int itot = std::stoi(argv[1]);
    const int jtot = std::stoi(argv[1]);
    const int ktot = std::stoi(argv[1]);
    const int ncells = itot*jtot*ktot;

    Field3d a (itot, jtot, ktot);
    Field3d at(itot, jtot, ktot);
   
    init(a.data(), at.data(), ncells);

    // Check results
    diff(at, a, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot); 
    printf("at=%.20f\n",at[itot*jtot+itot+itot/2]);
 
    // Time performance 
    std::clock_t start = std::clock(); 
   
    for (int i=0; i<nloop; ++i)
        diff(at, a, 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot); 
  
    double duration = (std::clock() - start ) / (double)CLOCKS_PER_SEC;
   
    printf("time/iter = %f s (%i iters)\n",duration/(double)nloop, nloop);
    
    return 0;
}
