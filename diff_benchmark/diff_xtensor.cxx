#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <vector>

#include <xtensor/xarray.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

class Field3d
{
    public:
        Field3d(const size_t itot, const size_t jtot, const size_t ktot) :
            itot_(itot), jtot_(jtot), ktot_(ktot), data_(itot*jtot*ktot) {}

        inline double& operator()(const size_t i, const size_t j, const size_t k)
        {
            return data_[i + j*itot_ + k*itot_*jtot_];
        }

        inline double operator()(const size_t i, const size_t j, const size_t k) const
        {
            return data_[i + j*itot_ + k*itot_*jtot_];
        }

        inline double& operator[](const size_t n)
        {
            return data_[n];
        }

        inline double* data() { return data_.data(); }
        inline std::vector<double>& vector() { return data_; }

    private:
        const size_t itot_;
        const size_t jtot_;
        const size_t ktot_;
        std::vector<double> data_;
};

void init(double* const __restrict__ a, double* const __restrict__ at, const size_t ncells)
{
    for (size_t i=0; i<ncells; ++i)
    {
        a[i]  = pow(i,2)/pow(i+1,2);
        at[i] = 0.;
    }
}

void diff(std::vector<double>& at_v, std::vector<double>& a_v, const double visc, 
          const double dxidxi, const double dyidyi, const double dzidzi, 
          const size_t itot, const size_t jtot, const size_t ktot)
{
    const std::vector<std::size_t> shape = {ktot, jtot, itot};
    auto a = xt::adapt(a_v, shape);
    auto at = xt::adapt(at_v, shape);

    auto at_c = xt::view(at, xt::range(1, ktot-1), xt::range(1, jtot-1), xt::range(1, itot-1));

    auto a_c = xt::view(a, xt::range(1, ktot-1), xt::range(1, jtot-1), xt::range(1, itot-1));
    auto a_w = xt::view(a, xt::range(1, ktot-1), xt::range(1, jtot-1), xt::range(0, itot-2));
    auto a_e = xt::view(a, xt::range(1, ktot-1), xt::range(1, jtot-1), xt::range(2, itot  ));
    auto a_s = xt::view(a, xt::range(1, ktot-1), xt::range(0, jtot-2), xt::range(1, itot-1));
    auto a_n = xt::view(a, xt::range(1, ktot-1), xt::range(2, jtot  ), xt::range(1, itot-1));
    auto a_b = xt::view(a, xt::range(0, ktot-2), xt::range(1, jtot-1), xt::range(1, itot-1));
    auto a_t = xt::view(a, xt::range(2, ktot  ), xt::range(1, jtot-1), xt::range(1, itot-1));

    at_c += visc * ( (a_e - 2*a_c + a_w) * dxidxi
                   + (a_n - 2*a_c + a_s) * dyidyi
                   + (a_t - 2*a_c + a_b) * dzidzi );
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Add the grid size as an argument!" << std::endl;
        return 1;
    }

    const int nloop = 10;
    const size_t itot = std::stoi(argv[1]);
    const size_t jtot = std::stoi(argv[1]);
    const size_t ktot = std::stoi(argv[1]);
    const size_t ncells = itot*jtot*ktot;

    Field3d a (itot, jtot, ktot);
    Field3d at(itot, jtot, ktot);
   
    init(a.data(), at.data(), ncells);

    // Check results
    diff(at.vector(), a.vector(), 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot); 
    printf("at=%.20f\n",at[itot*jtot+itot+itot/2]);
 
    // Time performance 
    std::clock_t start = std::clock(); 
   
    for (int i=0; i<nloop; ++i)
        diff(at.vector(), a.vector(), 0.1, 0.1, 0.1, 0.1, itot, jtot, ktot); 
  
    double duration = (std::clock() - start ) / (double)CLOCKS_PER_SEC;
   
    printf("time/iter = %f s (%i iters)\n", duration/(double)nloop, nloop);
    
    return 0;
}
