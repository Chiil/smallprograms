#include <cstdlib>
#include <cstdio>

template<typename TF>
struct Constants
{
    const TF g = 9.81;
    const TF cp = 1005.;
    const TF Rd = 287.;
    const TF theta_0 = 300.;
};

template<typename TF>
TF calculate_buoyancy(const TF theta, const Constants<TF>& constants)
{
    return constants.g/constants.theta_0 * theta;
}

int main(int argc, char* argv[])
{
    Constants<double> constants;

    double theta = std::atof(argv[1]);
    std::printf("%E\n", calculate_buoyancy(theta, constants));

    return 0;
}
