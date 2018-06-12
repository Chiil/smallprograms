#include <cstdlib>
#include <cstdio>

template<typename TF>
struct Constants
{
    static constexpr TF g = 9.81;
    static constexpr TF cp = 1005.;
    static constexpr TF Rd = 287.;
    static constexpr TF theta_0 = 300.;
};

template<typename TF>
TF calculate_buoyancy(const TF theta, const Constants<TF> constants)
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
