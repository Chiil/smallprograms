#include <cstdlib>
#include <cstdio>

template<typename TF>
TF calculate_buoyancy(const TF theta)
{
    const TF g = 9.81;
    const TF theta_0 = 300.;
    return g/theta_0 * theta;
}

int main(int argc, char* argv[])
{
    double theta = std::atof(argv[1]);
    std::printf("%E\n", calculate_buoyancy(theta));

    return 0;
}
