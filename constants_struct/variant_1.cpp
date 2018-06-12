#include <iostream>

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

int main()
{
    Constants<double> constants;

    double theta;
    std::cin >> theta;

    std::cout << calculate_buoyancy<double>(theta, constants) << std::endl;

    return 0;
}
