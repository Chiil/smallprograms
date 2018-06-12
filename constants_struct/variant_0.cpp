#include <iostream>

template<typename TF>
TF calculate_buoyancy(const TF theta)
{
    const TF g = 9.81;
    const TF theta_0 = 300.;
    return g/theta_0 * theta;
}

int main()
{
    double theta;
    std::cin >> theta;

    std::cout << calculate_buoyancy<double>(theta) << std::endl;

    return 0;
}
