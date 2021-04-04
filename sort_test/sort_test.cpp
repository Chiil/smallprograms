#include <algorithm>
#include <execution>
#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include <iomanip>
#include <boost/sort/spreadsort/integer_sort.hpp>

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cout << "Specify itot and nphotons." << std::endl;
        return 1;
    }

    const int itot = std::stoi(argv[1]);
    const int nphotons = std::stoi(argv[2]);
 
    std::vector<double> a(itot*itot*itot);

    std::vector<int> indices(nphotons);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0, itot*itot*itot-1);

    for (int& i : indices)
        i = distribution(generator);

    // Do the calculation unsorted.
    {
        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (const int i : indices)
        {
            a[i] += i;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        std::cout << "Duration unsorted: " << std::scientific << std::setprecision(4) << duration << std::endl;
    }

    // Do the sorting.
    {
        auto start_sort = std::chrono::high_resolution_clock::now();

        // std::sort(std::execution::par, indices.begin(), indices.end());
        boost::sort::spreadsort::integer_sort(indices.begin(), indices.end());

        auto end_sort = std::chrono::high_resolution_clock::now();
        double duration_sort = std::chrono::duration_cast<std::chrono::duration<double>>(end_sort - start_sort).count();

        std::cout << "Duration sorting: " << std::scientific << std::setprecision(4) << duration_sort << std::endl;

        auto start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (const int i : indices)
        {
            a[i] += i;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        std::cout << "Duration sorted (part, total): " << std::scientific << std::setprecision(4) << duration 
            << ", " << duration + duration_sort << std::endl;
    }

    return 0;
}
