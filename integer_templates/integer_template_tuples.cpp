#include <array>
#include <iostream>
#include <utility>


struct Func1
{
    template<int I, int J, int K>
    static void run(double d)
    {
        std::cout << "func<" << I << ", " << J << ", " << K << ">(" << d << ")\n";
    }
};


struct Func2
{
    template<int I, int J, int K>
    static void run(const int a, const int b)
    {
        std::cout << "func2<" << I << ", " << J << ", " << K << ">(" << a << ", " << b << ")\n";
    }
};


// constexpr std::array tuples =
// {
//     std::make_tuple(1, 1, 1), // No refinement
//     std::make_tuple(2, 2, 2), std::make_tuple(3, 3, 3), // Equal refinement, 3D run
//     std::make_tuple(2, 2, 1), std::make_tuple(3, 3, 1), // Horizontal refinement, 3D run
//     std::make_tuple(2, 1, 2), std::make_tuple(3, 1, 3), // Equal refinement, 2D run
//     std::make_tuple(2, 1, 1), std::make_tuple(3, 1, 1), // Horizontal refinement, 2D run
//     std::make_tuple(2, 7, 4), // Some weird config
// };


template<int... Is>
constexpr auto make_tuples(std::integer_sequence<int, Is...>)
{
    return std::array {
        std::make_tuple(1, 1, 1),       // No refinement (1, 1, 1)
        std::make_tuple(Is, Is, Is)..., // Equal refinement, 3D run (2, 2, 2)
        std::make_tuple(Is, Is, 1)...,  // Horizontal refinement, 3D run (2, 2, 1)
        std::make_tuple(Is, 1, Is)...,  // Equal refinement, 2D run (2, 1, 2)
        std::make_tuple(Is, 1, 1)...,   // Horizontal refinement, 2D run (2, 1, 1)
    };
}

constexpr std::array tuples = make_tuples(std::integer_sequence<int, 2, 3, 4>{});


template<class F, int I, int J, int K, class... Args>
void run_tuples(int i, int j, int k, Args&&... args)
{
    if (i == I && j == J && k == K)
        F::template run<I, J, K>(args ...);
}


template<class F, class... Args>
void call(int i, int j, int k, Args&&... args)
{
    if (std::count(tuples.begin(), tuples.end(), std::make_tuple(i, j, k)) == 0)
        std::cerr << "Chosen refinement ratio (" << i << ", " << j << ", " << k << ") has no compiled function." << std::endl;
    else if (std::count(tuples.begin(), tuples.end(), std::make_tuple(i, j, k)) > 1)
        std::cerr << "Chosen refinement ratio (" << i << ", " << j << ", " << k << ") has more than one compiled function." << std::endl;

    auto loop_over_tuples = [i, j, k, args...]<std::size_t... Is>(std::index_sequence<Is...>)
    {
        (run_tuples<F, std::get<0>(tuples[Is]), std::get<1>(tuples[Is]), std::get<2>(tuples[Is])>(i, j, k, args...), ...);
    };

    loop_over_tuples(std::make_index_sequence<tuples.size()>());
}


int main()
{
    call<Func1>(1, 1, 1, 33.0);
    call<Func1>(2, 2, 2, 33.0);
    call<Func2>(2, 1, 1, 55, 66);
    call<Func2>(6, 6, 6, 55, 66);

    return 0;
}
