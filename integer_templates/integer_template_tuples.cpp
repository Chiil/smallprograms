#include <array>
#include <iostream>
#include <utility>


struct Func1
{
    template<int I, int J, int K>
    static void run()
    {
        std::cout << "func<" << I << ", " << J << ", " << K << ">()\n";
    }
};


struct Func2
{
    template<int I, int J, int K>
    static void run()
    {
        std::cout << "func2<" << I << ", " << J << ", " << K << ">()\n";
    }
};


constexpr std::array tuples =
{
    std::make_tuple(1, 1, 1),
    std::make_tuple(2, 2, 2), std::make_tuple(3, 3, 3), // Equal refinement, 3D run
    std::make_tuple(2, 2, 1), std::make_tuple(3, 3, 1), // Horizontal nesting, 3D run
    std::make_tuple(2, 1, 2), std::make_tuple(3, 1, 3), // Equal refinement, 2D run
    std::make_tuple(2, 1, 1), std::make_tuple(3, 1, 1), // Horizontal nesting, 2D run
};


template<class F, int I, int J, int K>
void run_tuples(int i, int j, int k)
{
    if (i == I && j == J && k == K)
        F::template run<I, J, K>();
}


template<class F, std::size_t... Is>
void call(int i, int j, int k, std::index_sequence<Is...> is)
{
    (run_tuples<F, std::get<0>(tuples[Is]), std::get<1>(tuples[Is]), std::get<2>(tuples[Is])>(i, j, k), ...);
}


int main()
{
    call<Func1>(1, 1, 1, std::make_index_sequence<tuples.size()>());
    call<Func2>(2, 1, 1, std::make_index_sequence<tuples.size()>());

    return 0;
}
