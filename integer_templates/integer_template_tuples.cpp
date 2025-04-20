#include <array>
#include <iostream>
#include <utility>


struct Func
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
    std::make_tuple(2, 3, 4),
    std::make_tuple(3, 4, 5),
};


constexpr int max_i = 5;
using FuncType = void(*)();


template<class F, std::size_t N, std::size_t... Is>
constexpr auto make_ijk_array(
        std::array<std::tuple<int, int, int>, N> tuple, std::index_sequence<Is...>)
{
    return std::array<FuncType, sizeof...(Is)>{
        F::template run<std::get<0>(tuples[Is]), std::get<1>(tuples[Is]), std::get<2>(tuples[Is])>... };
}


template<class F, std::size_t N>
constexpr auto make_array(std::array<std::tuple<int, int, int>, N> tuples)
{
    return make_ijk_array<F>(tuples, std::make_index_sequence<N>{});
}


template<class F>
constexpr auto func_table = make_array<F>(tuples);


template<class F>
void call_func(int i, int j, int k)
{
    if (i >= 0 && i < max_i)
        func_table<F>[i]();
    else
        std::cerr << "Invalid combination\n";
}


int main()
{
    call_func<Func>(1, 1, 1);
    call_func<Func>(2, 2, 2);

    call_func<Func2>(1, 1, 1);
    call_func<Func2>(2, 2, 2);

    return 0;
}
