#include <array>
#include <iostream>
#include <utility>


template<int I, int J, int K>
void func()
{
    std::cout << "func<" << I << ", " << J << ", " << K << ">()\n";
}


constexpr std::array tuples =
{
    std::make_tuple(1, 1, 1),
    std::make_tuple(2, 3, 4),
    std::make_tuple(3, 4, 5),
};


constexpr int max_i = 5;
using FuncType = void(*)();


template<std::size_t N, std::size_t... Is>
constexpr auto make_ijk_array(
        std::array<std::tuple<int, int, int>, N> tuple, std::index_sequence<Is...>)
{
    return std::array<FuncType, sizeof...(Is)>{
        &func<std::get<0>(tuples[Is]), std::get<1>(tuples[Is]), std::get<2>(tuples[Is])>... };
}


template<std::size_t N>
constexpr auto make_array(std::array<std::tuple<int, int, int>, N> tuples)
{
    return make_ijk_array(tuples, std::make_index_sequence<N>{});
}


constexpr auto func_table = make_array(tuples);


void call_func(int i, int j, int k)
{
    if (i >= 0 && i < max_i)
        func_table[i]();
    else
        std::cerr << "Invalid combination\n";
}


int main()
{
    call_func(1, 1, 1);
    call_func(2, 2, 2);

    return 0;
}
