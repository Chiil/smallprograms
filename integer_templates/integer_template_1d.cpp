#include <array>
#include <iostream>
#include <utility>


template<int I>
void func()
{
    std::cout << "func<" << I << ">()\n";
}


constexpr int max_i = 5;
using FuncType = void(*)();


// Build 1D array for k
template<int... Ks>
constexpr auto make_array(std::integer_sequence<int, Ks...>) {
    return std::array<FuncType, sizeof...(Ks)>{ &func<Ks>... };
}


// Generate the full table
constexpr auto func_table = make_array(std::make_integer_sequence<int, max_i>{});


void call_func(int i, int j, int k)
{
    if (i >= 0 && i < max_i)
        func_table[i]();
    else
        std::cerr << "Invalid combination\n";
}


int main()
{
    call_func(2, 3, 4);
    call_func(1, 0, 0);
    return 0;
}
