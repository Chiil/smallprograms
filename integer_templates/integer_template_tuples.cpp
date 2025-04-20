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
    std::make_tuple(2, 2, 2),
    std::make_tuple(3, 3, 3),
};


template<class F, std::size_t N, std::size_t... Is>
constexpr auto make_ijk_array(
        std::array<std::tuple<int, int, int>, N> tuple, std::index_sequence<Is...>)
{
    return std::array {
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
    if (i == j && i == k && i >= 1 && i <= tuples.size())
        func_table<F>[i-1]();
    else
        std::cerr << "Invalid combination\n";
}


int main()
{
    call_func<Func>(1, 1, 1);
    call_func<Func>(2, 2, 2);
    call_func<Func>(2, 2, 3);

    call_func<Func2>(3, 3, 3);
    call_func<Func2>(2, 2, 1);

    return 0;
}
