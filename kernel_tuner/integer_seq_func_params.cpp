#include <functional>
#include <iostream>
#include <array>


struct Print_double
{
    template<int I, int J, int K, int C0>
    static void exec(const double d)
    {
        std::cout << I << ", " << J << ", " << K << " = " << Print_double::alphas[C0]*d << std::endl;
    }

    static constexpr std::array<double, 3> alphas = {0.1, 0.2, 0.3};
};


template<class Func, int I, int J, int K, int C0, class... Args>
void exec_run(
        const std::array<int, 3> idx,
        Args... args)
{
    if (idx == std::array<int, 3>{I, J, K})
    {
        std::cout << "Running block ("
            << I << ", "
            << J << ", "
            << K << ")" << std::endl;
        Func::template exec<I, J, K, C0>(args...);
    }
}


template<class Func, int I, int J, int K, int... C0s, class... Args>
void inner_run_c0(
        const std::array<int, 3> idx,
        std::integer_sequence<int, C0s...> c0s,
        Args... args)
{
    (exec_run<Func, I, J, K, C0s>(idx, args...), ...);
}


template<class Func, int I, int J, int... Ks, int... C0s, class... Args>
void inner_run(
        const std::array<int, 3> idx,
        std::integer_sequence<int, Ks...> ks,
        std::integer_sequence<int, C0s...> c0s,
        Args... args)
{
    (inner_run_c0<Func, I, J, Ks>(idx, c0s, args...), ...);
}


template<class Func, int I, int... Js, int... Ks, int... C0s, class... Args>
void outer_run(
        const std::array<int, 3> idx,
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks,
        std::integer_sequence<int, C0s...> c0s,
        Args... args)
{
    (inner_run<Func, I, Js>(idx, ks, c0s, args...), ...);
}


template<class Func, int... Is, int... Js, int... Ks, int... C0s, class... Args>
void run(
        const std::array<int, 3> idx,
        std::integer_sequence<int, Is...> is,
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks,
        std::integer_sequence<int, C0s...> c0s,
        Args... args)
{
    (outer_run<Func, Is>(idx, js, ks, c0s, args...), ...);
}


int main()
{
    constexpr std::integer_sequence<int, 1, 2, 4, 8> is{};
    constexpr std::integer_sequence<int, 1, 2, 4> js{};
    constexpr std::integer_sequence<int, 32, 64> ks{};
    constexpr auto alphas{std::make_integer_sequence<int, Print_double::alphas.size()>{}};

    double d = 3.;

    std::array<int, 3> print_double_idx = {1, 2, 32};
    run<Print_double>(print_double_idx, is, js, ks, alphas, d);

    return 0;
}
