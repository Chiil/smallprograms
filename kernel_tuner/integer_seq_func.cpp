#include <functional>
#include <iostream>
#include <array>


// Test functions.
struct Print
{
    template<int I, int J, int K>
    static void exec()
    {
        std::cout << I << ", " << J << ", " << K << std::endl;
    }
};


struct Print_double
{
    template<int I, int J, int K>
    static void exec(const double d)
    {
        std::cout << I << ", " << J << ", " << K << " = " << d << std::endl;
    }
};


// Tuner.
template<class Func, int I, int J, int K, class... Args>
void exec(
        std::array<int, 3>& fastest_idx,
        double& fastest,
        Args... args)
{
    auto start = std::chrono::high_resolution_clock::now();
    Func::template exec<I, J, K>(args...);
    auto end = std::chrono::high_resolution_clock::now();
    double duration =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    if (duration < fastest)
    {
        fastest = duration;
        fastest_idx = std::array<int, 3>{I, J, K};
    }
}


template<class Func, int I, int J, int... Ks, class... Args>
void inner(
        std::array<int, 3>& fastest_idx,
        double& fastest,
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    (exec<Func, I, J, Ks>(fastest_idx, fastest, args...), ...);
}


template<class Func, int I, int... Js, int... Ks, class... Args>
void outer(
        std::array<int, 3>& fastest_idx,
        double& fastest,
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    (inner<Func, I, Js>(fastest_idx, fastest, ks, args...), ...);
}


template<class Func, int... Is, int... Js, int... Ks, class... Args>
std::array<int, 3> tune(
        std::integer_sequence<int, Is...> is,
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    std::array<int, 3> fastest_idx{};
    double fastest = 1.e100;
    (outer<Func, Is>(fastest_idx, fastest, js, ks, args...), ...);

    std::cout << "Fastest block: ("
        << fastest_idx[0] << ", "
        << fastest_idx[1] << ", "
        << fastest_idx[2] << ") = "
        << fastest << " (s) " << std::endl;

    return fastest_idx;
}


// Executor.
template<class Func, int I, int J, int K, class... Args>
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
        Func::template exec<I, J, K>(args...);
    }
}


template<class Func, int I, int J, int... Ks, class... Args>
void inner_run(
        const std::array<int, 3> idx,
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    (exec_run<Func, I, J, Ks>(idx, args...), ...);
}


template<class Func, int I, int... Js, int... Ks, class... Args>
void outer_run(
        const std::array<int, 3> idx,
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    (inner_run<Func, I, Js>(idx, ks, args...), ...);
}
template<class Func, int... Is, int... Js, int... Ks, class... Args>
void run(
        const std::array<int, 3> idx,
        std::integer_sequence<int, Is...> is,
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    (outer_run<Func, Is>(idx, js, ks, args...), ...);
}


int main()
{
    constexpr std::integer_sequence<int, 1, 2, 4, 8> is{};
    constexpr std::integer_sequence<int, 1, 2, 4> js{};
    constexpr std::integer_sequence<int, 32, 64> ks{};

    double d = 3.;

    auto print_idx = tune<Print>(is, js, ks);
    auto print_double_idx = tune<Print_double>(is, js, ks, d);

    run<Print_double>(print_double_idx, is, js, ks, d);

    return 0;
}
