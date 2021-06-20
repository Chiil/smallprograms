#include <functional>
#include <iostream>


// 1D processing.
template<int I>
constexpr void print()
{
    std::cout << I << std::endl;
}


template<int... Is>
constexpr void test(
        std::integer_sequence<int, Is...> is)
{
    (print<Is>(), ...);
}
// End 1D.


// 2D processing.
template<int I, int J>
constexpr void print()
{
    std::cout << I << ", " << J << std::endl;
}


template<int I, int... Js>
constexpr void print(
        std::integer_sequence<int, Js...> js)
{
    (print<I, Js>(), ...);
}


template<int... Is, int... Js>
constexpr void test(
        std::integer_sequence<int, Is...> is,
        std::integer_sequence<int, Js...> js)
{
    (print<Is, Js...>(js), ...);
}
// End 2D.


// 3D processing.
template<int I, int J, int K>
constexpr void print()
{
    std::cout << I << ", " << J << ", " << K << std::endl;
}

template<int I, int J, int... Ks>
constexpr void print(
        std::integer_sequence<int, Ks...> ks)
{
    (print<I, J, Ks>(), ...);
}


template<int I, int... Js, int... Ks>
constexpr void print(
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks)
{
    (print<I, Js, Ks...>(ks), ...);
}


template<int... Is, int... Js, int... Ks>
constexpr void test(
        std::integer_sequence<int, Is...> is,
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks)
{
    (print<Is, Js..., Ks...>(js, ks), ...);
}
// End 3D.


int main()
{
    constexpr std::integer_sequence<int, 1, 2, 4, 8> is{};
    constexpr std::integer_sequence<int, 1, 2, 4> js{};
    constexpr std::integer_sequence<int, 32, 64> ks{};

    test(is); // 1D example.
    test(is, js); // 2D example.
    test(is, js, ks); // 3D example.

    return 0;
}
