#include <functional>
#include <iostream>


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


template<class Func, int I, int J, int... Ks, class... Args>
void inner(
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    (Func::template exec<I, J, Ks>(args...), ...);
}


template<class Func, int I, int... Js, int... Ks, class... Args>
void outer(
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    (inner<Func, I, Js>(ks, args...), ...);
}


template<class Func, int... Is, int... Js, int... Ks, class... Args>
void test(
        std::integer_sequence<int, Is...> is,
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    (outer<Func, Is>(js, ks, args...), ...);
}


int main()
{
    constexpr std::integer_sequence<int, 1, 2, 4, 8> is{};
    constexpr std::integer_sequence<int, 1, 2, 4> js{};
    constexpr std::integer_sequence<int, 32, 64> ks{};

    double d = 3.;

    test<Print>(is, js, ks);
    test<Print_double>(is, js, ks, d);

    return 0;
}
