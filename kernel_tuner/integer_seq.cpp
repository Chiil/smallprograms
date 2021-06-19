#include <functional>
#include <iostream>


template<int I>
void print()
{
    std::cout << I << std::endl;
}


template<int I, int J>
void print()
{
    std::cout << I << ", " << J << std::endl;
}


template<int I, int... Js>
void print(std::integer_sequence<int, Js...>)
{
    (print<I, Js>(), ...);
}


template<int... Is>
void test(std::integer_sequence<int, Is...> is)
{
    (print<Is>(), ...);
}


template<int... Is, int... Js>
void test(std::integer_sequence<int, Is...> is, std::integer_sequence<int, Js...> js)
{
    (print<Is, Js...>(js), ...);
}


int main()
{
    constexpr std::integer_sequence<int, 1, 2, 4, 8> is{};
    constexpr std::integer_sequence<int, 1, 2, 4> js{};

    test(is, js);

    test(is);

    return 0;
}
