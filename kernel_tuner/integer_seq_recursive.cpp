#include <functional>
#include <iostream>
#include <tuple>


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


template<int I, int J, int K>
void print()
{
    std::cout << I << ", " << J << ", " << K << std::endl;
}


template<int I=-1, int J=-1, int... Ks, class... Args>
void print(
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    if constexpr (I == -1 && J == -1)
        (print<Ks>(args...), ...);
    else if (I == -1 && J >= 0)
        (print<J, Ks>(args...), ...);
    else if (I >= 0 && J == -1)
        (print<I, Ks>(args...), ...);
    else
        (print<I, J, Ks>(args...), ...);
}


template<int I=-1, int... Js, int... Ks, class... Args>
void print(
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    (print<I, Js>(ks, args...), ...);
}


template<int... Is, int... Js, int... Ks, class... Args>
void print(
        std::integer_sequence<int, Is...> is,
        std::integer_sequence<int, Js...> js,
        std::integer_sequence<int, Ks...> ks,
        Args... args)
{
    (print<Is>(js, ks, args...), ...);
}


int main()
{
    constexpr std::integer_sequence<int, 0, 2, 4, 8> is{};
    constexpr std::integer_sequence<int, 0, 2> js{};
    constexpr std::integer_sequence<int, 0, 4, 8> ks{};

    std::cout << "(is):" << std::endl;
    print(is);

    std::cout << "(is, js):" << std::endl;
    print(is, js);

    std::cout << "(js, ks):" << std::endl;
    print(js, ks);

    std::cout << "(is, js, ks):" << std::endl;
    print(is, js, ks);

    return 0;
}
