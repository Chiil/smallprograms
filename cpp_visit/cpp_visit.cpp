#include <iostream>
#include <variant>


struct Advec_disabled {};
struct Advec_2 {};
struct Advec_2i5 {};


template<class... Ts> struct overload : Ts... { using Ts::operator()...; };
template<class... Ts> overload(Ts...) -> overload<Ts...>; // line not needed in C++20...


void advec_init(std::variant<Advec_disabled, Advec_2, Advec_2i5>& advec)
{
    std::visit(overload{
            []( Advec_disabled& ) { std::cout << "Disabled\n"; },
            []( Advec_2& ) { std::cout << "2\n"; },
            []( Advec_2i5& ) { std::cout << "2i5\n"; }}, advec);
};


int main(int argc, char** argv)
{
    std::variant<Advec_disabled, Advec_2, Advec_2i5> advec;

    if (std::stoi(argv[1]) == 0)
        advec = Advec_disabled();
    else if (std::stoi(argv[1]) == 1)
        advec = Advec_2();
    else if (std::stoi(argv[1]) == 2)
        advec = Advec_2i5();

    advec_init(advec);

    return 0;
}
