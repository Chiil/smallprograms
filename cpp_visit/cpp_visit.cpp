#include <iostream>
#include <variant>


struct Advec_disabled {};
struct Advec_2 {};
struct Advec_2i5 {};
using Advec = std::variant<Advec_disabled, Advec_2, Advec_2i5>;


template<class... Ts> struct overload : Ts... { using Ts::operator()...; };
template<class... Ts> overload(Ts...) -> overload<Ts...>; // line not needed in C++20...

void advec_init_disabled() { std::cout << "Disabled\n"; }
void advec_init_2(const int a) { std::cout << "2\n"; }
void advec_init_2i5(const int a, const int b) { std::cout << "2i5\n"; }


void advec_init(Advec& advec)
{
    std::visit(overload{
            []( Advec_disabled& ) { advec_init_disabled(); },
            []( Advec_2& )        { advec_init_2(1); },
            []( Advec_2i5& )      { advec_init_2i5(5, 7); }
        }, advec);
};


int main(int argc, char** argv)
{
    Advec advec;

    if (argc != 2)
        throw std::runtime_error("Only 1 argument allowed");

    if (std::stoi(argv[1]) == 0)
        advec = Advec_disabled();
    else if (std::stoi(argv[1]) == 1)
        advec = Advec_2();
    else if (std::stoi(argv[1]) == 2)
        advec = Advec_2i5();
    else
        throw std::runtime_error("Illegal value for advec");

    advec_init(advec);

    return 0;
}
