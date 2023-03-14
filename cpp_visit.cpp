#include <iostream>
#include <variant>


struct Advec_disabled {};
struct Advec_2 {};
struct Advec_2i5 {};


struct advection_init
{
    void operator()( Advec_disabled& ) { std::cout << "Disabled\n"; }
    void operator()( Advec_2& ) { std::cout << "2\n"; }
    void operator()( Advec_2i5& ) { std::cout << "2i5\n"; }
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

    std::visit(advection_init(), advec);

    return 0;
}
