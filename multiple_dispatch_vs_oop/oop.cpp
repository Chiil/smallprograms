#include <vector>
#include <iostream>

struct Boundary
{
    Boundary(const int itot, const int jtot) : a(itot*jtot), b(itot*jtot) {}

    virtual void process_boundaries()
    {
        for (auto& d : a)
            d += 1;

        for (auto& d : b)
            d += 10;
    }

    virtual void print() const
    {
        for (const double d : a)
            std::cout << d << ", ";
        std::cout << std::endl;

        for (const double d : b)
            std::cout << d << ", ";
        std::cout << std::endl;
    }

    std::vector<double> a;
    std::vector<double> b;
};

struct Boundary_default : public Boundary
{
    Boundary_default(const int itot, const int jtot) : Boundary(itot, jtot) {}
    void process_boundaries() { Boundary::process_boundaries(); }
};

struct Boundary_surface : public Boundary
{
    Boundary_surface(const int itot, const int jtot) : Boundary(itot, jtot), c(itot*jtot) {}

    void process_boundaries()
    {
        Boundary::process_boundaries();
        for (auto& d : c)
            d += 100;
    }

    void print() const
    {
        Boundary::print();
        for (const double d : c)
            std::cout << d << ", ";
        std::cout << std::endl;
    }

    std::vector<double> c;
};

int main()
{
    Boundary_default b_default(2, 2);
    Boundary_surface b_surface(2, 2);

    b_default.process_boundaries();
    b_surface.process_boundaries();

    b_default.print();
    b_surface.print();

    return 0;
}
