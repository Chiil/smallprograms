#include <vector>
#include <algorithm>

struct Boundary
{
    virtual ~Boundary() {};
    std::vector<double> a;
    std::vector<double> b;
};

struct BoundaryDefault : public Boundary
{
};

struct BoundarySurface : public Boundary
{
    std::vector<double> c;
};

void process_surface(BoundaryDefault& b)
{
}

void process_surface(BoundarySurface& b)
{
    for (double& d : b.c)
        d += 100.;
}

void process_boundaries(Boundary& b)
{
    for (double& d : b.a)
        d += 1.;
    for (double& d : b.b)
        d += 10.;

    if (auto b_cast = dynamic_cast<BoundaryDefault*>(&b))
        process_surface(*b_cast);
    else if (auto b_cast = dynamic_cast<BoundarySurface*>(&b))
        process_surface(*b_cast);
}

int  main()
{
    BoundaryDefault b_default;
    BoundarySurface b_surface;

    process_boundaries(b_default);
    process_boundaries(b_surface);

    return 0;
}
