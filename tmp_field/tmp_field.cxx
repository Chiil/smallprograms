#include <iostream>
#include <stdexcept>

class Tmps
{
    public:
        Tmps() : a(1.), b(2.),
                 a_in_use(false), b_in_use(false)
        {}

        double* get_tmp()
        {
            if (!a_in_use)
            {
                a_in_use = true;
                return &a;
            }
            else if (!b_in_use)
            {
                b_in_use = true;
                return &b;
            }
            else
                throw std::runtime_error("No free tmp field");
        }

    private:
        double a;
        double b;
        bool a_in_use;
        bool b_in_use;
};

void test(Tmps& tmps)
{
    double* tmp1 = tmps.get_tmp();
    double* tmp2 = tmps.get_tmp();

    std::cout << "tmp1, tmp2 = " << *tmp1 << ", " << *tmp2 << std::endl;
}

int main()
{
    Tmps tmps;
    try
    {
        test(tmps);

        double* tmp = tmps.get_tmp();
    }
    catch (std::exception& e)
    {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
