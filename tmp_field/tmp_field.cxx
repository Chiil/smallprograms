#include <vector>
#include <iostream>
#include <stdexcept>

class Tmp
{
    public:
        Tmp(double value) : value_(value), in_use_(false)
        {}

        double get_value() const
        {
            return value_;
        }

        bool get_in_use() const
        {
            return in_use_;
        }

        void use()
        {
            in_use_ = true;
        }

        void release()
        {
            in_use_ = false;
        }

    private:
        double value_;
        bool in_use_;
};

class Tmps
{
    public:
        Tmps() : tmp_list({1., 2})
        {}

        Tmp& get_tmp()
        {
            for (Tmp& tmp : tmp_list)
            {
                if (!tmp.get_in_use())
                {
                    tmp.use();
                    return tmp;
                }
            }
            throw std::runtime_error("No free tmp field");
        }

    private:
        std::vector<Tmp> tmp_list;
};

void test(Tmps& tmps)
{
    Tmp& tmp1 = tmps.get_tmp();
    Tmp& tmp2 = tmps.get_tmp();

    std::cout << "(test) tmp1, tmp2 = " << tmp1.get_value() << ", " << tmp2.get_value() << std::endl;

    tmp1.release();
    tmp2.release();
}

int main()
{
    Tmps tmps;
    try
    {
        test(tmps);

        Tmp& tmp1 = tmps.get_tmp();
        Tmp& tmp2 = tmps.get_tmp();

        std::cout << "(main) tmp1, tmp2 = " << tmp1.get_value() << ", " << tmp2.get_value() << std::endl;

        Tmp& tmp3 = tmps.get_tmp();
    }
    catch (std::exception& e)
    {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
