#include <vector>
#include <iostream>
#include <stdexcept>

class Tmp
{
    public:
        Tmp(double value) : value_(value), in_use_(false) {}
        double get_value() const { return value_; }
        bool get_in_use() const { return in_use_; }
        void use() { in_use_ = true; }
        void release() { in_use_ = false; }

    private:
        double value_;
        bool in_use_;
};

class Tmp_ptr
{
    public:
        Tmp_ptr(Tmp& tmp) : tmp_(tmp) { tmp_.use(); }
        ~Tmp_ptr() { tmp_.release(); }
        Tmp* operator->() { return &tmp_; }

        Tmp_ptr(const Tmp_ptr&) = delete;
        Tmp_ptr operator=(const Tmp_ptr&) = delete;

        Tmp_ptr(Tmp_ptr&&) = default;

    private:
        Tmp& tmp_;
};

class Tmps
{
    public:
        Tmps() : tmp_list({1., 2., 3.})
        {}

        Tmp_ptr get_tmp()
        {
            for (Tmp& tmp : tmp_list)
                if (!tmp.get_in_use())
                    return Tmp_ptr(tmp);

            throw std::runtime_error("No free tmp field");
        }

    private:
        std::vector<Tmp> tmp_list;
};

void test(Tmps& tmps)
{
    Tmp_ptr tmp1 = tmps.get_tmp();
    Tmp_ptr tmp2 = tmps.get_tmp();

    std::cout << "(test) tmp1, tmp2 = " << tmp1->get_value() << ", " << tmp2->get_value() << std::endl;
}

int main()
{
    Tmps tmps;
    try
    {
        Tmp_ptr tmp1 = tmps.get_tmp();

        test(tmps);

        Tmp_ptr tmp2 = tmps.get_tmp();
        Tmp_ptr tmp3 = tmps.get_tmp();

        std::cout << "(main) tmp1, tmp2, tmp3 = " << tmp1->get_value() << ", "
                                                  << tmp2->get_value() << ", "
                                                  << tmp3->get_value() << std::endl;

        Tmp_ptr tmp4 = tmps.get_tmp();
    }
    catch (std::exception& e)
    {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
