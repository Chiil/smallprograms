#include <iostream>

struct Test
{
    Test() { std::cout << "Constructing!" << std::endl; }
    ~Test() { std::cout << "Destructing!" << std::endl; }

    Test(const Test& test)
    {
        std::cout << "Copy constructing!" << std::endl;
    }

    Test& operator++()
    {
        std::cout << "Prefix incrementing!" << std::endl;
        return *this;
    }

    Test operator++(int)
    {
        Test test;
        std::cout << "Postfix incrementing!" << std::endl;
        return test;
    }
};

int main()
{
    std::cout << "Prefix test" << std::endl;
    {
        Test test;
        ++test;
    }
    std::cout << std::endl;

    std::cout << "Postfix test" << std::endl;
    {
        Test test;
        test++;
    }

    return 0;
}
