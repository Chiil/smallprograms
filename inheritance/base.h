#include <iostream>

namespace
{
    const int n = 100;
}

class Base
{
    public:
        Base()
        { 
            std::cout << "This is the base" << std::endl;
            print();
        }

        virtual ~Base() {};

    protected:
        void print() { std::cout << "Value of n = " << n << std::endl; }
};
