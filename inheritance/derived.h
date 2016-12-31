#include <iostream>

class Derived : public Base
{
    public:
        Derived()
        {
            std::cout << "This is the derived" << std::endl;
            print();
        }

        ~Derived() {};
};
