#include <iostream>
#include <string>

class Animal
{
    public:
        Animal(const double age, const double weight) :
            age_(age), weight_(weight) {}

        void print_age() { std::cout << age_ << std::endl; }
        void print_weight() { std::cout << weight_ << std::endl; }
        virtual void make_sound() = 0;

    private:
        double age_;
        double weight_;
};

class Dog : public Animal
{
    public:
        Dog(const double age, const double weight) :
            Animal(age, weight), sound_("Woof!") {}

        void make_sound() { std::cout << sound_ << std::endl; }

    private:
        const std::string sound_;
};

class Cat : public Animal
{
    public:
        Cat(const double age, const double weight) :
            Animal(age, weight), sound_("Meew") {}

        void make_sound() { std::cout << sound_ << std::endl; }

    private:
        const std::string sound_;
};

int main()
{
    Dog dog(10, 3);
    Cat cat(12, 1);

    dog.print_age();
    dog.print_weight();
    dog.make_sound();

    cat.print_age();
    cat.print_weight();
    cat.make_sound();

    return 0;
}
