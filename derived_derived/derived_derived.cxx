#include <iostream>

struct Animal
{
    virtual void makeSound() { std::cout << "Animal" << std::endl; }
};

struct Dog : public Animal
{
    void makeSound() { std::cout << "Dog" << std::endl; }
};

struct Labrador : public Dog
{
    void makeSound() { std::cout << "Labrador" << std::endl; }
};

int main()
{
    Animal* animal = new Labrador;
    animal->makeSound();

    Dog* dog = new Labrador;
    dog->makeSound();
    return 0;
}
