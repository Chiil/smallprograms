#include <iostream>

enum Animal_type { Dog_type, Cat_type };

struct Animal
{
    virtual Animal_type get_type() const = 0;
};

struct Dog : Animal
{
    void go_for_walk() const { std::cout << "Walking. Woof!" << std::endl; }
    Animal_type get_type() const { return Dog_type; }
};

struct Cat : Animal
{
    void be_evil() const { std::cout << "Being evil!" << std::endl; }
    Animal_type get_type() const { return Cat_type; }
};

void action_option1(Animal* animal)
{
    if (animal->get_type() == Dog_type)
        dynamic_cast<Dog*>(animal)->go_for_walk();
    else if (animal->get_type() == Cat_type)
        dynamic_cast<Cat*>(animal)->be_evil();
    else
        return;
}

void action_option2(Animal* animal)
{
    Dog* dog = dynamic_cast<Dog*>(animal);
    if (dog)
    {
        dog->go_for_walk();
        return;
    }

    Cat* cat = dynamic_cast<Cat*>(animal);
    if (cat)
    {
        cat->be_evil();
        return;
    }

    return;
}

int main()
{
    Animal* cat = new Cat();
    Animal* dog = new Dog();

    action_option1(cat);
    action_option2(cat);

    action_option1(dog);
    action_option2(dog);

    return 0;
}

