class Animal:
    def __init__(self, age, weight):
        self.age = age
        self.weight = weight

    def print_age(self):
        print(self.age)

    def print_weight(self):
        print(self.weight)

    def make_sound(self):
        raise RuntimeError("Cannot call base")

class Dog(Animal):
    def __init__(self, age, weight):
        Animal.__init__(self, age, weight)
        self.sound = "Woof!"

    def make_sound(self):
        print(self.sound)
        
class Cat(Animal):
    def __init__(self, age, weight):
        Animal.__init__(self, age, weight)
        self.sound = "Meew!"

    def make_sound(self):
        print(self.sound)

animal1 = Dog(10, 3)
animal2 = Cat(12, 1)

animal1.print_age()
animal1.print_weight()
animal1.make_sound()

animal2.print_age()
animal2.print_weight()
animal2.make_sound()
