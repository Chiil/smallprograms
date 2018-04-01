import abc

class Animal(metaclass = abc.ABCMeta):
    def __init__(self, age, weight):
        self.age = age
        self.weight = weight

    def print_age(self):
        print(self.age)

    def print_weight(self):
        print(self.weight)

    @abc.abstractmethod
    def make_sound(self):
        pass
        
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

animals = [ animal1, animal2 ]

for a in animals:
    a.print_age()
    a.print_weight()
    a.make_sound()
