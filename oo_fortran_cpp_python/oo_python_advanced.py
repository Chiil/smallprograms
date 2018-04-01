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

animals = []
animals.append( Dog(10, 3) )
animals.append( Cat(12, 1) )

for a in animals:
    a.print_age()
    a.print_weight()
    a.make_sound()
