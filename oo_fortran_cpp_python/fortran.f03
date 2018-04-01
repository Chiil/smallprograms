module mod_animal
    implicit none

    type, abstract :: Animal
        real :: age
        real :: weight
    contains
        procedure, pass :: print_age => animal_print_age
        procedure, pass :: print_weight => animal_print_weight
        procedure(animal_make_sound), pass, deferred :: make_sound
    end type Animal

    abstract interface
        subroutine animal_make_sound(this)
            import :: Animal
            class(Animal), intent(in) :: this
        end subroutine
    end interface

    type, extends(Animal), public :: Dog
        character(len=5), private :: sound = "Woof!"
    contains
        procedure :: make_sound => dog_make_sound
    end type Dog

    type, extends(Animal), public :: Cat
        character(len=5), private :: sound = "Meew!"
    contains
        procedure :: make_sound => cat_make_sound
    end type Cat

contains
    subroutine animal_print_age(this)
        class(Animal), intent(in) :: this
        print *, "Age: ", this%age
    end subroutine

    subroutine animal_print_weight(this)
        class(Animal), intent(in) :: this
        print *, "Weight: ", this%weight
    end subroutine

    subroutine dog_make_sound(this)
        class(Dog), intent(in) :: this
        print *, this%sound
    end subroutine

    subroutine cat_make_sound(this)
        class(Cat), intent(in) :: this
        print *, this%sound
    end subroutine
end module mod_animal

program animal_test
    use mod_animal
    implicit none

    type(Dog) :: animal_1
    animal_1 = Dog(10, 3)

    call animal_1%print_age
    call animal_1%print_weight
    call animal_1%make_sound
end program animal_test
