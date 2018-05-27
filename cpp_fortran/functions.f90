module functions
    use, intrinsic :: iso_c_binding, only : c_double, c_size_t, c_int, c_bool
    implicit none

contains
    subroutine square(a, n) bind(c, name="square")
        real(c_double), intent(inout), dimension(n) :: a
        integer(c_size_t), value :: n

        a(1:n) = a(1:n)**2
    end subroutine

    subroutine set_array(a, itot, jtot) bind(c, name="set_array")
        integer(c_int), intent(inout), dimension(itot, jtot) :: a
        integer(c_int), value :: itot
        integer(c_int), value :: jtot

        integer :: i, j

        do j=1,jtot
            do i=1,itot
                a(i,j) = (i-1) + 10*(j-1)
            end do
        end do
    end subroutine

    subroutine increment_int(a, b) bind(c, name="increment_int")
        integer(c_int), intent(inout) :: a
        integer(c_int), value :: b

        a = a + b
    end subroutine

    subroutine increment_double(a, b) bind(c, name="increment_double")
        real(c_double), intent(inout) :: a
        real(c_double), value :: b

        a = a + b
    end subroutine

    subroutine reverse_bool(a) bind(c, name="reverse_bool")
        logical(c_bool), intent(inout) :: a

        a = .not. a
    end subroutine
end module functions
