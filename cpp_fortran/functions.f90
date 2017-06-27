module functions

contains
    subroutine square(a, n) bind(c, name="square")
        use, intrinsic :: iso_c_binding, only : c_double, c_size_t
        implicit none
        real(c_double), intent(inout), dimension(n) :: a
        integer(c_size_t), value :: n

        a(1:n) = a(1:n)**2
    end subroutine

    subroutine set_array(a, itot, jtot) bind(c, name="set_array")
        use, intrinsic :: iso_c_binding, only : c_double, c_int
        implicit none
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
end module functions
