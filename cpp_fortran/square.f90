subroutine square(a, n)
    use :: iso_c_binding, only : c_double, c_int
    implicit none
    real(c_double), intent(inout), dimension(n) :: a
    integer(c_int), intent(in) :: n

    integer :: i

    do i=1,n
        a(i) = a(i)**2
    end do
end subroutine
