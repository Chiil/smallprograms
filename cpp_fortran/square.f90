subroutine square(a, n) bind(c, name="square")
    use, intrinsic :: iso_c_binding, only : c_double, c_int
    implicit none
    real(c_double), intent(inout), dimension(n) :: a
    integer(c_int), intent(in) :: n

    a(1:n) = a(1:n)**2
end subroutine
