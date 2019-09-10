program fortran_pointer
    implicit none
    integer, dimension(:,:), allocatable, target :: a
    integer, dimension(:,:), pointer :: a_ptr => NULL()

    integer :: nx, ny
    integer :: i, j

    nx = 4
    ny = 4

    allocate(a(nx, ny))

    a = 0

    a_ptr => a(2:4:2,:)

    call set_to_one(a_ptr, 2, ny)

    print *, a
end program fortran_pointer

subroutine set_to_one(a, nx, ny)
    integer, dimension(nx, ny), intent(out) :: a
    integer, intent(in) :: nx
    integer, intent(in) :: ny

    do j=1,ny
        do i=1,nx
            a(i,j) = 1
        end do
    end do
end subroutine
