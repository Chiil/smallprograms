module diff
contains
    subroutine diff_f90(at, a, visc, dxidxi, dyidyi, dzidzi, itot, jtot, ktot)
        use, intrinsic :: iso_c_binding, only : TF
        real(TF), intent(inout), dimension(itot, jtot, ktot) :: at
        real(TF), intent(in), dimension(itot, jtot, ktot) :: a
        real(TF), intent(in) :: visc, dxidxi, dyidyi, dzidzi
        integer, intent(in) :: itot, jtot, ktot

        do k = 2, ktot-1
            do j = 2, jtot-1
                do i = 2, itot-1
                    at(i,j,k) = at(i,j,k) + visc * ( &
                        + ( (a(i+1, j  , k  ) - a(i  , j  , k  ) ) &
                          - (a(i  , j  , k  ) - a(i-1, j  , k  ) ) ) * dxidxi &
                        + ( (a(i  , j+1, k  ) - a(i  , j  , k  ) ) &
                          - (a(i  , j  , k  ) - a(i  , j-1, k  ) ) ) * dyidyi &
                        + ( (a(i  , j  , k+1) - a(i  , j  , k  ) ) &
                          - (a(i  , j  , k  ) - a(i  , j  , k-1) ) ) * dzidzi &
                        )
                end do
            end do
        end do
    end subroutine diff_f90
end module diff
