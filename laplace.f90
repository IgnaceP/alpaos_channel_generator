subroutine laplaciansolver(h, k, boundary_i, boundary_j, outside_mask, chan, rows, cols, iterations, resolution)

    implicit none
    integer, intent(in) :: rows
    integer, intent(in) :: cols
    integer, intent(in) :: iterations
    real, intent(in) :: resolution

    real(8), dimension(rows,cols), intent(inout) :: h
    real(8), dimension(rows,cols), intent(in) :: k
    integer, dimension(rows,cols), intent(in) :: boundary_i
    integer, dimension(rows,cols), intent(in) :: boundary_j
    integer, dimension(rows,cols), intent(in) :: outside_mask
    integer, dimension(rows,cols), intent(in) :: chan
    
    integer :: i, j, iter
    real(8), dimension(rows,cols) :: h_old ! Temporary solution array

    
    ! Initialize temporary solution array
    h_old = 0.0

    do iter = 1, iterations
        h_old = h
        
        do i = 2, rows - 1
            do j = 2, cols - 1
                h(i,j) = (h_old(i - 1, j) + h_old(i + 1, j) + h_old(i, j - 1) + h_old(i, j + 1) - (resolution ** 2) * k(i, j)) / 4.0
            end do
        end do
        
        do i = 1, rows
            do j = 1, cols
                if (h(boundary_i(i, j)+1, boundary_j(i, j)+1) > 0) then
                    h(i, j) = h(boundary_i(i, j)+1, boundary_j(i, j)+1)
                end if

                if (chan(i, j) /= 0) then
                    h(i, j) = 0
                end if

                if (outside_mask(i, j) /= 0) then
                    h(i, j) = 0
                end if
        
            end do
        end do
    end do
    

end subroutine laplaciansolver
