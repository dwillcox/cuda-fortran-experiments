module work_module

  use cudafor
  
  implicit none
  
contains
  
  attributes(global) subroutine work(x, xWidth, xLength)
    integer, value, intent(in) :: xWidth, xLength
    double precision, intent(inout) :: x(1:xWidth,1:xLength)
    integer          :: i

    i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
    if (i > xLength) then
       return
    else
       x(2, i) = x(1, i)
    endif
  end subroutine work
  
end module work_module


program mparray

  use work_module
  use cudafor

  implicit none
  
  double precision, managed, pointer :: x(:,:) => NULL()
  double precision :: xsum
  integer, parameter :: N = 1000000
  integer, parameter :: M = 2
  integer, parameter :: blockDim = 1024
  integer :: gridDim  
  logical :: pinnedflag
  integer :: istat, i
  character(len=200) :: cudaErrorMessage

  ! Allocate managed array
  allocate(x(M, N))
  
  ! Initialize host data array
  x(1,:) = 1.0d0
  x(2,:) = 0.0d0

  ! Set Grid dimensions
  gridDim = ceiling(real(N)/blockDim)
  
  ! Asynchronously work on chunks of x
  call work<<<gridDim, blockDim>>>(x, M, N)

  istat = cudaDeviceSynchronize()
  if (istat /= 0) then
     write(*,*) 'i = ', i
     cudaErrorMessage = cudaGetErrorString(istat)
     write(*,*) cudaErrorMessage
  end if
  
  ! Note that calling the intrinsic sum function
  ! will give the wrong answer!
  xsum = 0.0d0
  do i = 1, N
     xsum = xsum + x(1, i) + x(2, i)
  end do

  write(*,*) 'Sum of x is: ', xsum

  if (xsum .eq. 2.0d6) then
     write(*,*) 'SUCCESS!'
  else
     write(*,*) 'TEST FAILED!'
  end if
  
end program mparray
