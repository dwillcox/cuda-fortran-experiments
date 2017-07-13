module work_module

  use cudafor
  
  implicit none

  type abtype
     double precision :: a
     double precision :: b
  end type abtype
  
contains
  
  attributes(global) subroutine work(x, xLength)
    integer, value, intent(in) :: xLength
    type(abtype), intent(inout) :: x(1:xLength)
    integer          :: i

    i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
    if (i > xLength) then
       return
    else
       x(i) % b = x(i) % a
    endif
  end subroutine work
  
end module work_module


program mparray

  use work_module
  use cudafor

  implicit none
  
  type(abtype), managed, allocatable :: x(:)
  double precision :: xsum
  integer, parameter :: N = 1000000
  integer, parameter :: blockDim = 1024
  integer :: gridDim  
  logical :: pinnedflag
  integer :: istat, i
  character(len=200) :: cudaErrorMessage

  ! Allocate managed array
  allocate(x(N))
  
  ! Initialize host data array
  do i = 1, N
     x(i) % a = 1.0d0
     x(i) % b = 0.0d0
  end do

  ! Set Grid dimensions
  gridDim = ceiling(real(N)/blockDim)
  
  ! Asynchronously work on chunks of x
  call work<<<gridDim, blockDim>>>(x, N)

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
     xsum = xsum + x(i) % a + x(i) % b
  end do

  write(*,*) 'Sum of x is: ', xsum

  if (xsum .eq. 2.0d6) then
     write(*,*) 'SUCCESS!'
  else
     write(*,*) 'TEST FAILED!'
  end if
  
end program mparray
