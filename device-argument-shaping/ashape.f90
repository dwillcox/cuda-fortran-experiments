module tmod

  implicit none

contains

  attributes(global) subroutine atest(xmat)
    double precision :: xmat(100)
    xmat = 100.0
    call btest(xmat)
  end subroutine atest

  attributes(device) subroutine btest(xmat)
    double precision :: xmat(10,10)
    xmat = 10.0
  end subroutine btest

end module tmod

program ashape
  ! ashape tests whether or not passing a 1D
  ! array to a 2D array argument on the device
  ! is disallowed due to array reshaping
  ! restrictions on the device.
  ! RESULT: this is okay, prints 10's
  use cudafor
  use tmod
  
  implicit none

  double precision :: xm(100)
  double precision, device :: ym(100)

  xm = 0.0d0
  ym = xm
  call atest<<<1,1>>>(ym)
  xm = ym
  write(*,*) 'array entries: ', xm
end program ashape
