module work_module

contains
  
  subroutine work(x, i)
    integer, device  :: i
    double precision, device :: x(:)
    x(i) = real(i)
  end subroutine work
  
end module work_module

program runstreams

  use work_module
  use cudafor

  implicit none
  double precision, pinned, allocatable :: x(:)
  double precision, allocatable, device :: xd(:)
  integer, device :: id
  double precision :: xsum
  integer, parameter :: N = 1000000
  integer, parameter :: numStreams = 32
  integer, parameter :: blockSize = 1024
  logical :: pinnedflag
  integer :: streamSize, istat, i, chunkOffset, chunkSize
  integer(kind=cuda_stream_kind) :: streams(numStreams)
  type(dim3) :: gridDim, blockDim

  ! Allocate data array in pinned memory and check
  allocate(x(N), stat=istat, pinned=pinnedflag)
  if (istat /= 0) then
     write(*,*) 'Failed to allocate x array.'
     stop
  else
     if (.not. pinnedflag) then
        write(*,*) 'Allocated x array but failed to pin.'
     endif
  endif

  ! Initialize host data array
  x = 0.0d0

  ! Allocate data array in device
  allocate(xd(N))

  ! Set cuda stream, block sizes
  streamSize = ceiling(real(N)/numStreams)
  blockDim   = dim3(blockSize,1,1)
  gridDim    = dim3(ceiling(real(streamSize)/blockSize),1,1)

  ! Create streams
  do i = 1, numStreams
     istat = cudaStreamCreate(streams(i))
  enddo

  ! Asynchronously copy chunks of x to device
  do i = 1, numStreams
     chunkOffset = (i-1) * streamSize
     chunkSize  = min(streamSize, N-chunkOffset)
     istat = cudaMemcpyAsync(xd(chunkOffset+1), x(chunkOffset+1), chunkSize, streams(i))
  enddo

  ! Asynchronously work on chunks of x
  do id = 1, N
     call work(xd, id)
  enddo

  ! Asynchronously copy chunks of x back to host
  do i = 1, numStreams
     chunkOffset = (i-1) * streamSize
     chunkSize  = min(streamSize, N-chunkOffset)     
     istat = cudaMemcpyAsync(x(chunkOffset+1), xd(chunkOffset+1), chunkSize, streams(i))
  enddo

  ! Synchronize streams
  do i = 1, numStreams
     istat = cudaStreamSynchronize(streams(i))
  enddo

  ! Destroy streams
  do i = 1, numStreams
     istat = cudaStreamDestroy(streams(i))
  enddo

  ! Note that calling the intrinsic sum function
  ! will give the wrong answer!
  xsum = 0.0d0
  do i = 1, N
     xsum = xsum + real(x(i))
  end do

  write(*,*) 'Sum of x is: ', xsum
  
end program runstreams
