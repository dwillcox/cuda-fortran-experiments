module work_module

  use cudafor
  
  implicit none
  
contains
  
  attributes(global) subroutine work(x, coffset, n, xPitch, xLength)
    integer, value, intent(in) :: coffset, n, xPitch, xLength
    double precision, intent(inout) :: x(1:xPitch,1:xLength)
    integer          :: i

    i = blockDim%x * (blockIdx%x - 1) + threadIdx%x
    if (i > n) then
       return
    else
       i = i + coffset
       x(2, i) = x(1, i)
    endif
  end subroutine work
  
end module work_module


program cmpitch

  ! This program creates a 2-D array on the host of dimensions 2 x 1,000,000
  ! Where the elements in row 1 are initialized to 1.0d0
  ! and the elements in row 2 are initialized to 0.0d0

  ! Memory is allocated for this array on the device using cudaMallocPitch
  ! CUDA streams asynchronously transfer data to device
  ! CUDA streams run kernel setting elements in row 2 to the corresponding value in row 1.
  ! CUDA streams asynchronously transfer data back to host

  ! The array contents are summed and printed. Correct execution will yield 2.0d+6

  use work_module
  use cudafor

  implicit none
  
  double precision, pinned, allocatable :: x(:,:)
  double precision, allocatable, device :: xd(:,:)
  double precision :: xsum
  integer, parameter :: N = 1000000
  integer, parameter :: M = 2
  integer, parameter :: numStreams = 32
  integer, parameter :: blockDim = 1024
  logical :: pinnedflag
  integer :: streamSize, istat, i, chunkOffset, chunkSize
  integer :: xPitch
  integer(kind=cuda_count_kind)  :: cuPitch, cuWidth, cuLength
  integer(kind=cuda_stream_kind) :: streams(numStreams)
  character(len=200) :: cudaErrorMessage
  integer :: gridDim

  ! Allocate data array in pinned memory and check
  allocate(x(M,N), stat=istat, pinned=pinnedflag)
  if (istat /= 0) then
     write(*,*) 'Failed to allocate x array.'
     stop
  else
     if (.not. pinnedflag) then
        write(*,*) 'Allocated x array but failed to pin.'
     endif
  endif

  ! Initialize host data array
  x(1,:) = 1.0d0
  x(2,:) = 0.0d0

  ! Allocate data array in device
  cuLength = N
  cuWidth  = M
  istat = cudaMallocPitch(xd, cuPitch, cuWidth, cuLength)
  if (istat /= 0) then
     cudaErrorMessage = cudaGetErrorString(istat)
     write(*,*) 'Allocating Pitched Device Memory:'
     write(*,*) cudaErrorMessage
     write(*,*) 'cuPitch = ', cuPitch
  end if

  xPitch = cuPitch

  ! Set cuda stream, block sizes
  streamSize = ceiling(real(N)/numStreams)
  gridDim    = ceiling(real(streamSize)/blockDim)

  ! Create streams
  do i = 1, numStreams
     istat = cudaStreamCreate(streams(i))
  enddo

  ! Asynchronously copy chunks of x to device
  do i = 1, numStreams
     chunkOffset = (i-1) * streamSize
     chunkSize  = min(streamSize, N-chunkOffset)
     istat = cudaMemcpy2DAsync(xd(:, chunkOffset+1:), cuPitch, &
                                x(:, chunkOffset+1:), cuWidth, cuWidth, &
                                chunkSize, cudaMemcpyHostToDevice, streams(i))
     if (istat /= 0) then
        write(*,*) 'i = ', i
        cudaErrorMessage = cudaGetErrorString(istat)
        write(*,*) cudaErrorMessage
     end if
  enddo
  
  ! Asynchronously work on chunks of x
  do i = 1, numStreams
     chunkOffset = (i-1) * streamSize
     chunkSize  = min(streamSize, N-chunkOffset)     
     call work<<<gridDim, blockDim, 0, streams(i)>>>(xd, chunkOffset, chunkSize, xPitch, N)
  enddo

  ! Asynchronously copy chunks of x back to host
  do i = 1, numStreams
     chunkOffset = (i-1) * streamSize
     chunkSize  = min(streamSize, N-chunkOffset)
     istat = cudaMemcpy2DAsync(x(:, chunkOffset+1:), cuWidth, &
                               xd(:, chunkOffset+1:), cuPitch, cuWidth, &
                               chunkSize, cudaMemcpyDeviceToHost, streams(i))
     if (istat /= 0) then
        write(*,*) 'i = ', i
        cudaErrorMessage = cudaGetErrorString(istat)
        write(*,*) cudaErrorMessage
     end if
  enddo
  
  ! Synchronize streams
  do i = 1, numStreams
     istat = cudaStreamSynchronize(streams(i))
     if (istat /= 0) then
        write(*,*) 'i = ', i
        cudaErrorMessage = cudaGetErrorString(istat)
        write(*,*) cudaErrorMessage
     end if
  enddo

  ! Destroy streams
  do i = 1, numStreams
     istat = cudaStreamDestroy(streams(i))
     if (istat /= 0) then
        write(*,*) 'i = ', i
        cudaErrorMessage = cudaGetErrorString(istat)
        write(*,*) cudaErrorMessage
     end if     
  enddo

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
  
end program cmpitch
