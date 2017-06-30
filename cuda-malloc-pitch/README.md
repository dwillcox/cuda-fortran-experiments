Test program for `cudaMallocPitch`.

Correct execution of the program should print the following, as tested
on a GTX 640 and GTX 960 using PGI 17.4, GCC 6.3.1, and version 8.0.61
of the CUDA Toolkit.

```
 Sum of x is:     2000000.000000000     
 SUCCESS!
```

On Summitdev, however, I'm getting the following error when attempting
to allocate memory using `cudaMallocPitch`:

```
0: cudaMallocPitch: extra pitch amount is not a multiple of the datatype size
```

On Summitdev, I'm compiling and executing using the included Makefile
and the following loaded modules:

```
bash-4.2$ module list

    Currently Loaded Modules:

    1) hsi/5.0.2.p5   3) DefApps         5) pgi/17.4                         7) python/3.5.2
    2) xalt/0.7.5     4) cuda/8.0.61-1   6) spectrum_mpi/10.1.0.2-20161221
```

Additionally, I'm using the following environment variable settings
with the included PGI RC file (`.mypgcpprc.summitdev.gcc631.pgi174`):

```
  export PGI_LOCALRC=~/.mypgcpprc.summitdev.gcc631.pgi174
  export LD_LIBRARY_PATH=/sw/summitdev/gcc/6.3.1-20170301/lib64:$LD_LIBRARY_PATH
```
