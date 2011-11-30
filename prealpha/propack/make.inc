###  Compiler flags for fortran and C modules.

#  Sun
PLAT = SUNOS

F77 = ifort  
FFLAGS = -O3 -pg  -fPIC
FFLAGS_NOOPT = -O0 -pg -fPIC
FFLAGS_ACCURATE = $(FFLAGS) 
CC = icc
CFLAGS = -O3 -fPIC
LINKER = ifort -shared
MAKELIB = ar rc
LINKFLAGS = -O3 -pg 
#BLAS =  -xlic_lib=sunperf
BLAS =  -L/usr/local/intel/mkl/lib/intel64 -Wl,--start-group -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group -liomp5 -lpthread -lsvml -limf
RANLIB = ranlib
MGS = mgs.risc
