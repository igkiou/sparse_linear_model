MKLDIR = /usr/local/intel/mkl
MATLABDIR = /usr/local/matlab
ifdef USE_CUDA
	CULADIR = /usr/local/cula
	CUDADIR = /usr/local/cuda
endif

MATLABARCH = glnxa64
MKLARCH = intel64
ifdef USE_CUDA
	CULAARCH = lib64
	CUDAARCH = lib64
endif
MEXEXT = $(shell $(MATLABDIR)/bin/mexext)
MAPFILE = mexFunction.map

MKLLIBS = -L$(MKLDIR)/lib/$(MKLARCH) -Wl,--start-group -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group
MATLABLIBS = -L$(MATLABDIR)/bin/$(MATLABARCH) -lmx -lmex -lmat
RPATH = -Wl,-rpath-link,$(MATLABDIR)/bin/$(MATLABARCH)
LIBS = $(RPATH) $(MATLABLIBS) $(MKLLIBS) -lm -lpthread
ifdef USE_CUDA
	CULALIBS = -L$(CULADIR)/$(CULAARCH) -lcula_core -lcula_lapack
	LIBS += $(CULALIBS)
	CUDALIBS = -lcublas -lcudart
	LIBS += $(CUDALIBS) 
endif

MKLINCLUDE = -I$(MKLDIR)/include 
MATLABINCLUDE= -I$(MATLABDIR)/extern/include
INCLUDES = $(MKLINCLUDE) $(MATLABINCLUDE)
ifdef USE_CUDA
	CULAINCLUDE= -I$(CULADIR)/include
	INCLUDES += $(CULAINCLUDE)
	CUDAINCLUDE= -I$(CUDADIR)/include
	INCLUDES += $(CUDAINCLUDE)
endif

CC = icc
MEXFLAGS = -DUSE_MATLAB_INTERFACE -DMATLAB_MEX_FILE -D_GNU_SOURCE -DNDEBUG -fexceptions -fno-omit-frame-pointer
MKLFLAGS = -DMKL_ILP64 -DUSE_DOUBLE_PRECISION
GENERALFLAGS = -fPIC -g -Wall -Wunused-variable -Wcheck -Wextra-tokens -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wpointer-arith -Wreturn-type -Wsign-compare -Wuninitialized
OPTIMFLAGS = -axSSE3,SSE4.1,SSE4.2,AVX -O3 -pipe -ipo -fast -parallel -openmp -pthread
REPORTSFLAGS = -opt-report 3 -openmp-report2 -par-report3 -vec-report3 -Winline -Wimplicit
FPFLAGS = -fp-model fast=2 -no-prec-sqrt
GUIDEFLAG = -guide=4
PROFGENFLAG = -prof-gen -profile-functions -profile-loops
PROFUSEFLAG = -prof-use
DEBUGFLAG = -g -D__DEBUG__
ifdef DEBUG_MODE
	CFLAGS = $(DEBUGFLAG) $(MEXFLAGS) $(MKLFLAGS) $(GENERALFLAGS) -pthread -openmp
else
	CFLAGS = $(MEXFLAGS) $(MKLFLAGS) $(GENERALFLAGS) $(OPTIMFLAGS)
	ifdef PRODUCE_REPORTS
		CFLAGS += $(REPORTSFLAGS) 
	endif
	ifdef USE_GUIDE
		CFLAGS += $(GUIDEFLAG) 
	endif
	ifdef GENERATE_PROFILE
		CFLAGS += $(PROFGENFLAG) 
	endif
	ifdef USE_PROFILE
		CFLAGS += $(PROFUSEFLAG) 
	endif
endif

ifdef USE_CUDA
	CFLAGS += -DUSE_CUDA
	NVCC = nvcc 
	#NVCFLAGS = -ccbin=$(CC) -O3 -g -arch=sm_13 -Xcompiler -O3 -Xcompiler -fast -Xcompiler -parallel -Xcompiler -fPIC -Xcompiler -fexceptions -Xcompiler -fno-omit-frame-pointer -Xcompiler -Wall
	NVCFLAGS = -ccbin=$(CC) -O3 -g -Xcompiler -O3 -Xcompiler -fast -Xcompiler -parallel -Xcompiler -fPIC -Xcompiler -fexceptions -Xcompiler -fno-omit-frame-pointer -Xcompiler -Wall
	NVCFLAGS += $(MKLFLAGS) 
endif

LDFLAGS = -pthread -shared -Wl,--version-script,$(MATLABDIR)/extern/lib/$(MATLABARCH)/$(MAPFILE) -Wl,--no-undefined
