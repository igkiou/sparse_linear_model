MKLDIR = /usr/local/intel/mkl
MATLABDIR = /usr/local/matlabR2009a
CULADIR = /usr/local/cula
CUDADIR = /usr/local/cuda
ROOTDIR = $(shell pwd)
INCLUDEDIR = $(ROOTDIR)
SRCDIR = $(ROOTDIR)
MEXDIR = $(ROOTDIR)
BINDIR = $(ROOTDIR)

MATLABARCH = glnxa64
MKLARCH = intel64
ICCARCH = intel64
CULAARCH = lib64
CUDAARCH = lib64
MEXEXT = $(shell $(MATLABDIR)/bin/mexext)
MAPFILE = mexFunction.map

MKLLIBS = -L$(MKLDIR)/lib/$(MKLARCH) $(MKLDIR)/lib/$(MKLARCH)/libmkl_solver_ilp64.a -Wl,--start-group -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -Wl,--end-group
MATLABLIBS = -L$(MATLABDIR)/bin/$(MATLABARCH) -lmx -lmex -lmat
RPATH = -Wl,-rpath-link,$(MATLABDIR)/bin/$(MATLABARCH)
CULALIBS = -L$(CULADIR)/$(CULAARCH) -lcula_core -lcula_lapack
# CUDALIBS = -L$(CUDADIR)/$(CUDAARCH) -lcublas -lcudart
CUDALIBS = -lcublas -lcudart
LIBS = $(RPATH) $(MATLABLIBS) $(MKLLIBS) $(CULALIBS) $(CUDALIBS) -lm -lpthread 

MKLINCLUDE = -I$(MKLDIR)/include 
MATLABINCLUDE= -I$(MATLABDIR)/extern/include
CULAINCLUDE= -I$(CULADIR)/include
CUDAINCLUDE= -I$(CUDADIR)/include
INCLUDES = $(MKLINCLUDE) $(MATLABINCLUDE) $(CULAINCLUDE) $(CUDAINCLUDE) -I$(INCLUDEDIR)

CC = icc
ifdef DEBUG_MODE
	CFLAGS = $(DEBUGFLAG) $(MEXFLAGS) $(MKLFLAGS) $(GENERALFLAGS) 
else
	CFLAGS = $(MEXFLAGS) $(MKLFLAGS) $(GENERALFLAGS) $(OPTIMFLAGS)
	ifdef PRODUCE_REPORTS
		CFLAGS += $(REPORTSFLAGS) 
	endif
	ifdef GENERATE_PROFILE
		CFLAGS += $(PROFGENFLAG) 
	endif
	ifdef USE_PROFILE
		CFLAGS += $(PROFUSEFLAG) 
	endif
endif

CC = icc
MEXFLAGS = -DUSE_MATLAB_INTERFACE -DMATLAB_MEX_FILE -D_GNU_SOURCE -DNDEBUG -fexceptions -fno-omit-frame-pointer
MKLFLAGS = -DMKL_ILP64 -DUSE_DOUBLE_PRECISION
GENERALFLAGS = -fPIC -Wall -g
OPTIMFLAGS = 
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
LDFLAGS = -pthread -shared -Wl,--version-script,$(MATLABDIR)/extern/lib/$(MATLABARCH)/$(MAPFILE) -Wl,--no-undefined

all: $(MEXDIR)/cudaSample.$(MEXEXT) $(MEXDIR)/mklSample.$(MEXEXT) \
	$(MEXDIR)/cudaSampleLegacy.$(MEXEXT) $(MEXDIR)/cudaSampleCula.$(MEXEXT) \
	$(MEXDIR)/cudaSampleCulaMalloc.$(MEXEXT) $(MEXDIR)/cudaSampleCulaHost.$(MEXEXT) \
	$(MEXDIR)/mklSampleOpenMP.$(MEXEXT) $(MEXDIR)/cudaSampleKernel.$(MEXEXT)

$(MEXDIR)/cudaSample.$(MEXEXT): $(MEXDIR)/cudaSample.o
	$(CC) $(LDFLAGS) $(INCLUDES) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/cudaSample.o: $(MEXDIR)/cudaSample.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(MEXDIR)/cudaSample.cpp

$(MEXDIR)/mklSample.$(MEXEXT): $(MEXDIR)/mklSample.o
	$(CC) $(LDFLAGS) $(INCLUDES) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/mklSample.o: $(MEXDIR)/mklSample.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(MEXDIR)/mklSample.cpp

$(MEXDIR)/cudaSampleLegacy.$(MEXEXT): $(MEXDIR)/cudaSampleLegacy.o
	$(CC) $(LDFLAGS) $(INCLUDES) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/cudaSampleLegacy.o: $(MEXDIR)/cudaSampleLegacy.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(MEXDIR)/cudaSampleLegacy.cpp
	
$(MEXDIR)/cudaSampleCula.$(MEXEXT): $(MEXDIR)/cudaSampleCula.o
	$(CC) $(LDFLAGS) $(INCLUDES) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/cudaSampleCula.o: $(MEXDIR)/cudaSampleCula.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(MEXDIR)/cudaSampleCula.cpp
	
$(MEXDIR)/cudaSampleCulaMalloc.$(MEXEXT): $(MEXDIR)/cudaSampleCulaMalloc.o
	$(CC) $(LDFLAGS) $(INCLUDES) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/cudaSampleCulaMalloc.o: $(MEXDIR)/cudaSampleCulaMalloc.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(MEXDIR)/cudaSampleCulaMalloc.cpp
	
$(MEXDIR)/cudaSampleCulaHost.$(MEXEXT): $(MEXDIR)/cudaSampleCulaHost.o
	$(CC) $(LDFLAGS) $(INCLUDES) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/cudaSampleCulaHost.o: $(MEXDIR)/cudaSampleCulaHost.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(MEXDIR)/cudaSampleCulaHost.cpp

$(MEXDIR)/cudaSampleKernel.$(MEXEXT): $(MEXDIR)/cudaSampleKernel.o
	icpc $(LDFLAGS) $(INCLUDES) $(LIBS) $(CFLAGS) -o $@ $^ $(MEXDIR)/cudaSampleKernel_sub.o

$(MEXDIR)/cudaSampleKernel.o: $(MEXDIR)/cudaSampleKernel.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(MEXDIR)/cudaSampleKernel.cpp
	
$(MEXDIR)/mklSampleOpenMP.$(MEXEXT): $(MEXDIR)/mklSampleOpenMP.o
	$(CC) $(LDFLAGS) $(INCLUDES) $(LIBS) $(CFLAGS) -o $@ $^

$(MEXDIR)/mklSampleOpenMP.o: $(MEXDIR)/mklSampleOpenMP.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $(MEXDIR)/mklSampleOpenMP.cpp

clean:	
	rm -rf *.o *~
	rm -rf $(MEXDIR)/*.o $(MEXDIR)/*~
	rm -rf $(SRCDIR)/*.o $(SRCDIR)/*~

distclean:	
	rm -rf *.o *~
	rm -rf $(MEXDIR)/*.o $(MEXDIR)/*.$(MEXEXT) $(MEXDIR)/*~
	rm -rf $(SRCDIR)/*.o $(SRCDIR)/*~
