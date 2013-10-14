#
# Choose your compiler
#
# 

## use gcc
#CC = gcc-4.2
#CPP = g++-4.2
#CFLAGS=-O3 -ffast-math -fomit-frame-pointer -fPIC -mtune=k8 -march=k8 -Werror
#CC = gcc
#CPP = g++
#AR = ar
#CFLAGS =-march=native -O3 -ffast-math -fopenmp -pthread -pipe -msse2 -fPIC -W -Wall -Wextra -g -pedantic -fexceptions -fno-omit-frame-pointer

## use icc
#CC=/agbs/share/sw/icc/bin/icc
#CFLAGS= -fast -DCOMPILE_WITH_ICC -Werror
CC = icc
CPP = icpc
AR = xiar
CFLAGS = -DCOMPILE_WITH_ICC -fPIC -g -Wall -Wunused-variable -Wcheck -Wextra-tokens -Wformat -Wformat-security -Wmissing-declarations -Wmissing-prototypes -Wpointer-arith -Wreturn-type -Wsign-compare -Wuninitialized -axSSE3,SSE4.1,SSE4.2,AVX -align -O3 -pipe -ipo -fast -parallel -openmp -pthread -fexceptions -fno-omit-frame-pointer


#
# MPI KMEANS FLAGS
#
VERBOSEFLAG=-DKMEANS_VERBOSE=0 # 0: silent, 1:iteration counter, 2:everything
#PRECISION=-DINPUT_TYPE=0 # 0: double, 1:float 

#
# MATLAB
#
MATLABDIR=/usr/local/matlab/
MATLAB_INCLUDE=-I$(MATLABDIR)/extern/include

# 64 bit
SUFFIX=mexa64
MATLAB_LIB=-L$(MATLABDIR)/bin/glnxa64 -lmex

LIBS=/usr/lib/gcc/x86_64-linux-gnu/4.4/libstdc++.a /usr/lib/libm.a

all: standalone matlab libmpikmeans python
matlab: 	mpi_kmeans_mex.$(SUFFIX) mpi_assign_mex.$(SUFFIX)
standalone:	mpi_kmeans_main mpi_assign_main
python:	cython_wrapper

mpi_kmeans.o:	mpi_kmeans.cxx mpi_kmeans.h
	$(CC) $(CFLAGS) $(VERBOSEFLAG) $(PRECISION) -c -o $@ mpi_kmeans.cxx

libmpikmeans:	mpi_kmeans.o
	$(AR) rc libmpikmeans.a mpi_kmeans.o
	ranlib libmpikmeans.a
	$(CC) -shared -Wl,-soname=libmpikmeans.so -fPIC $(CFLAGS) -o libmpikmeans.so $(VERBOSEFLAGS) $(PRECISION) mpi_kmeans.cxx

mpi_kmeans_main.o:	mpi_kmeans_main.cxx
	$(CC) $(CFLAGS) $(BOOST_INCLUDE) -c -o mpi_kmeans_main.o mpi_kmeans_main.cxx 

mpi_assign_main.o:	mpi_assign_main.cxx
	$(CC) $(CFLAGS) $(BOOST_INCLUDE) -c -o mpi_assign_main.o mpi_assign_main.cxx 

mpi_kmeans_main:	libmpikmeans mpi_kmeans_main.o
	$(CC) mpi_kmeans_main.o  $(CFLAGS) -L/usr/lib/ -static -o mpi_kmeans -lm libmpikmeans.a \
		$(BOOST_LIB) $(LIBS)

mpi_assign_main:	libmpikmeans mpi_assign_main.o
	$(CC) mpi_assign_main.o  $(CFLAGS) -L/usr/lib/ -static -o mpi_assign -lm libmpikmeans.a \
		$(BOOST_LIB) $(LIBS)

%_mex.o:	%_mex.cxx
	$(CC) $(CFLAGS) $(MATLAB_INCLUDE) $(VERBOSEFLAG) $(PRECISION) -c $^ -o $@

mpi_kmeans_mex.$(SUFFIX):	libmpikmeans mpi_kmeans_mex.o 
	$(CC) mpi_kmeans_mex.o -shared -o mpi_kmeans_mex.$(SUFFIX) libmpikmeans.a $(MATLAB_LIB) 

mpi_assign_mex.$(SUFFIX):	libmpikmeans mpi_assign_mex.o
	$(CC) mpi_assign_mex.o -shared -o mpi_assign_mex.$(SUFFIX) libmpikmeans.a $(MATLAB_LIB)

cython_wrapper:	py_kmeans.c mpi_kmeans.o
	$(CPP) $(CFLAGS) $(PYTHON_INCLUDE) $(NUMPY_INCLUDE) -c -o py_kmeans.o py_kmeans.c
	$(CPP) $(CFLAGS) $(PYTHON_LIB) -lm -pthread -shared py_kmeans.o mpi_kmeans.o  -o py_kmeans.so 

test:	
	matlab -nojvm -r "test_code;exit"

clean:	
	rm -f *.o
	rm -f *.mexglx
	rm -f *.mexa64
	rm -f libmpikmeans.so
	rm -f libmpikmeans.a
	rm -f mpi_assign mpi_kmeans

