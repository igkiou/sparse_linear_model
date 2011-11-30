/* 
MEX interface for LAPACK routine bdsqr.
Matlab calling sequence:
  [sigma,bnd] = bdsqr(alpha,beta)   

  Part of PROPACK by R. Larsen
*/

/* Stephen Becker, 11/12/08
 * Now, dbdqr is in a C file.  So, no underscore for the name.
 * Also, for Windows, the Lapack libraries don't need underscores
 * either.  These are controlled by pre-processor definitions
 * that I put in; either edit this source file, or, better, pass
 * in a -D[name of variable].
 * e.g. mex ... -DWINDOWS ...
 *
 * 11/6/09
 * Noticed major problems with 64-bit systems
 * Download dbdsqr.f from netlib and use in place of MATLAB's LAPACK version
 * Compile as follows (linux)
 * mex bdsqr_mex.c dbdqr.c dbdsqr.f -DDBDQR_IN_C -output bdsqr -llapack -largeArrayDims -lblas
 * or...
 * mex bdsqr_mex.c dbdqr.f dbdsqr.f -UDBDQR_IN_C -output bdsqr -llapack -largeArrayDims -lblas
 *
 * on Windows, to compile dbdsqr.f, need a Fortran compiler
 * See http://www.codingday.com/compile-lapack-and-blas-as-dll-on-windows/
 * and the compiler:
 *  http://sourceforge.net/projects/mingw/files/
 * (once compiled to dbdsqr.o, can distribute)
 * 
 * 11/9/09
 * Defining DBDQR_IN_C by default
 * Switching function prototypes to use ptrdiff_t instead of int
 * Adding "extern" to definitions
 * Couldn't compile fortran on windows (mingw works well, but mangles names
 * in a way that's not compatible with the MS Visual C++ compilter)
 * However, by switching to ptrdiff_t in the definitions, and using mwlapack library,
 * it now works on it's own, so no need to use dbdsqr.f explicitly. Problem solved!
 * On Linux, compile as:
 *  mex -v bdsqr_mex.c dbdqr.c -output bdsqr -llapack -largeArrayDims -lblas
 *
 *  12/4/09
 *  use blas.h if its available
 *
 * */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
/*
#include <algorithm>
*/

#include <mkl.h>
#include <omp.h>
#include "mex.h"
#include "matrix.h"
#include "../useblas.h"
#include "../sparse_classification.h"

/* the gateway function to be called by Matlab: */
void mexFunction(int nlhs, mxArray *plhs[], 
		 int nrhs, const mxArray *prhs[])
{
	if (nrhs != 2) {
		mexErrMsgTxt("bdsqr requires two input arguments");
	} else if  (nlhs != 2) {
		mexErrMsgTxt("bdsqr requires two output arguments");
	}

	INT M = mxGetM(prhs[0]); /* get the dimensions of the input */
	INT N = mxGetN(prhs[0]);

	/* make sure input input vectors are same length */
	if (M != mxGetM(prhs[1])) {
		mexErrMsgTxt("alpha and beta must have the same size");
	}

	/* make sure input is m x 1 */
	if ((N != 1) || (mxGetN(prhs[1]) != 1) || (N != mxGetN(prhs[1]))) {
		mexErrMsgTxt("alpha and beta must be a m x 1 vectors");
	}

	/* Create/allocate return arguments */
	plhs[0] = mxCreateNumericMatrix(M, 1, MXPRECISION_CLASS, mxREAL);
	if (plhs[0] == NULL) {
		mexErrMsgTxt("unable to allocate output");
	} 
	DOUBLE *d = mxGetPr(plhs[0]); 
	memcpy((void *) d, (void *) mxGetPr(prhs[0]), M * sizeof(DOUBLE));
	
	plhs[1] = mxCreateNumericMatrix(M, 1, MXPRECISION_CLASS, mxREAL);
	if (plhs[1] == NULL) {
		mexErrMsgTxt("unable to allocate output");
	} 
	DOUBLE *bnd = mxGetPr(plhs[1]); /* automatically zeroed out */

	DOUBLE *e = mxMalloc(M * sizeof(DOUBLE)); 
	memcpy((void *) e, (void *) mxGetPr(prhs[1]), M * sizeof(DOUBLE));
	
	DOUBLE *wrk = mxMalloc(4 * M * sizeof(DOUBLE)); 
	bdsqr_caller(M, d, bnd, e, wrk);
	
	/* Free work arrays */
	mxFree(e); 
	mxFree(wrk);
}
