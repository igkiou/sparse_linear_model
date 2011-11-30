/*
 * Mex implementation of LEARN_SENSING.
 *
 * Requires LAPACK and BLAS. Has been tested with both MKL's and MATLAB's 
 * implementations. 
 *
 */

/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// #include <culapack.h>
#include <mkl.h>
#include <omp.h>
#include "mex.h"
#include "matrix.h"
#include "useblas.h"
#include "sparse_classification.h"

/* The gateway routine. */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs > 3) {
		ERROR("Less than three input arguments are required.");
    } else if (nrhs < 2) {
		ERROR("At least two input arguments are required.");
    } 
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    } 
	
    /* Get input matrix dimensions*/
    INT N = (INT) mxGetM(prhs[0]);
	INT K = (INT) mxGetN(prhs[0]);
	INT M = (INT)*(DOUBLE*) mxGetData(prhs[1]);
	
	/* 
	 * Get initial Phi 
	 */
	plhs[0] = mxCreateNumericMatrix(M, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Phi = (DOUBLE*) mxGetData(plhs[0]);
	DOUBLE *inPhi;
	if (nrhs == 3) {
		if (((INT) mxGetM(prhs[2]) != M) || mxGetN(prhs[2]) != N) {
			ERROR("Invalid initialization matrix.");	
		}
		inPhi = (DOUBLE*) mxGetData(prhs[2]);
		datacpy(Phi, inPhi , M * N);
	} else {
		random_sensing(Phi, M, N);
	}
	
	DOUBLE *D = (DOUBLE*) mxGetData(prhs[0]);
	learn_sensing(Phi, D, M, N, K);
}
