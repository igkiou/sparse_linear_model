/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <mkl.h>
#include <omp.h>
#include "mex.h"
#include "matrix.h"
#include "useblas.h"
#include "sparse_classification.h"
	
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 4) {
		ERROR("Four or less input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
	}
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }
	
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *S = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *Dinit = (DOUBLE*) mxGetData(prhs[2]);
	CHAR familyName;
	if (nrhs >= 4) {
		familyName = (CHAR)*(CHAR*) mxGetData(prhs[3]);
	} else {
		familyName = 'P';
	}
	
	INT N = (INT) mxGetM(prhs[0]);
	INT numSamples = (INT) mxGetN(prhs[0]);
	INT K = (INT) mxGetM(prhs[1]);
	
	if ((INT) mxGetNumberOfElements(prhs[2]) != N * K) {
		ERROR("Number of elements of dictionary does not match signal dimension (first dimension of data matrix) times dictionary size (first dimension of sparse code matrix).");
	} else if ((INT) mxGetN(prhs[1]) != numSamples) {
		ERROR("Second dimension of sparse code matrix does not match number of samples (second dimension of data matrix).");
	}
	
	plhs[0] = mxCreateNumericMatrix(N, K, MXPRECISION_CLASS, mxREAL);
	DOUBLE *D = (DOUBLE *) mxGetData(plhs[0]);
	EXPONENTIAL_TYPE family = convertExponentialName(familyName);
	l2exp_learn_basis_gradient_projection_backtracking(D, X, S, Dinit, N, K, numSamples, family);
}
