/*
#define __DEBUG__
*/

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
#include "useblas.h"
#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs > 5) {
		ERROR("Five or less input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    } 
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }  
	
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *D = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *lambda = (DOUBLE*) mxGetData(prhs[2]);
	CHAR familyName;
	if (nrhs >= 4) {
		if (!mxIsChar(prhs[3])) {
			ERROR("Fourth argument must be of type CHAR.");
		}
		familyName = (CHAR)*(CHAR*) mxGetData(prhs[3]);
	} else {
		familyName = 'P';
	}
	DOUBLE *beta;
	if (nrhs >= 5) {
		beta = (DOUBLE*) mxGetData(prhs[4]);
	} else {
		beta = NULL;
	}
	
	INT N = (INT) mxGetM(prhs[1]);
	INT K = (INT) mxGetN(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[0]);
	
	if ((INT) mxGetM(prhs[0]) != N) {
		ERROR("First dimension of X matrix does not match dimension size (first dimension of D matrix).");
	}
	
	plhs[0] = mxCreateNumericMatrix(K, numSamples, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *S = (DOUBLE *) mxGetData(plhs[0]);
	EXPONENTIAL_TYPE family = convertExponentialName(familyName);
	
	if (numSamples < CUSTOM_OMP_NUM_THREADS) {
		omp_set_num_threads(numSamples);
	}

	l1exp_featuresign(S, X, D, lambda, N, K, numSamples, family, beta);
}
