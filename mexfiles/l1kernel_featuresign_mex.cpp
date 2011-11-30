/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs > 4) {
		ERROR("Four or fewer input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    } 
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }  
	
	DOUBLE *KDX = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *KDD = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *lambda = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *beta;
	if ((nrhs >= 4) && ((INT) mxGetNumberOfElements(prhs[3]) != 0)) {
		beta = (DOUBLE*) mxGetData(prhs[3]);
	} else {
		beta = NULL;
	}

	INT numSamples = (INT) mxGetN(prhs[0]);
	INT K = (INT) mxGetM(prhs[0]);
	
	if ((INT) mxGetM(prhs[1]) != K) {
		ERROR("First dimension of dictionary-dictionary kernel does not match number of atoms (first dimension of dictionary-data kernel).");
	} else if ((INT) mxGetN(prhs[1]) != K) {
		ERROR("Second dimension of dictionary-dictionary kernel does not match number of atoms (first dimension of dictionary-data kernel).");
	}
	
	plhs[0] = mxCreateNumericMatrix(K, numSamples, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *S = (DOUBLE *) mxGetData(plhs[0]);
	
	if (numSamples < CUSTOM_OMP_NUM_THREADS) {
		omp_set_num_threads(numSamples);
	}
	
	l1kernel_featuresign(S, KDX, KDD, lambda, K, numSamples, beta);
}
