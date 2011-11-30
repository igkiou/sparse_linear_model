/*
#define __DEBUG__
*/

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs > 6) {
		ERROR("Six or less input arguments are required.");
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
	DOUBLE *beta;
	if (nrhs >= 4) {
		beta = (DOUBLE*) mxGetData(prhs[3]);
	} else {
		beta = NULL;
	}
	DOUBLE *KDD;
	if ((nrhs >= 5) && ((INT) mxGetNumberOfElements(prhs[4]) != 0)) {
		KDD = (DOUBLE*) mxGetData(prhs[4]);
	} else {
		KDD = NULL;
	}
	DOUBLE *KDX;
	if ((nrhs >= 6) && ((INT) mxGetNumberOfElements(prhs[5]) != 0)) {
		KDX = (DOUBLE*) mxGetData(prhs[5]);
	} else {
		KDX = NULL;
	}
	
	INT N = (INT) mxGetM(prhs[1]);
	INT K = (INT) mxGetN(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[0]);
	
	if ((INT) mxGetM(prhs[0]) != N) {
		ERROR("First dimension of X matrix does not match dimension size (first dimension of D matrix).");
	} else if ((nrhs >= 5) && ((INT) mxGetM(prhs[4]) != 0) && ((INT) mxGetM(prhs[4]) != K)) {
		ERROR("First dimension of dictionary-dictionary kernel does not match number of atoms (second dimension of D matrix).");
	} else if ((nrhs >= 5) && ((INT) mxGetN(prhs[4]) != 0) && ((INT) mxGetN(prhs[4]) != K)) {
		ERROR("Second dimension of dictionary-dictionary kernel does not match number of atoms (second dimension of D matrix).");
	} else if ((nrhs >= 6) && ((INT) mxGetM(prhs[5]) != 0) && ((INT) mxGetM(prhs[5]) != K)) {
		ERROR("First dimension of dictionary-data kernel does not match number of atoms (second dimension of D matrix).");
	} else if ((nrhs >= 6) && ((INT) mxGetN(prhs[5]) != 0) && ((INT) mxGetN(prhs[5]) != numSamples)) {
		ERROR("Second dimension of dictionary-data kernel does not match number of samples (second dimension of data matrix).");
	}
	
	plhs[0] = mxCreateNumericMatrix(K, numSamples, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *S = (DOUBLE *) mxGetData(plhs[0]);
	
	if (numSamples < CUSTOM_OMP_NUM_THREADS) {
		omp_set_num_threads(numSamples);
	}
	
	l1qp_featuresign(S, X, D, lambda, N, K, numSamples, beta, KDD, KDX);
}
