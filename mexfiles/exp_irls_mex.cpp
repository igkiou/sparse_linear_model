/*
#define __DEBUG__
*/

#include "mex.h"
#include "matrix.h"
#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs > 4) {
		ERROR("Four or less input arguments are required.");
    } else if (nrhs < 2) {
		ERROR("At least two input arguments are required.");
    } 
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }  
	
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *D = (DOUBLE*) mxGetData(prhs[1]);
	CHAR familyName;
	if (nrhs >= 3) {
		if (!mxIsChar(prhs[2])) {
			ERROR("Third argument must be of type CHAR.");
		}
		familyName = (CHAR)*(CHAR*) mxGetData(prhs[2]);
	} else {
		familyName = 'P';
	}
	DOUBLE *beta;
	if (nrhs >= 4) {
		beta = (DOUBLE*) mxGetData(prhs[3]);
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
	
	exp_irls(S, X, D, N, K, numSamples, family, beta);
}
