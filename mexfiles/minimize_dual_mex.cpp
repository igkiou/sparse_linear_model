#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <mkl.h>
#include "mex.h"
#include "matrix.h"
#include "useblas.h"
#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs != 7) {
		ERROR("Seven arguments are required.");
    }
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *dualLambdaOrig = (DOUBLE*) mxGetData(prhs[0]);
	INT length = (INT)*(DOUBLE*) mxGetData(prhs[1]); 
	DOUBLE *SSt = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *SXt = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *SXtXSt = (DOUBLE*) mxGetData(prhs[4]);
	DOUBLE trXXt = *(DOUBLE*) mxGetData(prhs[5]);
	DOUBLE c = *(DOUBLE*) mxGetData(prhs[6]);
	
    INT N = (INT) mxGetN(prhs[3]);
	INT K = (INT) mxGetM(prhs[2]);
	
/*
	if ((INT) mxGetN(prhs[1]) != numSamples) {
		ERROR("Second dimension of code matrix does not match number of samples (second dimension of signal matrix).");
	} else if ((nrhs > 4) && ((INT) mxGetM(prhs[4]) != N)) {
		ERROR("First dimension of initial dictionary does not match signal dimension (first dimension of signal matrix).");
	} else if ((nrhs > 4) && ((INT) mxGetN(prhs[4]) != K)) {
		ERROR("Second dimension of initial dictionary does not match number of atoms (first dimension of code matrix).");
	}
*/
	
	plhs[0] = mxCreateNumericMatrix(K, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *dualLambdaOpt = (DOUBLE *) mxGetData(plhs[0]);

	minimize_dual(dualLambdaOpt, dualLambdaOrig, length, SSt, SXt, SXtXSt, trXXt, c, N, K);

}
