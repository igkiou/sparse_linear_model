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
	if (nrhs > 7) {
		ERROR("Seven or less arguments are required.");
    } else if (nrhs < 4) {
		ERROR("At least four arguments are required.");
	}
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *A = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *Dorig = (DOUBLE*) mxGetData(prhs[2]);
	INT numSearches = (INT)*(DOUBLE*) mxGetData(prhs[3]); 
	char kernelName;
	if (nrhs >= 5) {
		if (!mxIsChar(prhs[4])) {
			ERROR("Fifth argument must be of type CHAR.");
		}
		kernelName = (char)*(char*) mxGetData(prhs[4]);
	} else {
		kernelName = 'G';
	}
	DOUBLE *param1;
	DOUBLE *param2;
	if (nrhs >= 6) {
		param1 = (DOUBLE*) mxGetData(prhs[5]);
	} else {
		param1 = NULL;
	}
	if (nrhs >= 7) {
		param2 = (DOUBLE*) mxGetData(prhs[6]);
	} else {
		param2 = NULL;
	}
	
    INT N = (INT) mxGetM(prhs[0]);
	INT K = (INT) mxGetM(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[0]);
	
	if ((INT) mxGetN(prhs[1]) != numSamples) {
		ERROR("Second dimension of code matrix does not match number of samples (second dimension of signal matrix).");
	} else if ((INT) mxGetM(prhs[2]) != N) {
		ERROR("First dimension of initial dictionary does not match signal dimension (first dimension of signal matrix).");
	} else if ((INT) mxGetN(prhs[2]) != K) {
		ERROR("Second dimension of initial dictionary does not match number of atoms (first dimension of code matrix).");
	} else if ((kernelName != 'G') && (kernelName != 'g')) {
		ERROR("Kernel dictionary learning only implemented for Gaussian kernel.");
	}
	
	plhs[0] = mxCreateNumericMatrix(N, K, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Dopt = (DOUBLE *) mxGetData(plhs[0]);
	KERNEL_TYPE kernelType = convertKernelName(kernelName);
	minimize_kernel_basis(Dopt, Dorig, numSearches, X, A, N, K, numSamples, kernelType, param1, param2);
}
