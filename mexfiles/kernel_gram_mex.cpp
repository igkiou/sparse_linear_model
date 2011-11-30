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
	if (nrhs > 5) {
		ERROR("Five or less input arguments are required.");
    } else if (nrhs < 1) {
		ERROR("At least one input argument is required.");
	}
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }
	
	DOUBLE *X1 = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X2;
	if (nrhs >= 2) {
		X2 = (DOUBLE*) mxGetData(prhs[1]);
	}
	CHAR kernelName;
	if (nrhs >= 3) {
		if (!mxIsChar(prhs[2])) {
			ERROR("Third argument must be of type CHAR.");
		}
		kernelName = (CHAR)*(CHAR*) mxGetData(prhs[2]);
	} else {
		kernelName = 'G';
	}
	
	DOUBLE *param1;
	DOUBLE *param2;
	if (nrhs >= 4) {
		param1 = (DOUBLE*) mxGetData(prhs[3]);
	} else {
		param1 = NULL;
	}
	if (nrhs >= 5) {
		param2 = (DOUBLE*) mxGetData(prhs[4]);
	} else {
		param2 = NULL;
	}
	
	INT N = (INT) mxGetM(prhs[0]);
	INT numSamples1 = (INT) mxGetN(prhs[0]);
	INT numSamples2;
	if (nrhs >= 2) {
		numSamples2 = (INT) mxGetN(prhs[1]);
	} else {
		numSamples2 = 0;
	}
	
	if ((numSamples2 != 0) && ((INT) mxGetM(prhs[1]) != N)) {
		ERROR("The signal dimension (first dimension) of the second sample matrix does not match the signal dimension (first dimension) of the first sample matrix.");
	}
	
	KERNEL_TYPE kernelType = convertKernelName(kernelName);

	if (numSamples2 == 0) {
	 	X2 = NULL;
		plhs[0] = mxCreateNumericMatrix(numSamples1, numSamples1, \
										MXPRECISION_CLASS, mxREAL);
	} else {
		plhs[0] = mxCreateNumericMatrix(numSamples1, numSamples2, \
										MXPRECISION_CLASS, mxREAL);
	}
	DOUBLE *kernelMat = (DOUBLE *) mxGetData(plhs[0]);

	kernel_gram(kernelMat, X1, X2, N, numSamples1, numSamples2, kernelType, \
				param1, param2, NULL, NULL);
}	
