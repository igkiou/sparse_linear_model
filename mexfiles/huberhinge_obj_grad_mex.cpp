/*
#define __DEBUG__
*/

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 7) {
		ERROR("Seven or less input arguments are required.");
	} else if (nrhs < 4) {
		ERROR("At least four input arguments are required.");
	}

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
	}

	DOUBLE *wb = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *Y = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *lambdap = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *kernelMatrix;
	if (nrhs >= 5) {
		kernelMatrix = (DOUBLE*) mxGetData(prhs[4]);
	} else {
		kernelMatrix = NULL;
	}
	INT biasFlag;
	if (nrhs >= 6) {
		biasFlag = (INT)*(DOUBLE*) mxGetData(prhs[5]);
	} else {
		biasFlag = 1;
	}
	INT regularizationFlag;
	if (nrhs >= 7) {
		regularizationFlag = (INT)*(DOUBLE*) mxGetData(prhs[6]);
	} else {
		regularizationFlag = 1;
	}

	INT numSamples = (INT) mxGetN(prhs[1]);
    INT M = (INT) mxGetM(prhs[1]);
	
	if ((INT) mxGetNumberOfElements(prhs[0]) != (M + 1)) {
		ERROR("Number of elements of wb does not match feature number (first dimension of data matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[2]) != numSamples) {
		ERROR("Number of elements of label matrix does not match number of samples (second dimension of data matrix).");
	}
	
	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *deriv;
	INT derivFlag;
	
	if (nlhs == 2) {
		plhs[1] = mxCreateNumericMatrix((M + 1), 1, MXPRECISION_CLASS, mxREAL);
		deriv = (DOUBLE *) mxGetData(plhs[1]);
		derivFlag = 1;
	} else {
		deriv = NULL;
		derivFlag = 0;
	}
	
	DOUBLE *Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
	DOUBLE *KX;
	if (kernelMatrix != NULL) {
		KX = (DOUBLE *) MALLOC(M * numSamples * sizeof(DOUBLE));
	} else {
		KX = NULL;
	}

	huberhinge_obj_grad(obj, deriv, wb, X, Y, lambdap, kernelMatrix, M, numSamples, \
			biasFlag, derivFlag, regularizationFlag, Ypred, KX);
	
	FREE(Ypred);
	if (kernelMatrix != NULL) {
		FREE(KX);
	}
}
