/*
#define __DEBUG__
*/

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 5) {
		ERROR("Four input arguments are required.");
	}

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *W = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *Y = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *lambdap = (DOUBLE*) mxGetData(prhs[3]);
	INT numTasks = (INT) * (DOUBLE*) mxGetData(prhs[4]);
//	INT numSamples = (INT) * (DOUBLE*) mxGetData(prhs[5]);
//	INT N = (INT) * (DOUBLE*) mxGetData(prhs[6]);

//	INT numTasks = (INT) mxGetN(prhs[0]);
	INT numSamples = (INT) mxGetN(prhs[1]);
    INT N = (INT) mxGetM(prhs[1]);

//	if ((INT) mxGetM(prhs[0]) != N) {
	if ((INT) mxGetNumberOfElements(prhs[0]) != N * numTasks) {
		ERROR("First dimension of W does not match feature number (first dimension of data matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[2]) != numSamples) {
		ERROR("Number of elements of label matrix does not match number of samples (second dimension of data matrix).");
	}

	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *deriv;
	INT derivFlag;

	if (nlhs == 2) {
//		plhs[1] = mxCreateNumericMatrix(N, numTasks, MXPRECISION_CLASS, mxREAL);
		plhs[1] = mxCreateNumericMatrix(N * numTasks, 1, MXPRECISION_CLASS, mxREAL);
		deriv = (DOUBLE *) mxGetData(plhs[1]);
		derivFlag = 1;
	} else {
		deriv = NULL;
		derivFlag = 0;
	}

	DOUBLE *oneVec = (DOUBLE *) MALLOC(1 * numTasks * sizeof(DOUBLE));
	DOUBLE *expMat = (DOUBLE *) MALLOC(numTasks * numSamples * sizeof(DOUBLE));
	DOUBLE *logArg = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
	cramersinger_approx_obj_grad(obj, deriv, W, X, Y, lambdap, N, numSamples, numTasks, derivFlag, oneVec, expMat, logArg);

	FREE(oneVec);
	FREE(expMat);
	FREE(logArg);
}
