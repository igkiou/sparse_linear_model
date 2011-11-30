/*
#define __DEBUG__
*/

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 6) {
		ERROR("Six input arguments are required.");
	}

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *W = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *Y = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *gammap = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *rhop = (DOUBLE*) mxGetData(prhs[4]);
	DOUBLE *lambdap = (DOUBLE*) mxGetData(prhs[5]);
//	INT numSamples = (INT) * (DOUBLE*) mxGetData(prhs[5]);
//	INT N = (INT) * (DOUBLE*) mxGetData(prhs[6]);

//	INT numTasks = (INT) mxGetN(prhs[0]);
	INT numSamples = (INT) mxGetN(prhs[1]);
    INT N = (INT) mxGetM(prhs[1]);
    INT numTasks = (INT) mxGetNumberOfElements(prhs[0]) / N;

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

	DOUBLE *oneVecSvdVec = (DOUBLE *) MALLOC(1 * numTasks * sizeof(DOUBLE));
	INT minNNumTasks = IMIN(N, numTasks);
	INT maxNumSamplesMinNNumTasks = IMAX(numSamples, minNNumTasks);
	DOUBLE *logArgDerivVec = (DOUBLE *) MALLOC(1 * maxNumSamplesMinNNumTasks * sizeof(DOUBLE));
	DOUBLE *expMatVtMat = (DOUBLE *) MALLOC(maxNumSamplesMinNNumTasks * numTasks * sizeof(DOUBLE));
	DOUBLE *dataBuffer = (DOUBLE *) MALLOC(N * numTasks * sizeof(DOUBLE));

	cramersinger_nuclear_obj_grad(obj, deriv, W, X, Y, gammap, rhop, lambdap, N, numSamples, numTasks, derivFlag, \
						oneVecSvdVec, expMatVtMat, logArgDerivVec, dataBuffer, NULL, 0);

	FREE(oneVecSvdVec);
	FREE(logArgDerivVec);
	FREE(expMatVtMat);
	FREE(dataBuffer);
}
