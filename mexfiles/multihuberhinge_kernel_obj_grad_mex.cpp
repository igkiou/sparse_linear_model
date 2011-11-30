/*
 * multihuberhinge_kernel_obj_grad_mex.c
 *
 *  Created on: Apr 27, 2011
 *      Author: igkiou
 */

/*
#define __DEBUG__
*/

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 7) {
		ERROR("Seven or less input arguments are required.");
	} else if (nrhs < 6) {
		ERROR("At least six input arguments are required.");
	}

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
	}

	DOUBLE *kernelMatrix = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *W = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *Y = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *lambdap = (DOUBLE*) mxGetData(prhs[4]);
	DOUBLE *classLabels = (DOUBLE*) mxGetData(prhs[5]);
	INT regularizationFlag;
	if (nrhs >= 7) {
		regularizationFlag = (INT)*(DOUBLE*) mxGetData(prhs[6]);
	} else {
		regularizationFlag = 1;
	}

	INT numSamples = (INT) mxGetN(prhs[2]);
    INT M = (INT) mxGetM(prhs[2]);
    INT numTasks = (INT) mxGetN(prhs[1]);

	if ((INT) mxGetM(prhs[1]) != M) {
		ERROR("First dimension of W does not match feature number (first dimension of data matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[3]) != numSamples) {
		ERROR("Number of elements of label matrix does not match number of samples (second dimension of data matrix).");
	} else if ((INT) mxGetM(prhs[0]) != M) {
		ERROR("First dimension of kernel matrix does not match feature number (first dimension of data matrix).");
	} else if ((INT) mxGetN(prhs[0]) != M) {
		ERROR("Second dimension of kernel matrix does not match feature number (first dimension of data matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[5]) != numTasks) {
		ERROR("Number of elements in classLabels does not match number of tasks (second dimension of weight matrix).");
	}

	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *deriv;
	INT derivFlag;

	if (nlhs == 2) {
		plhs[1] = mxCreateNumericMatrix(M * M, 1, MXPRECISION_CLASS, mxREAL);
		deriv = (DOUBLE *) mxGetData(plhs[1]);
		derivFlag = 1;
	} else {
		deriv = NULL;
		derivFlag = 0;
	}

	multihuberhinge_kernel_obj_grad(obj, deriv, kernelMatrix, W, X, Y, lambdap, classLabels, \
			M, numSamples, numTasks, derivFlag, regularizationFlag);
}
