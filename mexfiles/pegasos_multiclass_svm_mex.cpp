/*
 * pegasos_multiclass_svm_mex.c
 *
 *  Created on: Apr 1, 2011
 *      Author: igkiou
 */

/*
#define __DEBUG__
*/

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 8) {
		ERROR("Eight or less input arguments are required.");
    } else if (nrhs < 4) {
		ERROR("At least four input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *Y = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *lambda = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *classLabels = (DOUBLE*) mxGetData(prhs[3]);

	INT N = (INT) mxGetM(prhs[0]);
	INT numSamples = (INT) mxGetN(prhs[0]);
	INT numTasks = (INT) mxGetNumberOfElements(prhs[3]);

	if ((INT) mxGetNumberOfElements(prhs[1]) != numSamples) {
		ERROR("Number of labels different from number of samples (second dimension of data matrix).");
	}

	INT biasFlag;
	if (nrhs >= 5) {
		biasFlag = (INT) * (DOUBLE *) mxGetData(prhs[4]);
	} else {
		biasFlag = 0;
	}
	INT numIters;
	if (nrhs >= 6) {
		numIters = (INT) * (DOUBLE *) mxGetData(prhs[5]);
	} else {
		numIters = numSamples;
	}
	INT batchSize;
	if (nrhs >= 7) {
		batchSize = (INT) * (DOUBLE *) mxGetData(prhs[6]);
	} else {
		batchSize = 1;
	}
	INT returnAverageFlag;
	if (nrhs >= 8) {
		returnAverageFlag = (INT) * (DOUBLE *) mxGetData(prhs[7]);
	} else {
		returnAverageFlag = 1;
	}

	plhs[0] = mxCreateNumericMatrix(N , numTasks, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *weights = (DOUBLE *) mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(numTasks, 1, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *bias = (DOUBLE *) mxGetData(plhs[1]);

	pegasos_multiclass_svm(weights, bias, X, Y, lambda, classLabels, N, numSamples, \
			numTasks, biasFlag, numIters, batchSize, returnAverageFlag);
}
