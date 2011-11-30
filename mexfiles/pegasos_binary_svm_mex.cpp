/*
 * pegasos_binary_svm_mex.c
 *
 *  Created on: Mar 31, 2011
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
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *Y = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *lambda = (DOUBLE*) mxGetData(prhs[2]);

	INT N = (INT) mxGetM(prhs[0]);
	INT numSamples = (INT) mxGetN(prhs[0]);

	if ((INT) mxGetNumberOfElements(prhs[1]) != numSamples) {
		ERROR("Number of labels different from number of samples (second dimension of data matrix).");
	}

	INT biasFlag;
	if (nrhs >= 4) {
		biasFlag = (INT) * (DOUBLE *) mxGetData(prhs[3]);
	} else {
		biasFlag = 0;
	}
	INT numIters;
	if (nrhs >= 5) {
		numIters = (INT) * (DOUBLE *) mxGetData(prhs[4]);
	} else {
		numIters = numSamples;
	}
	INT batchSize;
	if (nrhs >= 6) {
		batchSize = (INT) * (DOUBLE *) mxGetData(prhs[5]);
	} else {
		batchSize = 1;
	}
	INT returnAverageFlag;
	if (nrhs >= 7) {
		returnAverageFlag = (INT) * (DOUBLE *) mxGetData(prhs[6]);
	} else {
		returnAverageFlag = 1;
	}

	plhs[0] = mxCreateNumericMatrix(N , 1, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *weights = (DOUBLE *) mxGetData(plhs[0]);
	plhs[1] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL); /* x */
	DOUBLE *bias = (DOUBLE *) mxGetData(plhs[1]);

//	pegasos_svm_vl(weights, X, Y, N, numSamples, lambda, numIters);
	pegasos_binary_svm(weights, bias, X, Y, lambda, N, numSamples, biasFlag, numIters, batchSize, returnAverageFlag);
}
