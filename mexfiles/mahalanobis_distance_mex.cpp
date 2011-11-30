/*
 * mahalanobis_distance_mex.c
 *
 *  Created on: Jul 15, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 4) {
		ERROR("Four or fewer input arguments are required.");
    } else if (nrhs < 3) {
		ERROR("At least three input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X1 = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X2 = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *S = (DOUBLE*) mxGetData(prhs[2]);
	INT sqrtFlag;
	if (nrhs >= 4) {
		sqrtFlag = (INT)*(DOUBLE*) mxGetData(prhs[3]);
		if (sqrtFlag > 0) {
			sqrtFlag = 1;
		}
	} else {
		sqrtFlag = 0;
	}

	INT N = (INT) mxGetM(prhs[0]);
	INT numSamples1 = (INT) mxGetN(prhs[0]);
	INT numSamples2 = (INT) mxGetN(prhs[1]);

	if ((numSamples2 != 0) && ((INT) mxGetM(prhs[1]) != N)) {
		ERROR("The signal dimension (first dimension) of the second sample matrix does not match the signal dimension (first dimension) of the first sample matrix.");
	} else if (mxGetM(prhs[2]) != N) {
		ERROR("The first dimension of the Mahalanobis matrix does not match the signal dimension (first dimension) of the first sample matrix.");
	} else if (mxGetN(prhs[2]) != N) {
		ERROR("The second dimension of the Mahalanobis matrix does not match the signal dimension (first dimension) of the first sample matrix.");
	}

	if (numSamples2 == 0) {
	 	X2 = NULL;
		plhs[0] = mxCreateNumericMatrix(numSamples1, numSamples1, MXPRECISION_CLASS, mxREAL);
	} else {
		plhs[0] = mxCreateNumericMatrix(numSamples1, numSamples2, MXPRECISION_CLASS, mxREAL);
	}
	DOUBLE *distanceMat = (DOUBLE *) mxGetData(plhs[0]);

	DOUBLE *tempX1 = (DOUBLE *) MALLOC(N * numSamples1 * sizeof(DOUBLE));
	DOUBLE *tempX2 = (DOUBLE *) MALLOC(N * numSamples2 * sizeof(DOUBLE));
	DOUBLE *normMat1 = (DOUBLE *) MALLOC(numSamples1 * 1 * sizeof(DOUBLE));
	DOUBLE *oneVec = (DOUBLE *) MALLOC(numSamples1 * 1 * sizeof(DOUBLE));

	mahalanobis_distance(distanceMat, X1, X2, S, N, numSamples1, numSamples2, \
					sqrtFlag, tempX1, tempX2, normMat1, oneVec);

	FREE(tempX1);
	FREE(tempX2);
	FREE(normMat1);
	FREE(oneVec);
}
