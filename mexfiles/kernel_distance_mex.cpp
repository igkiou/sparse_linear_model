/*
 * kernel_distance_mex.c
 *
 *  Created on: Jul 16, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 2) {
		ERROR("Two or fewer input arguments are required.");
    } else if (nrhs < 1) {
		ERROR("At least one input argument is required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *K = (DOUBLE*) mxGetData(prhs[0]);
	INT sqrtFlag;
	if (nrhs >= 2) {
		sqrtFlag = (INT)*(DOUBLE*) mxGetData(prhs[1]);
		if (sqrtFlag > 0) {
			sqrtFlag = 1;
		}
	} else {
		sqrtFlag = 0;
	}

	INT numSamples = (INT) mxGetM(prhs[0]);

	if ((INT) mxGetM(prhs[0]) != numSamples) {
		ERROR("Kernel matrix must be square.");
	}

	plhs[0] = mxCreateNumericMatrix(numSamples, numSamples, MXPRECISION_CLASS, mxREAL);
	DOUBLE *distanceMat = (DOUBLE *) mxGetData(plhs[0]);

	DOUBLE *normMat = (DOUBLE *) MALLOC(numSamples * 1 * sizeof(DOUBLE));
	DOUBLE *oneVec = (DOUBLE *) MALLOC(numSamples * 1 * sizeof(DOUBLE));

	kernel_distance(distanceMat, K, numSamples, sqrtFlag, normMat, oneVec);

	FREE(normMat);
	FREE(oneVec);
}
