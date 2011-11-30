/*
 * l1_distance_mex.c
 *
 *  Created on: Aug 19, 2011
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

	DOUBLE *X1 = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X2;
	if (nrhs >= 2) {
		X2 = (DOUBLE*) mxGetData(prhs[1]);
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

	if (numSamples2 == 0) {
	 	X2 = NULL;
		plhs[0] = mxCreateNumericMatrix(numSamples1, numSamples1, MXPRECISION_CLASS, mxREAL);
	} else {
		plhs[0] = mxCreateNumericMatrix(numSamples1, numSamples2, MXPRECISION_CLASS, mxREAL);
	}
	DOUBLE *distanceMat = (DOUBLE *) mxGetData(plhs[0]);

	l1_distance(distanceMat, X1, X2, N, numSamples1, numSamples2);
}
