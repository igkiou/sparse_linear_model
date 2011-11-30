/*
 * nrml_fpc_mex.c
 *
 *  Created on: Aug 18, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 9) {
		ERROR("Nine or fewer input arguments are required.");
    } else if (nrhs < 4) {
		ERROR("At least four input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);

	CHAR distLabelName;
	if (mxGetNumberOfElements(prhs[1]) != 0) {
		if (!mxIsChar(prhs[1])) {
			ERROR("Second argument must be of type CHAR.");
		}
		distLabelName = (CHAR)*(CHAR*) mxGetData(prhs[1]);
	} else {
		distLabelName = 'R';
	}

	DOUBLE *matlabConstraintMat = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *tau = (DOUBLE*) mxGetData(prhs[3]);

	INT N = (INT) mxGetM(prhs[0]);
	INT numSamples = (INT) mxGetN(prhs[0]);
	INT numConstraints = (INT) mxGetM(prhs[2]);
	INT numRepeats = (INT) mxGetNumberOfElements(prhs[3]);

	DOUBLE tolerance;
	if ((nrhs >= 5) && (mxGetNumberOfElements(prhs[4]) != 0)) {
		tolerance = *(DOUBLE*) mxGetData(prhs[4]);
	} else {
		tolerance = 0.00001;
	}

	DOUBLE delta;
	if ((nrhs >= 6) && (mxGetNumberOfElements(prhs[5]) != 0)) {
		delta = *(DOUBLE*) mxGetData(prhs[5]);
	} else {
		delta = 0.00001;
	}

	INT numIters;
	if ((nrhs >= 7) && (mxGetNumberOfElements(prhs[6]) != 0)) {
		numIters = (INT)*(DOUBLE*) mxGetData(prhs[6]);
	} else {
		numIters = 100000;
	}

	DOUBLE tauMultiplier;
	if ((nrhs >= 8) && (mxGetNumberOfElements(prhs[7]) != 0)) {
		tauMultiplier = *(DOUBLE*) mxGetData(prhs[7]);
	} else {
		tauMultiplier = 10000;
	}

	DOUBLE tauRate;
	if ((nrhs >= 9) && (mxGetNumberOfElements(prhs[8]) != 0)) {
		tauRate = *(DOUBLE*) mxGetData(prhs[8]);
	} else {
		tauRate = 0.9;
	}

	INT *constraintMat;
	DOUBLE *betaVec;
	DIST_LABEL_TYPE distLabelType = convertDistLabelName(distLabelName);
	if (distLabelType == DIST_LABEL_TARGETS) {
		if (mxGetN(prhs[2]) != 3) {
			ERROR("Constraint matrix does not meet specified format: second dimension not equal to three.");
		}
		constraintMat = (INT *) MALLOC(numConstraints * 2 * sizeof(INT));
		betaVec = (DOUBLE *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));
	} else if (distLabelType == DIST_LABEL_BOUNDS) {
		if (mxGetN(prhs[2]) != 4) {
			ERROR("Constraint matrix does not meet specified format: second dimension not equal to four.");
		}
		constraintMat = (INT *) MALLOC(numConstraints * 3 * sizeof(INT));
		betaVec = (DOUBLE *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));
	} else if ((distLabelType == DIST_LABEL_RELATIONAL) \
				|| (distLabelType == DIST_LABEL_SQRHINGE) \
				|| (distLabelType == DIST_LABEL_HUBERHINGE)) {
		if (mxGetN(prhs[2]) != 5) {
			ERROR("Constraint matrix does not meet specified format: second dimension not equal to five.");
		}
		constraintMat = (INT *) MALLOC(numConstraints * 4 * sizeof(INT));
		betaVec = (DOUBLE *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));
	}
	convertDistanceLabelMat(constraintMat, betaVec, distLabelType, \
					matlabConstraintMat, numConstraints, numSamples);

	plhs[0] = mxCreateNumericMatrix(N, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *A = (DOUBLE *) mxGetData(plhs[0]);

	if (numRepeats > 1) {
		nrml_fpc(A, X, distLabelType, constraintMat, betaVec, tau, delta, \
				numIters, tolerance, N, numConstraints, numRepeats);
	} else {
		nrml_fpc_continuation(A, X, distLabelType, constraintMat, betaVec, *tau, \
				delta, numIters, tolerance, tauMultiplier, tauRate, N, \
				numConstraints);
	}

	FREE(constraintMat);
	FREE(betaVec);
}
