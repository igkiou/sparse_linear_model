/*
 * nmmds_svp_mex.c
 *
 *  Created on: Aug 3, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 8) {
		ERROR("Eight or fewer input arguments are required.");
    } else if (nrhs < 5) {
		ERROR("At least five input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    }

	INT numPoints = (INT) * (DOUBLE *) mxGetData(prhs[0]);
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
	DOUBLE *weights = (DOUBLE*) mxGetData(prhs[3]);
	INT rank = (INT) * (DOUBLE*) mxGetData(prhs[4]);

	INT numConstraints = (INT) mxGetM(prhs[2]);

	DOUBLE tolerance;
	if ((nrhs >= 6) && (mxGetNumberOfElements(prhs[5]) != 0)) {
		tolerance = *(DOUBLE*) mxGetData(prhs[5]);
	} else {
		tolerance = 0.00001;
	}

	DOUBLE delta;
	if ((nrhs >= 7) && (mxGetNumberOfElements(prhs[6]) != 0)) {
		delta = *(DOUBLE*) mxGetData(prhs[6]);
	} else {
		delta = 0.1;
	}

	INT numIters;
	if ((nrhs >= 8) && (mxGetNumberOfElements(prhs[7]) != 0)) {
		numIters = (INT)*(DOUBLE*) mxGetData(prhs[7]);
	} else {
		numIters = 50000;
	}

	if ((mxGetNumberOfElements(prhs[3]) != 0) && \
		(mxGetNumberOfElements(prhs[3]) != numConstraints)) {
		ERROR("First dimension of vector weights does not match number of constraints (first dimension of constraint matrix).");
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
					matlabConstraintMat, numConstraints, numPoints);

	plhs[0] = mxCreateNumericMatrix(numPoints, numPoints, MXPRECISION_CLASS, mxREAL);
	DOUBLE *K = (DOUBLE *) mxGetData(plhs[0]);

	nrkl_svp(K, distLabelType, constraintMat, betaVec, weights, rank, delta, \
			numIters, tolerance, numPoints, numConstraints);

	FREE(constraintMat);
	FREE(betaVec);
}
