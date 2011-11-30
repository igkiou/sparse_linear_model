/*
 * nrkl_obj_grad_mex.c
 *
 *  Created on: Aug 22, 2011
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
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *K = (DOUBLE *) mxGetData(prhs[0]);
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

	INT numPoints = (INT) mxGetM(prhs[0]);
	INT numConstraints = (INT) mxGetM(prhs[2]);

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

	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *grad = NULL;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(numPoints, numPoints, MXPRECISION_CLASS, mxREAL);
		grad = (DOUBLE *) mxGetData(plhs[1]);
	}

	 kl_obj_grad(obj, grad, K, distLabelType, constraintMat, betaVec, weights, \
			 numPoints, numConstraints);

	FREE(constraintMat);
	FREE(betaVec);
}
