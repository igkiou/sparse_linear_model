/*
 * nmmds_mex.c
 *
 *  Created on: Jul 30, 2011
 *      Author: igkiou
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs > 7) {
		ERROR("Seven or fewer input arguments are required.");
    } else if (nrhs < 4) {
		ERROR("At least four input arguments are required.");
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
	DOUBLE tau = * (DOUBLE*) mxGetData(prhs[3]);

	INT numConstraints = (INT) mxGetM(prhs[2]);

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
		delta = 0.1;
	}

	INT numIters;
	if ((nrhs >= 7) && (mxGetNumberOfElements(prhs[6]) != 0)) {
		numIters = (INT)*(DOUBLE*) mxGetData(prhs[6]);
	} else {
		numIters = 50000;
	}

	DIST_LABEL_TYPE distLabelType = convertDistLabelName(distLabelName);
	if (distLabelType != DIST_LABEL_RELATIONAL) {
		ERROR("NRML_SVT currently only supports relational constraints.");
	}
	if (mxGetN(prhs[2]) != 5) {
		ERROR("Constraint matrix does not meet specified format: second dimension not equal to five.");
	}
	INT *constraintMat = (INT *) MALLOC(numConstraints * 4 * sizeof(INT));
	DOUBLE *betaVec = (DOUBLE *) MALLOC(numConstraints * 1 * sizeof(DOUBLE));
	convertDistanceLabelMat(constraintMat, betaVec, distLabelType, \
					matlabConstraintMat, numConstraints, numPoints);

	plhs[0] = mxCreateNumericMatrix(numPoints, numPoints, MXPRECISION_CLASS, mxREAL);
	DOUBLE *K = (DOUBLE *) mxGetData(plhs[0]);

	nrkl_svt(K, constraintMat, betaVec, tau, delta, numIters, tolerance, \
				numPoints, numConstraints);

	FREE(constraintMat);
	FREE(betaVec);
}
