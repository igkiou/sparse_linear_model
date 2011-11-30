/*
#define __DEBUG__
*/

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs > 2) {
		ERROR("Two or fewer input arguments are required.");
    } else if (nrhs < 1) {
		ERROR("At least one input argument is required.");
	}
	
	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *D = (DOUBLE *) mxGetData(prhs[0]);
	INT N = (INT) mxGetM(prhs[0]);
	INT K = (INT) mxGetN(prhs[0]);

	DOUBLE radius;
	if ((nrhs >= 2) && (mxGetNumberOfElements(prhs[1]) != 0)) {
		radius = * (DOUBLE *) mxGetData(prhs[1]);
	} else {
		radius = 1;
	}
	
	plhs[0] = mxCreateNumericMatrix(N, K, MXPRECISION_CLASS, mxREAL);
	DOUBLE *DNorm = (DOUBLE *) mxGetData(plhs[0]);
	datacpy(DNorm, D, N * K);

	DOUBLE *normVec;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(1, K, MXPRECISION_CLASS, mxREAL); /* x */
		normVec = (DOUBLE *) mxGetData(plhs[1]);
	} else {
		normVec = NULL;
	}
	l2_sphere_projection_batch(DNorm, normVec, radius, N, K);
}
