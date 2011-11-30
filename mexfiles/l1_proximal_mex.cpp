/*
 * proximal_l1_mex.c
 *
 *  Created on: Apr 19, 2011
 *      Author: igkiou
 */

#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	/* Check number of input arguments */
	if (nrhs != 2) {
		ERROR("Two input arguments are required.");
    }

	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    }

	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *tau = (DOUBLE*) mxGetData(prhs[1]);

	INT N = (INT) mxGetNumberOfElements(prhs[0]);

	plhs[0] = mxCreateNumericMatrix(N, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Xr = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *norm;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
		norm = (DOUBLE *) mxGetData(plhs[1]);
	} else {
		norm = NULL;
	}

	datacpy(Xr, X, N);
	l1_proximal(Xr, norm, *tau, N);
}
