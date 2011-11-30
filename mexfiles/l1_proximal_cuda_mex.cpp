/*
 * l1_proximal_cuda_mex.cpp
 *
 *  Created on: Nov 13, 2011
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

	CUDOUBLE *h_X = (CUDOUBLE*) mxGetData(prhs[0]);
	CUDOUBLE tau = *(CUDOUBLE*) mxGetData(prhs[1]);

	CUINT N = (CUINT) mxGetNumberOfElements(prhs[0]);

	plhs[0] = mxCreateNumericMatrix(N, 1, MXPRECISION_CLASS, mxREAL);
	CUDOUBLE *h_Xr = (CUDOUBLE *) mxGetData(plhs[0]);
	CUDOUBLE *h_norm;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
		h_norm = (CUDOUBLE *) mxGetData(plhs[1]);
	} else {
		h_norm = NULL;
	}

	culaInitialize();
	CUHANDLE handle;
	cublasInitialize(&handle);
	CUDOUBLE *Xr = NULL;
	cumalloc(handle, (void**)&Xr, N * sizeof(CUDOUBLE));
	cuh2dcpy(handle, Xr, h_X, N);
	l1_proximal_cuda(handle, Xr, h_norm, tau, N);
	cud2hcpy(handle, h_Xr, Xr, N);
	cufree(handle, Xr);
	culaShutdown();
	cublasShutdown(handle);
}
