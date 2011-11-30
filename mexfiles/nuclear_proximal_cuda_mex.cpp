/*
 * nuclear_proximal_cuda_mex.c
 *
 *  Created on: Nov 7, 2011
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

	CUINT M = (CUINT) mxGetM(prhs[0]);
	CUINT N = (CUINT) mxGetN(prhs[0]);

	plhs[0] = mxCreateNumericMatrix(M, N, MXPRECISION_CLASS, mxREAL);
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
	cumalloc(handle, (void**)&Xr, M * N * sizeof(CUDOUBLE));
	cuh2dcpy(handle, Xr, h_X, M * N);
	nuclear_proximal_cuda(handle, Xr, h_norm, tau, M, N, NULL, NULL, NULL);
	cud2hcpy(handle, h_Xr, Xr, M * N);
	cufree(handle, Xr);
	culaShutdown();
	cublasShutdown(handle);
}
