/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>

#include <mkl.h>
#include "mex.h"
#include "matrix.h"
#include "useblas.h"
#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs != 3) {
		ERROR("Three input arguments are required.");
    }
	
	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	INT M = (INT)*(DOUBLE*) mxGetData(prhs[1]);
	INT N = (INT)*(DOUBLE*) mxGetData(prhs[2]);
	
	if ((INT) mxGetNumberOfElements(prhs[0]) != M * N) {
		ERROR("Number of elements of Phi matrix does not match dimensions provided.");
	}	

	plhs[0] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *obj = (DOUBLE *) mxGetData(plhs[0]);
	DOUBLE *deriv;
	INT derivFlag;
	
	if (nlhs == 2) {
		plhs[1] = mxCreateNumericMatrix(M * N, 1, MXPRECISION_CLASS, mxREAL);
		deriv = (DOUBLE *) mxGetData(plhs[1]);
		derivFlag = 1;
	} else {
		deriv = NULL;
		derivFlag = 0;
	}
	
	DOUBLE *PhiPhit = (DOUBLE *) MALLOC(M * M * sizeof(DOUBLE));
	
	orth_obj_grad(obj, deriv, X, M, N, derivFlag, PhiPhit);
	
	FREE(PhiPhit);
}
