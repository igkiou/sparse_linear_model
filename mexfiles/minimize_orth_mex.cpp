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
	if (nrhs != 4) {
		ERROR("Four input arguments are required.");
    }
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *Phi = (DOUBLE*) mxGetData(prhs[0]);
	INT length =  (INT)*(DOUBLE*) mxGetData(prhs[1]); 
	INT M = (INT)*(DOUBLE*) mxGetData(prhs[2]);
	INT N = (INT)*(DOUBLE*) mxGetData(prhs[3]);
	
	if ((INT) mxGetNumberOfElements(prhs[0]) != M * N) {
		ERROR("Number of elements of Phi matrix does not match dimensions provided.");
	}	
	
	plhs[0] = mxCreateNumericMatrix(M * N, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Phiopt = (DOUBLE *) mxGetData(plhs[0]);

	minimize_orth(Phiopt, Phi, length, M, N);
}
