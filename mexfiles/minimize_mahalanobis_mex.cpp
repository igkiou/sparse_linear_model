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
	if (nrhs != 6) {
		ERROR("Six input arguments are required.");
    }
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	INT length =  (INT)*(DOUBLE*) mxGetData(prhs[1]); 
	DOUBLE *DDt2 = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *DDt3 = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *VL = (DOUBLE*) mxGetData(prhs[4]);
	DOUBLE *L = (DOUBLE*) mxGetData(prhs[5]);
	
	INT N = (INT) mxGetN(prhs[2]);
	
	if ((INT) mxGetNumberOfElements(prhs[0]) != N * N) {
		ERROR("Number of elements of M matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetN(prhs[2]) != N) {
		ERROR("Second dimension of DDt2 matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetM(prhs[3]) != N) {
		ERROR("First dimension of DDt3 matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetN(prhs[3]) != N) {
		ERROR("Second dimension of DDt3 matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetM(prhs[4]) != N) {
		ERROR("First dimension of VL matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetN(prhs[4]) != N) {
		ERROR("Second dimension of VL matrix does not match dimension size (first dimension of DDt2 matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[5]) != N) {
		ERROR("Number of elements of L matrix does not match dimension size (first dimension of DDt2 matrix).");
	}	
	
	plhs[0] = mxCreateNumericMatrix(N, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Xopt = (DOUBLE *) mxGetData(plhs[0]);
	
	minimize_mahalanobis(Xopt, X, length, DDt2, DDt3, VL, L, N);	
}
