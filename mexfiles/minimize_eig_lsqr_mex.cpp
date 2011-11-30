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
	if (nrhs != 10) {
		ERROR("Ten input arguments are required.");
    }
	
	/* Check number of output arguments */
	if (nlhs > 1) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[0]);
	INT length =  (INT)*(DOUBLE*) mxGetData(prhs[1]); 
	DOUBLE *Xdata = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *Y = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *DDt2 = (DOUBLE*) mxGetData(prhs[4]);
	DOUBLE *DDt3 = (DOUBLE*) mxGetData(prhs[5]);
	DOUBLE *VL = (DOUBLE*) mxGetData(prhs[6]);
	DOUBLE *L = (DOUBLE*) mxGetData(prhs[7]);
	DOUBLE alphaReg = *(DOUBLE*) mxGetData(prhs[8]);
	DOUBLE betaReg = *(DOUBLE*) mxGetData(prhs[9]);
	
	INT N = (INT) mxGetM(prhs[2]);
	INT numSamples = (INT) mxGetN(prhs[2]);
	INT M = (INT) mxGetM(prhs[3]);
	
	if ((INT) mxGetNumberOfElements(prhs[0]) != M * N) {
		ERROR("Number of elements of Phi does not match product of reduced dimension number (first dimension of Y matrix) times dimension number (first dimension of data matrix).");
	} else if ((INT) mxGetN(prhs[3]) != numSamples) {
		ERROR("Second dimension of Y matrix does not match number of samples (second dimension of data matrix).");
	} else if ((INT) mxGetM(prhs[4]) != N) {
		ERROR("First dimension of DDt2 matrix does not match dimension number (first dimension of data matrix).");
	} else if ((INT) mxGetN(prhs[4]) != N) {
		ERROR("Second dimension of DDt2 matrix does not match dimension number (first dimension of data matrix).");
	} else if ((INT) mxGetM(prhs[5]) != N) {
		ERROR("First dimension of DDt3 matrix does not match dimension number (first dimension of data matrix).");
	} else if ((INT) mxGetN(prhs[5]) != N) {
		ERROR("Second dimension of DDt3 matrix does not match dimension number (first dimension of data matrix).");
	} else if ((INT) mxGetM(prhs[6]) != N) {
		ERROR("First dimension of VL matrix does not match dimension number (first dimension of data matrix).");
	} else if ((INT) mxGetN(prhs[6]) != N) {
		ERROR("Second dimension of VL matrix does not match dimension number (first dimension of data matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[7]) != N) {
		ERROR("Number of elements of L matrix does not match dimension number (first dimension of data matrix).");
	}	
	
	plhs[0] = mxCreateNumericMatrix(M, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Xopt = (DOUBLE *) mxGetData(plhs[0]);
	
	minimize_eig_lsqr_smalldata(Xopt, X, length, Xdata, Y, DDt2, DDt3, VL, L, alphaReg, betaReg, M, N, numSamples);
}
