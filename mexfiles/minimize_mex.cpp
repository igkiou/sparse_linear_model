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
	
    INT M = (INT) mxGetM(prhs[0]);
	INT N = (INT) mxGetN(prhs[0]);
	
	if ((INT) mxGetM(prhs[2]) != N) {
		ERROR("First dimension of DDt2 matrix does not match dimension size (second dimension of Phi matrix).");
	} else if ((INT) mxGetN(prhs[2]) != N) {
		ERROR("Second dimension of DDt2 matrix does not match dimension size (second dimension of Phi matrix).");
	} else if ((INT) mxGetM(prhs[3]) != N) {
		ERROR("First dimension of DDt3 matrix does not match dimension size (second dimension of Phi matrix).");
	} else if ((INT) mxGetN(prhs[3]) != N) {
		ERROR("Second dimension of DDt3 matrix does not match dimension size (second dimension of Phi matrix).");
	} else if ((INT) mxGetM(prhs[4]) != N) {
		ERROR("First dimension of VL matrix does not match dimension size (second dimension of Phi matrix).");
	} else if ((INT) mxGetN(prhs[4]) != N) {
		ERROR("Second dimension of VL matrix does not match dimension size (second dimension of Phi matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[5]) != N) {
		ERROR("Number of elements of L matrix does not match dimension size (second dimension of Phi matrix).");
	}	

#ifdef __DEBUG__
	PRINTF("Printing M: %d, N: %d, length: %d\n\n", M, N, length);
	PRINTF("Printing X:\n");
	print_matrix(X, M, N);
	PRINTF("\n");
	PRINTF("Printing DDt2:\n");
	print_matrix(DDt2, N, N);
	PRINTF("\n");
	PRINTF("Printing DDt3:\n");
	print_matrix(DDt3, N, N);
	PRINTF("\n");
	PRINTF("Printing VL:\n");
	print_matrix(VL, N, N);
	PRINTF("\n");
	PRINTF("Printing L:\n");
	print_matrix(L, N, N);
	PRINTF("\n");
#endif
	
	plhs[0] = mxCreateNumericMatrix(M, N, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Xopt = (DOUBLE *) mxGetData(plhs[0]);
	
	minimize(Xopt, X, length, DDt2, DDt3, VL, L, M, N);
	
}
