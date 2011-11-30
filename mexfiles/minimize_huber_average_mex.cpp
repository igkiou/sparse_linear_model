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
	
	DOUBLE *Phi = (DOUBLE*) mxGetData(prhs[0]);
	INT length =  (INT)*(DOUBLE*) mxGetData(prhs[1]); 
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *Y = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *weights = (DOUBLE*) mxGetData(prhs[4]);
	DOUBLE *bias = (DOUBLE*) mxGetData(prhs[5]);
	
    INT Mb = (INT) mxGetM(prhs[4]);
	INT Nb = (INT) mxGetM(prhs[2]);
	INT numSamples = (INT) mxGetN(prhs[2]);
	INT numTasks = (INT) mxGetN(prhs[4]);
	INT MN = (INT) mxGetNumberOfElements(prhs[0]);
	DOUBLE numPoolD = SQRT((DOUBLE) Mb * (DOUBLE) Nb / (DOUBLE) MN);
	INT numPool;
	INT M;
	INT N;
	
	if (ceil(numPoolD) != numPoolD) {
		ERROR("Number of elements of Phi is not a divisor of the product of feature number (first dimension of weight matrix) times dimension number (first dimension of data matrix).");
	} else {
		numPool = (INT) numPoolD;
		M = Mb / numPool;
		N = Nb / numPool;
	}
	if ((INT) mxGetM(prhs[3]) != numSamples) {
		ERROR("First dimension of label matrix does not match number of samples (second dimension of data matrix).");
	} else if ((INT) mxGetN(prhs[3]) != numTasks) {
		ERROR("Second dimension of label matrix does not match number of tasks (second dimension of weight matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[5]) != numTasks) {
		ERROR("Number of elements of bias matrix does not match number of tasks (second dimension of weight matrix).");
	}
	
	plhs[0] = mxCreateNumericMatrix(M * N, 1, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Phiopt = (DOUBLE *) mxGetData(plhs[0]);

	minimize_huber_average(Phiopt, Phi, length, X, Y, weights, bias, M, N, numSamples, numPool, numTasks);
}
