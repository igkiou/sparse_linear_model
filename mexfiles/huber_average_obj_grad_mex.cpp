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
#include <omp.h>
#include "mex.h"
#include "matrix.h"
#include "useblas.h"
#include "sparse_classification.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	/* Check number of input arguments */
	if (nrhs != 5) {
		ERROR("Five input arguments are required.");
    }
	
	/* Check number of output arguments */
	if (nlhs > 2) {
		ERROR("Too many output arguments.");
    } 
	
	DOUBLE *Phi = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *Y = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *weights = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *bias = (DOUBLE*) mxGetData(prhs[4]);
	
    INT Mb = (INT) mxGetM(prhs[3]);
	INT Nb = (INT) mxGetM(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[1]);
	INT numTasks = (INT) mxGetN(prhs[3]);
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
	if ((INT) mxGetM(prhs[2]) != numSamples) {
		ERROR("First dimension of label matrix does not match number of samples (second dimension of data matrix).");
	} else if ((INT) mxGetN(prhs[2]) != numTasks) {
		ERROR("Second dimension of label matrix does not match number of tasks (second dimension of weight matrix).");
	} else if ((INT) mxGetNumberOfElements(prhs[4]) != numTasks) {
		ERROR("Number of elements of bias matrix does not match number of tasks (second dimension of weight matrix).");
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
	
	if (numTasks < CUSTOM_OMP_NUM_THREADS) {
		omp_set_num_threads(numTasks);
	}
	huber_average_obj_grad_multitask(obj, deriv, Phi, X, Y, weights, bias, \
					M, N, numSamples, numPool, numTasks, derivFlag);
	
/*
	DOUBLE *Ypred = (DOUBLE *) MALLOC(1 * numSamples * sizeof(DOUBLE));
	DOUBLE *Ytemp = (DOUBLE *) MALLOC(N * 1 * sizeof(DOUBLE));
	huber_obj_grad_for(obj, deriv, Phi, X, Y, weights, bias, \
					M, N, numSamples, 1, Ypred, Ytemp);
	FREE(Ypred);
	FREE(Ytemp);
*/
	
}
