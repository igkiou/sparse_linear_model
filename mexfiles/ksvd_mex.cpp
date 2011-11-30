/*
 * NOTE: Incomplete.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// #include <culapack.h>
#include "mex.h"
#include "matrix.h"
#include "useblas.h"
#include "sparse_classification.h"

void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
	DOUBLE *Deq = (DOUBLE*) mxGetData(prhs[0]);
	DOUBLE *D = (DOUBLE*) mxGetData(prhs[1]);
	DOUBLE *Xeq = (DOUBLE*) mxGetData(prhs[2]);
	DOUBLE *X = (DOUBLE*) mxGetData(prhs[3]);
	DOUBLE *Gamma = (DOUBLE*) mxGetData(prhs[4]);
	DOUBLE *extPhi = (DOUBLE*) mxGetData(prhs[5]);
	DOUBLE *cholPhiLambda = (DOUBLE*) mxGetData(prhs[6]);
	
    INT Norig = (INT) mxGetM(prhs[1]);
	INT K = (INT) mxGetN(prhs[1]);
	INT numSamples = (INT) mxGetN(prhs[3]);
	INT N = (INT) mxGetM(prhs[5]);
	INT numMeasurements;
	
	if (N != 0) {
		numMeasurements = N - Norig;
	} else {
		numMeasurements = 0;
		N = Norig;
	}
	
	plhs[0] = mxCreateNumericMatrix(Norig, K, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Dout = (DOUBLE *) mxGetData(plhs[0]);
	datacpy(Dout, D, Norig * K);
	
	plhs[1] = mxCreateNumericMatrix(K, numSamples, MXPRECISION_CLASS, mxREAL);
	DOUBLE *Gammaout = (DOUBLE *) mxGetData(plhs[1]);
	datacpy(Gammaout, Gamma, K * numSamples);
	
	DOUBLE *Deqtemp;
	if (numMeasurements != 0) {
		Deqtemp = (DOUBLE*) MALLOC(N * K * sizeof(DOUBLE));
		datacpy(Deqtemp, Deq, N * K);
	} else {
		Deqtemp = Deq;
	}
	
	ksvd_update(Deqtemp, Dout, Xeq, X, Gammaout, extPhi, cholPhiLambda, Norig, K, numSamples, numMeasurements);
	
	if (numMeasurements != 0) {
		FREE(Deqtemp);
	}
}
