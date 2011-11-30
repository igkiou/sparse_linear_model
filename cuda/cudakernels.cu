/*
 * cudakernels.c
 *
 *  Created on: Nov 7, 2011
 *      Author: igkiou
 */

#include "cudakernels.h"
#include "usecuda.h"

#define IMIN(X, Y)  ((X) < (Y) ? (X) : (Y))
#define IMAX(X, Y)  ((X) > (Y) ? (X) : (Y))
#define SIGN(X)  ((X) > 0 ? (1) : (((X) < 0 ? (-(1)) : (0))))
#define ABS(X)  ((X) > 0 ? (X) : (-(X)))

__global__ void cuSoftThreshold_sub(CUDOUBLE *X, CUDOUBLE tau, CUINT N) {
	CUINT iterN = blockDim.x * blockIdx.x + threadIdx.x;

//	if (iterN < N) {
//		if ((threshTemp = ABS(X[iterN]) - tau) > 0) {
//			X[iterN] = SIGN(X[iterN]) * threshTemp;
//		} else {
//			X[iterN] = 0;
//		}
//	}

	iterN = IMAX(iterN, 0);
	iterN = IMIN(iterN, N);
	CUDOUBLE threshTemp = ABS(X[iterN]) - tau;
	(threshTemp > 0) ? (X[iterN] = SIGN(X[iterN]) * threshTemp) \
						: (X[iterN] = 0);
}

void cuSoftThreshold(CUDOUBLE *X, CUDOUBLE tau, CUINT N) {
	CUINT threadsPerBlock = 8;
	CUINT blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
	cuSoftThreshold_sub<<<blocksPerGrid, threadsPerBlock>>>(X, tau, N);
}
