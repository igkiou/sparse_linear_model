/*
 * distance.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __DISTANCES_H__
#define __DISTANCES_H__

#include "useblas.h"

/* TODO: Also add l-inf distance. */
/* TODO: Extend to matrix distances? */
/* TODO: Also add BLAS wrappers for norms? */
/* TODO: Also add matrix norms? */
/* TODO: These functions can benefit tremendously from VML. */
typedef enum {
	DISTANCE_L2 = 0,
	DISTANCE_MAHALANOBIS = 1,
	DISTANCE_KERNEL = 2,
	DISTANCE_L1 = 3,
	DISTANCE_INVALID = - 1
} DISTANCE_TYPE;

DISTANCE_TYPE convertDistanceName(char distanceName);

DOUBLE dotw(INT *n, DOUBLE *x, INT *incx, DOUBLE *y, INT *incy, DOUBLE *A, \
			DOUBLE *Ax);

DOUBLE nrm2w(INT *n, DOUBLE *x, INT *incx, DOUBLE *A, DOUBLE *Ax);

void quadform(DOUBLE *XtAX, DOUBLE *X, DOUBLE *A, INT M, INT N, DOUBLE alpha, \
		DOUBLE beta, INT transposeFlag, DOUBLE *AX);

void l2_distance(DOUBLE *distanceMat, DOUBLE *X1, DOUBLE *X2, INT N, INT numSamples1, INT numSamples2, INT sqrtFlag, \
					DOUBLE *normMat1, DOUBLE *oneVec);

DOUBLE l2_distance_vec(DOUBLE *x1, DOUBLE *x2, INT N, INT sqrtFlag, \
						DOUBLE *tempVec);

void mahalanobis_distance(DOUBLE *distanceMat, DOUBLE *X1, DOUBLE *X2, DOUBLE *A, \
						INT N, INT numSamples1, INT numSamples2, INT sqrtFlag, \
						DOUBLE *tempX1, DOUBLE *tempX2, DOUBLE *normMat1, \
						DOUBLE *oneVec);

DOUBLE mahalanobis_distance_vec(DOUBLE *x1, DOUBLE *x2, DOUBLE *A, INT N, \
							INT sqrtFlag, DOUBLE *tempVec, DOUBLE *tempVec2);

void mahalanobis_distance_factored(DOUBLE *distanceMat, DOUBLE *X1, DOUBLE *X2, \
						DOUBLE *U, INT N, INT numSamples1, INT numSamples2, INT M, \
						INT sqrtFlag, DOUBLE *tempX1, DOUBLE *tempX2, \
						DOUBLE *normMat1, DOUBLE *oneVec);

void kernel_distance(DOUBLE *distanceMat, DOUBLE *K, INT numSamples, INT sqrtFlag, \
					DOUBLE *normMat, DOUBLE *oneVec);

/*void l1_distance(DOUBLE *distanceMat, DOUBLE *X1, DOUBLE *X2, INT N, INT numSamples1, INT numSamples2, INT sqrtFlag, \
					DOUBLE *normMat1, DOUBLE *oneVec);*/

void l1_distance(DOUBLE *distanceMat, DOUBLE *X1, DOUBLE *X2, INT N, \
				INT numSamples1, INT numSamples2);

DOUBLE l1_distance_vec(DOUBLE *x1, DOUBLE *x2, INT N, DOUBLE *tempVec);

#endif /* __DISTANCES_H__ */

