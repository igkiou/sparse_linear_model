/*
 * projection.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __PROJECTION_H__
#define __PROJECTION_H__

#include "useblas.h"

/* TODO: Implement l1 projection. */
/* TODO: Implement linf projection. */
/* TODO: Add projection to simplex, or positive hyperplane. */
/* TODO: Also add general wrapper mex l1 projection. */
/* TODO: Extend to matrix projections? */
typedef enum {
	PROJECTION_L2 = 0,
	PROJECTION_KERNEL = 2,
	PROJECTION_L1 = 3,
	PROJECTION_INVALID = - 1
} PROJECTION_TYPE;

PROJECTION_TYPE convertProjectionName(CHAR projectionName);

void sphere_projection(DOUBLE *dataMat, DOUBLE *normVec, \
					PROJECTION_TYPE projectionType, DOUBLE radius, INT N, \
					INT K);

void ball_projection(DOUBLE *dataMat, DOUBLE *normVec, \
					PROJECTION_TYPE projectionType, DOUBLE radius, INT N, \
					INT K);

void l2_sphere_projection(DOUBLE *D, DOUBLE *norm, DOUBLE radius, INT N);

void l2_sphere_projection_batch(DOUBLE *D, DOUBLE *norm, DOUBLE radius, INT N, \
							INT K);

void l2_ball_projection(DOUBLE *D, DOUBLE *norm, DOUBLE radius, INT N);

void l2_ball_projection_batch(DOUBLE *D, DOUBLE *norm, DOUBLE radius, INT N, \
							INT K);

void kernel_sphere_projection_batch(DOUBLE *kernelMat, DOUBLE *norm, DOUBLE radius, \
							INT K);

void kernel_ball_projection_batch(DOUBLE *kernelMat, DOUBLE *norm, DOUBLE radius, \
						INT K);

#endif /* __PROJECTION_H__ */
