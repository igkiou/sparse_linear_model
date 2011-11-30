/*
#define __DEBUG__
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useinterfaces.h"
#include "useblas.h"
#include "projection.h"

PROJECTION_TYPE convertProjectionName(CHAR projectionName) {
	if (projectionName == '2') {
		return PROJECTION_L2;
	} else if (projectionName == '1') {
		return PROJECTION_L1;
	} else if ((projectionName == 'K') || (projectionName == 'k')) {
		return PROJECTION_KERNEL;
	} else {
		ERROR("Unknown projection name.");
		return PROJECTION_INVALID;
	}
}

void sphere_projection(DOUBLE *dataMat, DOUBLE *normVec, \
					PROJECTION_TYPE projectionType, DOUBLE radius, INT N, \
					INT K) {

	if (projectionType == PROJECTION_L2) {
		l2_sphere_projection_batch(dataMat, normVec, radius, N, K);
	} else if (projectionType == PROJECTION_L1) {
		ERROR("L1 projection not implemented yet.");
	} else if (projectionType == PROJECTION_KERNEL) {
		kernel_sphere_projection_batch(dataMat, normVec, radius, K);
	} else {
		ERROR("Unknown projection type.");
	}
}

void ball_projection(DOUBLE *dataMat, DOUBLE *normVec, \
					PROJECTION_TYPE projectionType, DOUBLE radius, INT N, \
					INT K) {

	if (projectionType == PROJECTION_L2) {
		l2_ball_projection_batch(dataMat, normVec, radius, N, K);
	} else if (projectionType == PROJECTION_L1) {
		ERROR("L1 projection not implemented yet.");
	} else if (projectionType == PROJECTION_KERNEL) {
		kernel_ball_projection_batch(dataMat, normVec, radius, K);
	} else {
		ERROR("Unknown projection type.");
	}
}


/* TODO: Parallelize these with OpenMP. */
void l2_sphere_projection(DOUBLE *D, DOUBLE *norm, DOUBLE radius, INT N) {
	
	DOUBLE normtemp;
	DOUBLE alpha;
	INT incx = 1;
	
	normtemp = NRM2(&N, D, &incx);
	if (normtemp != 0) {
		alpha = radius / normtemp;
		SCAL(&N, &alpha, D, &incx);
	}
	if (norm != NULL) {
		*norm = normtemp;
	}
}

void l2_sphere_projection_batch(DOUBLE *D, DOUBLE *norm, DOUBLE radius, INT N, \
								INT K) {

	INT iterK;
	if (norm == NULL) {
		for (iterK = 0; iterK < K; ++iterK) {
			l2_sphere_projection(&D[iterK * N], NULL, radius, N);
		}
	} else {
		for (iterK = 0; iterK < K; ++iterK) {
			l2_sphere_projection(&D[iterK * N], &norm[iterK], radius, N);
		}
	}
}

void l2_ball_projection(DOUBLE *D, DOUBLE *norm, DOUBLE radius, INT N) {

	DOUBLE normtemp;
	DOUBLE alpha;
	INT incx = 1;

	normtemp = NRM2(&N, D, &incx);
	if (normtemp > radius) {
		alpha = radius / normtemp;
		SCAL(&N, &alpha, D, &incx);
	}
	if (norm != NULL) {
		*norm = normtemp;
	}
}

void l2_ball_projection_batch(DOUBLE *D, DOUBLE *norm, DOUBLE radius, INT N, \
							INT K) {
	
	INT iterK;
	if (norm == NULL) {
		for (iterK = 0; iterK < K; ++iterK) {
			l2_ball_projection(&D[iterK * N], NULL, radius, N);
		}
	} else {
		for (iterK = 0; iterK < K; ++iterK) {
			l2_ball_projection(&D[iterK * N], &norm[iterK], radius, N);
		}
	}
}

void kernel_sphere_projection_batch(DOUBLE *kernelMat, DOUBLE *norm, \
									DOUBLE radius, INT K) {
	DOUBLE *normVec = norm;
	INT normVecFlag = 0;
	if (norm == NULL) {
		normVec = (DOUBLE *) MALLOC(K * sizeof(DOUBLE));
		normVecFlag = 1;
	}

	INT iterK;
	DOUBLE alpha;
	INT incx = 1;
	for (iterK = 0; iterK < K; ++iterK) {
		normVec[iterK] = SQRT(kernelMat[iterK * K + iterK]);
		if (normVec[iterK] != 0) {
			alpha = radius / normVec[iterK];
			SCAL(&K, &alpha, &kernelMat[iterK * K], &incx);
		}
	}

	incx = K;
	for (iterK = 0; iterK < K; ++iterK) {
		if (normVec[iterK] != 0) {
			alpha = radius / normVec[iterK];
			SCAL(&K, &alpha, &kernelMat[iterK], &incx);
		}
	}

	if (normVecFlag == 1) {
		FREE(normVec);
	}
}

void kernel_ball_projection_batch(DOUBLE *kernelMat, DOUBLE *norm, \
								DOUBLE radius, INT K) {
	DOUBLE *normVec = norm;
	INT normVecFlag = 0;
	if (norm == NULL) {
		normVec = (DOUBLE *) MALLOC(K * sizeof(DOUBLE));
		normVecFlag = 1;
	}

	INT iterK;
	DOUBLE alpha;
	INT incx = 1;
	for (iterK = 0; iterK < K; ++iterK) {
		normVec[iterK] = SQRT(kernelMat[iterK * K + iterK]);
		if (normVec[iterK] > radius) {
			alpha = radius / normVec[iterK];
			SCAL(&K, &alpha, &kernelMat[iterK * K], &incx);
		}
	}

	incx = K;
	for (iterK = 0; iterK < K; ++iterK) {
		if (normVec[iterK] > radius) {
			alpha = radius / normVec[iterK];
			SCAL(&K, &alpha, &kernelMat[iterK], &incx);
		}
	}

	if (normVecFlag == 1) {
		FREE(normVec);
	}
}
