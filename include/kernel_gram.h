/*
 * kernel_gram.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __KERNEL_GRAM_H__
#define __KERNEL_GRAM_H__

/*
 * TODO: These functions can benefit tremendously from VML.
 */

#include "useblas.h"

typedef enum {
	KERNEL_LINEAR = 0,
	KERNEL_GAUSSIAN = 1,
	KERNEL_POLYNOMIAL = 2,
	KERNEL_SOBOLEV = 4,
	KERNEL_INVALID = - 1
} KERNEL_TYPE;

KERNEL_TYPE convertKernelName(CHAR kernelName);

void kernel_gram(DOUBLE *kernelMat, DOUBLE *X1, DOUBLE *X2, INT N, \
		INT numSamples1, INT numSamples2, KERNEL_TYPE kernelType, \
		DOUBLE *pparam1, DOUBLE *pparam2, DOUBLE *normMat1, DOUBLE *oneVec);

void kernel_gram_linear(DOUBLE *kernelMat, DOUBLE *X1, DOUBLE *X2, INT N, \
					INT numSamples1, INT numSamples2);

void kernel_gram_gaussian(DOUBLE *kernelMat, DOUBLE *X1, DOUBLE *X2, INT N, \
			INT numSamples1, INT numSamples2, DOUBLE *pparam1, \
			DOUBLE *normMat1, DOUBLE *oneVec);

void kernel_gram_polynomial(DOUBLE *kernelMat, DOUBLE *X1, DOUBLE *X2, INT N, \
			INT numSamples1, INT numSamples2, DOUBLE *pparam1, \
			DOUBLE *pparam2);

void kernel_gram_sobolev(DOUBLE *kernelMat, DOUBLE *X1, DOUBLE *X2, INT N, \
			INT numSamples1, INT numSamples2, DOUBLE *pparam1);

#endif /* __KERNEL_GRAM_H__ */
