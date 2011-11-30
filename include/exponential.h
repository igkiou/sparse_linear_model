/*
 * link_func.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __LINK_FUNC_H__
#define __LINK_FUNC_H__

#include "useblas.h"

typedef enum {
	EXPONENTIAL_POISSON = 0,
	EXPONENTIAL_BERNOULLI = 1,
	EXPONENTIAL_INVALID = - 1
} EXPONENTIAL_TYPE;

EXPONENTIAL_TYPE convertExponentialName(CHAR exponentialName);

void link_func(DOUBLE *aVal, DOUBLE *aPrime, DOUBLE *aDoublePrime, DOUBLE *X, \
			INT N, INT numSamples, EXPONENTIAL_TYPE family, INT valFlag, \
			INT primeFlag, INT doublePrimeFlag);

void link_func_poisson(DOUBLE *aVal, DOUBLE *aPrime, DOUBLE *aDoublePrime, \
					DOUBLE *X, INT N, INT numSamples, INT valFlag, \
					INT primeFlag, INT doublePrimeFlag);

void link_func_bernoulli(DOUBLE *aVal, DOUBLE *aPrime, DOUBLE *aDoublePrime, \
					DOUBLE *X, INT N, INT numSamples, INT valFlag, \
					INT primeFlag, INT doublePrimeFlag);

void link_func_dual(DOUBLE *aVal, DOUBLE *aPrime, DOUBLE *aDoublePrime, \
			DOUBLE *X, INT N, INT numSamples, EXPONENTIAL_TYPE family, \
			INT valFlag, INT primeFlag, INT doublePrimeFlag);

void link_func_dual_poisson(DOUBLE *aVal, DOUBLE *aPrime, \
					DOUBLE *aDoublePrime, DOUBLE *X, INT N, INT numSamples, \
					INT valFlag, INT primeFlag, INT doublePrimeFlag);

void link_func_dual_bernoulli(DOUBLE *aVal, DOUBLE *aPrime, \
					DOUBLE *aDoublePrime, DOUBLE *X, INT N, INT numSamples, \
					INT valFlag, INT primeFlag, INT doublePrimeFlag);

#endif /* __LINK_FUNC_H__ */
