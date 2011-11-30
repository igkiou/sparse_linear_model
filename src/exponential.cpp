#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "useinterfaces.h"
#include "useblas.h"
#include "exponential.h"

/*
 * TODO: Check again for parallelization of these, and make sure they are thread
 * save.
 * TODO: Replace log(1 + x) with log1p(x).
 */

EXPONENTIAL_TYPE convertExponentialName(CHAR exponentialName) {
	if ((exponentialName == 'P') || (exponentialName == 'p')) {
		return EXPONENTIAL_POISSON;
	} else if ((exponentialName == 'B') || (exponentialName == 'b')) {
		return EXPONENTIAL_BERNOULLI;
	} else {
		ERROR("Unknown exponential family name.");
		return EXPONENTIAL_INVALID;
	}
}

void link_func(DOUBLE *aVal, DOUBLE *aPrime, DOUBLE *aDoublePrime, DOUBLE *X, \
			INT N, INT numSamples, EXPONENTIAL_TYPE family, INT valFlag, \
			INT primeFlag, INT doublePrimeFlag) {

	if (family == EXPONENTIAL_POISSON) {
		link_func_poisson(aVal, aPrime, aDoublePrime, X, N, numSamples,\
						valFlag, primeFlag, doublePrimeFlag);
	} else if (family == EXPONENTIAL_BERNOULLI) {
		link_func_bernoulli(aVal, aPrime, aDoublePrime, X, N, numSamples,\
						valFlag, primeFlag, doublePrimeFlag);
	} else {
		ERROR("Unknown exponential family distribution.");
	}
	
}

void link_func_poisson(DOUBLE *aVal, DOUBLE *aPrime, DOUBLE *aDoublePrime, \
					DOUBLE *X, INT N, INT numSamples, INT valFlag, \
					INT primeFlag, INT doublePrimeFlag) {

	INT iterX;
	INT iterN;
	DOUBLE tempVal;
	if ((valFlag == 1) || (primeFlag == 1) || (doublePrimeFlag == 1)) {

/*
		#pragma omp parallel for private(iterX, iterN, tempVal) shared(X, aVal, aPrime, aDoublePrime) \
				firstprivate(N, numSamples, valFlag, primeFlag, doublePrimeFlag)
*/
		for (iterX = 0; iterX < numSamples; ++iterX) {
			aVal[iterX] = 0;
			for (iterN = 0; iterN < N; ++iterN) {
				tempVal = EXP(X[iterX * N + iterN]);
				if (valFlag == 1) {
					aVal[iterX] += tempVal;
				}

				if (primeFlag == 1) {
					aPrime[iterX * N + iterN] = tempVal;
				}

				if (doublePrimeFlag == 1) {
					aDoublePrime[iterX * N + iterN] = tempVal;
				}
			}
		}
	}
}

void link_func_bernoulli(DOUBLE *aVal, DOUBLE *aPrime, DOUBLE *aDoublePrime, \
					DOUBLE *X, INT N, INT numSamples, INT valFlag, \
					INT primeFlag, INT doublePrimeFlag) {

	INT iterX;
	INT iterN;
	DOUBLE tempVal;
	DOUBLE tempVal2;
	if ((valFlag == 1) || (primeFlag == 1) || (doublePrimeFlag == 1)) {

/*
		#pragma omp parallel for private(iterX, iterN, tempVal, tempVal2) shared(X, aVal, aPrime, aDoublePrime) \
				firstprivate(N, numSamples, valFlag, primeFlag, doublePrimeFlag)
*/
		for (iterX = 0; iterX < numSamples; ++iterX) {
			aVal[iterX] = 0;
			for (iterN = 0; iterN < N; ++iterN) {
				tempVal = EXP(X[iterX * N + iterN]);
				if (valFlag == 1) {
					aVal[iterX] += LOG(tempVal + 1);
				}

				if (primeFlag == 1) {
					aPrime[iterX * N + iterN] = tempVal / (tempVal + 1);
				}

				if (doublePrimeFlag == 1) {
					tempVal2 = tempVal / (tempVal + 1);
					aDoublePrime[iterX * N + iterN] = tempVal2 - SQR(tempVal2);
				}
			}
		}
	}
}

void link_func_dual(DOUBLE *aVal, DOUBLE *aPrime, DOUBLE *aDoublePrime, \
					DOUBLE *X, INT N, INT numSamples, EXPONENTIAL_TYPE family, \
					INT valFlag, INT primeFlag, INT doublePrimeFlag) {

	if ((family == 'P') || (family == 'p')) {
		link_func_dual_poisson(aVal, aPrime, aDoublePrime, X, N, numSamples, \
							valFlag, primeFlag, doublePrimeFlag);
	} else if ((family == 'B') || (family == 'b')) {
		link_func_dual_bernoulli(aVal, aPrime, aDoublePrime, X, N, numSamples, \
							valFlag, primeFlag, doublePrimeFlag);
	} else {
		ERROR("Unknown exponential family distribution.");
	}
}

void link_func_dual_poisson(DOUBLE *aVal, DOUBLE *aPrime, \
					DOUBLE *aDoublePrime, DOUBLE *X, INT N, INT numSamples, \
					INT valFlag, INT primeFlag, INT doublePrimeFlag) {

	INT iterX;
	INT iterN;
	DOUBLE currX;
	DOUBLE tempVal;
	if ((valFlag == 1) || (primeFlag == 1) || (doublePrimeFlag == 1)) {
/*
		#pragma omp parallel for private(iterX, iterN, tempVal) shared(X, aVal, aPrime, aDoublePrime) \
				firstprivate(N, numSamples, valFlag, primeFlag, doublePrimeFlag)
*/
		for (iterX = 0; iterX < numSamples; ++iterX) {
			aVal[iterX] = 0;
			for (iterN = 0; iterN < N; ++iterN) {
				currX = X[iterX * N + iterN];
				tempVal = LOG(currX);
				if (valFlag == 1) {
					aVal[iterX] += currX * tempVal - currX;
				}

				if (primeFlag == 1) {
					aPrime[iterX * N + iterN] = tempVal;
				}

				if (doublePrimeFlag == 1) {
					aDoublePrime[iterX * N + iterN] = 1 / currX;
				}
			}
		}
	}
}

void link_func_dual_bernoulli(DOUBLE *aVal, DOUBLE *aPrime, \
					DOUBLE *aDoublePrime, DOUBLE *X, INT N, INT numSamples, \
					INT valFlag, INT primeFlag, INT doublePrimeFlag) {

	INT iterX;
	INT iterN;
	DOUBLE currX;
	DOUBLE tempVal;
	DOUBLE tempValSym;
	if ((valFlag == 1) || (primeFlag == 1) || (doublePrimeFlag == 1)) {
/*
		#pragma omp parallel for private(iterX, iterN, tempVal, tempVal2) shared(X, aVal, aPrime, aDoublePrime) \
				firstprivate(N, numSamples, valFlag, primeFlag, doublePrimeFlag)
*/
		for (iterX = 0; iterX < numSamples; ++iterX) {
			aVal[iterX] = 0;
			for (iterN = 0; iterN < N; ++iterN) {
				currX = X[iterX * N + iterN];
				tempVal = LOG(currX);
				tempValSym = LOG(1 - currX);
				if (valFlag == 1) {
					aVal[iterX] += currX * tempVal + (1 - currX) * tempValSym;
				}

				if (primeFlag == 1) {
					aPrime[iterX * N + iterN] = tempVal - tempValSym;
				}

				if (doublePrimeFlag == 1) {
					aDoublePrime[iterX * N + iterN] = 1 / (currX * (1 - currX));
				}
			}
		}
	}
}
