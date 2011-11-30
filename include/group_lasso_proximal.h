/*
 * group_lasso_proximal.h
 *
 *  Created on: Aug 24, 2011
 *      Author: igkiou
 */

#ifndef __GROUP_LASSO_PROXIMAL_H__
#define __GROUP_LASSO_PROXIMAL_H__

#include "useblas.h"

void convertSetLabelMat(INT *locality, DOUBLE *orig, INT *setSizes, INT K, \
						INT numSets);

void copyDictionaryToLocality(DOUBLE *XOut, DOUBLE *XIn, INT *locality, INT N, \
							INT K);

void copyCodeToOrig(DOUBLE *XOut, DOUBLE *XIn, INT *locality, INT K, \
						INT numSamples);

void copyCodeToLocality(DOUBLE *XOut, DOUBLE *XIn, INT *locality, INT K, \
						INT numSamples);

void group_lasso_proximal(DOUBLE *X, DOUBLE *norm, DOUBLE tau, INT *setSizes, \
						DOUBLE *setWeights, INT numSets);

void group_lasso_ista(DOUBLE *A, DOUBLE *X, DOUBLE *D, DOUBLE *lambda, \
			INT *setSizes, DOUBLE *setWeights, DOUBLE delta, INT numIters, \
			DOUBLE tolerance, INT N, INT K, INT numSamples, INT numSets, \
			INT numRepeats);

void group_lasso_ista_inner(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE tau, \
			INT *setSizes, DOUBLE *setWeights, DOUBLE delta, INT numIters, \
			DOUBLE tolerance, INT K, INT numSets, DOUBLE *AOld, DOUBLE *KDDA);

void group_lasso_fista(DOUBLE *A, DOUBLE *X, DOUBLE *D, DOUBLE *lambda, \
			INT *setSizes, DOUBLE *setWeights, DOUBLE delta0, INT numIters, \
			DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, INT N, INT K, \
			INT numSamples, INT numSets, INT numRepeats);

void group_lasso_fista_inner(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE tau, \
			INT *setSizes, DOUBLE *setWeights, DOUBLE delta0, INT numIters, \
			DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, INT K, \
			INT numSets, DOUBLE *AOld, DOUBLE *KDDA, DOUBLE *L, DOUBLE *LfGrad, \
			DOUBLE *LfGD, DOUBLE *ALfGDDiff);

void group_lasso_line_search(DOUBLE *delta, DOUBLE *LfGDShrink, DOUBLE *L, \
		DOUBLE LfObj, DOUBLE *LfGrad, DOUBLE tau, INT *setSizes, \
		DOUBLE *setWeights, INT lineSearchFlag, DOUBLE eta, DOUBLE *KDX, \
		DOUBLE *KDD, INT K, INT numSets, DOUBLE *LfGD, DOUBLE *ALfGDDiff, \
		DOUBLE *KDDA);

void group_lasso_shrinkage(DOUBLE *AShrink, DOUBLE *AShrinkNorm, DOUBLE *A, \
				DOUBLE tau, INT *setSizes, DOUBLE *setWeights, INT K, \
				INT numSets);

void group_lasso_Q_func_mod(DOUBLE *QObj, DOUBLE *A, DOUBLE *LfGD, DOUBLE LfObj, \
			DOUBLE *LfGrad, DOUBLE delta, INT K, DOUBLE *ALfGDDiff);

#endif /* __GROUP_LASSO_PROXIMAL_H__ */
