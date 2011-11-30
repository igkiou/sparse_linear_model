/*
 * metric_learning.h
 *
 *  Created on: Jul 14, 2011
 *      Author: igkiou
 */

#ifndef __METRIC_LEARNING_H__
#define __METRIC_LEARNING_H__

#include "useblas.h"

typedef enum {
	DIST_LABEL_BOUNDS = 0,
	DIST_LABEL_TARGETS = 1,
	DIST_LABEL_RELATIONAL = 2,
	DIST_LABEL_SQRHINGE = 3,
	DIST_LABEL_HUBERHINGE = 4,
	DIST_LABEL_INVALID = - 1
} DIST_LABEL_TYPE;

DIST_LABEL_TYPE convertDistLabelName(CHAR distLabelName);

void convertDistanceLabelMat(INT *labelMat, DOUBLE *valVec, \
							DIST_LABEL_TYPE labelType, DOUBLE *matlabLabelMat, \
							INT numLabels, INT numSamples);

void getDistanceViolations(DOUBLE *violations, INT *labelMat, DOUBLE *valVec, \
							DIST_LABEL_TYPE labelType, INT numLabels);

void itml(DOUBLE *A, DOUBLE *X, INT *constraintMat, DOUBLE *boundVec, \
		DOUBLE tolerance, DOUBLE gamma, INT maxEpochs, INT randomizationFlag, \
		INT N, INT numConstraints, INT *constraintPerm, DOUBLE *lambda, \
		DOUBLE *lambdaOld, DOUBLE *vec, DOUBLE *Avec);

void itkml(DOUBLE *A, DOUBLE *K, INT *constraintMat, DOUBLE *boundVec, \
		DOUBLE *Phi, INT factorizedFlag, DOUBLE tolerance, DOUBLE gamma, \
		INT maxEpochs, INT randomizationFlag, INT numSamples, \
		INT numConstraints, INT *constraintPerm, DOUBLE *lambda, \
		DOUBLE *lambdaOld, DOUBLE *vec);

void legoUpdate(DOUBLE *A, DOUBLE trueDist, DOUBLE eta, \
			DOUBLE *x1, DOUBLE *x2, INT N, DOUBLE *vec, DOUBLE *Avec);

void lego(DOUBLE *A, DOUBLE *X, INT *pairMat, DOUBLE *distVec, \
		DOUBLE tolerance, DOUBLE eta0, INT maxEpochs, INT randomizationFlag, \
		INT continuationFlag, INT N, INT numPairs, DOUBLE *AOld, INT *pairPerm, \
		DOUBLE *vec, DOUBLE *Avec);

void nrml_fpc_continuation(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE tau, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, DOUBLE tauMultiplier, \
			DOUBLE tauRate, INT N, INT numConstraints);

void nrml_fpc(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE* tau, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, INT N, INT numConstraints, \
			INT numRepeats);

void nrml_fp_inner(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE tauTarget, \
			DOUBLE delta, INT numIters, DOUBLE tolerance, DOUBLE tauMultiplier, \
			DOUBLE tauRate, INT N, INT numConstraints, DOUBLE *AOld, \
			DOUBLE *AOldVec, DOUBLE *vec1, DOUBLE *vec2, DOUBLE *Vr, \
			INT *isuppz, DOUBLE *work, INT lwork);

void nrml_apg_continuation(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE tau, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, \
			DOUBLE tauMultiplier, DOUBLE tauRate, INT N, INT numConstraints);

void nrml_apg(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE *tau, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, \
			INT N, INT numConstraints, INT numRepeats);

void nrml_apg_inner(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE tauTarget, \
			DOUBLE delta0, INT numIters, DOUBLE tolerance, INT lineSearchFlag, \
			DOUBLE eta, DOUBLE tauMultiplier, DOUBLE tauRate, INT N, \
			INT numConstraints, DOUBLE *AOld, DOUBLE *AOldVec, DOUBLE *L, \
			DOUBLE *LfGrad, DOUBLE *LfGD, DOUBLE *ALfGDDiff, DOUBLE *vec1, \
			DOUBLE *vec2, DOUBLE *Vr, INT *isuppz, DOUBLE *work, INT lwork);

void nrml_line_search(DOUBLE *delta, DOUBLE *LfGDShrink, DOUBLE *L, \
		DOUBLE LfObj, DOUBLE *LfGrad, DOUBLE tau, INT lineSearchFlag, \
		DOUBLE eta, DOUBLE *X, DIST_LABEL_TYPE problemType, INT *constraintMat, \
		DOUBLE *betaVec, INT N, INT numConstraints, DOUBLE *LfGD, \
		DOUBLE *ALfGDDiff, DOUBLE *AOldVec, DOUBLE *vec1, DOUBLE *vec2, \
		DOUBLE *Vr, DOUBLE *work, INT lwork);

void nrml_shrinkage(DOUBLE *AShrink, DOUBLE *AShrinkNorm, DOUBLE *A, \
				DOUBLE tau, INT N, DOUBLE *lVec, DOUBLE *Vr, \
				DOUBLE *work, INT lwork);

void nrml_Q_func_mod(DOUBLE *QObj, DOUBLE *A, DOUBLE *LfGD, DOUBLE LfObj, \
			DOUBLE *LfGrad, DOUBLE delta, INT N, DOUBLE *ALfGDDiff);

void frml_pgd(DOUBLE *A, DOUBLE *X, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE kappa, \
			DOUBLE delta0, INT numIters, DOUBLE tolerance, INT stepFlag, INT N, \
			INT numConstraints);

void frml_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, DOUBLE *X, \
		DIST_LABEL_TYPE problemType, INT *constraintMat, DOUBLE *betaVec, \
		DOUBLE kappa, INT N, INT numConstraints, DOUBLE *Avec, DOUBLE *vec1, \
		DOUBLE *vec2);

void ml_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, DOUBLE *X, \
			DIST_LABEL_TYPE problemType, INT *constraintMat, DOUBLE *betaVec, \
			INT N, INT numConstraints, DOUBLE *Avec, DOUBLE *vec1, \
			DOUBLE *vec2);

void ml_target_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, DOUBLE *X, \
			INT *constraintMat, DOUBLE *targetVec, INT N, INT numConstraints, \
			DOUBLE *Avec, DOUBLE *vec);

void ml_relational_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, \
			DOUBLE *X, INT *constraintMat, DOUBLE *marginVec, INT N, \
			INT numConstraints, DOUBLE *Avec, DOUBLE *vecij, DOUBLE *veckl);

void ml_sqrhinge_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, \
			DOUBLE *X, INT *constraintMat, DOUBLE *marginVec, INT N, \
			INT numConstraints, DOUBLE *Avec, DOUBLE *vecij, DOUBLE *veckl);

void ml_bound_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *A, \
			DOUBLE *X, INT *constraintMat, DOUBLE *boundVec, INT N, \
			INT numConstraints, DOUBLE *Avec, DOUBLE *vec);

#endif /* __METRIC_LEARNING_H__ */
