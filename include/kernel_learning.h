/*
 * kernel_learning.h
 *
 *  Created on: Apr 27, 2011
 *      Author: igkiou
 */

#ifndef __KERNEL_LEARNING_H__
#define __KERNEL_LEARNING_H__

#include "useblas.h"
#include "metric_learning.h"

void itkl(DOUBLE *K, INT *constraintMat, DOUBLE *boundVec, \
		DOUBLE tolerance, DOUBLE gamma, INT maxEpochs, INT randomizationFlag, \
		INT numSamples, INT numConstraints, INT *constraintPerm, DOUBLE *lambda, \
		DOUBLE *lambdaOld, DOUBLE *vec);

void nrkl_svt(DOUBLE *K, INT *constraintMat, DOUBLE *betaVec, DOUBLE tau, \
			DOUBLE delta, INT numIters, DOUBLE tolerance, INT numPoints, \
			INT numConstraints);

void nrkl_fpc_continuation(DOUBLE *K, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE *weights, \
			DOUBLE tau, DOUBLE delta, INT numIters, DOUBLE tolerance, \
			DOUBLE tauMultiplier, DOUBLE tauRate, INT numPoints, \
			INT numConstraints);

void nrkl_fpc(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, DOUBLE* tau, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, INT numPoints, INT numConstraints, \
			INT numRepeats);

void nrkl_fp_inner(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, DOUBLE tauTarget, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, DOUBLE tauMultiplier, \
			DOUBLE tauRate, INT numPoints, INT numConstraints, DOUBLE *KOld, \
			DOUBLE *lVec, DOUBLE *Vr, INT *isuppz, DOUBLE *work, INT lwork);

void nrkl_svp(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, INT rank, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, INT numPoints, INT numConstraints);

void nrkl_apg_continuation(DOUBLE *K, DIST_LABEL_TYPE problemType, \
			INT *constraintMat, DOUBLE *betaVec, DOUBLE *weights, \
			DOUBLE tau, DOUBLE delta0, 	INT numIters, DOUBLE tolerance, \
			INT lineSearchFlag, DOUBLE eta, DOUBLE tauMultiplier, \
			DOUBLE tauRate, INT numPoints, INT numConstraints);

void nrkl_apg(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, DOUBLE *tau, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, \
			INT numPoints, INT numConstraints, INT numRepeats);

void nrkl_apg_inner(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, DOUBLE tauTarget, \
			DOUBLE delta0, INT numIters, DOUBLE tolerance, INT lineSearchFlag, \
			DOUBLE eta, DOUBLE tauMultiplier, DOUBLE tauRate, INT numPoints, \
			INT numConstraints, DOUBLE *KOld, DOUBLE *L, DOUBLE *LfGrad, \
			DOUBLE *LfGD, DOUBLE *KLfGDDiff, DOUBLE *lVec, DOUBLE *Vr, \
			INT *isuppz, DOUBLE *work, INT lwork);

void nrkl_line_search(DOUBLE *delta, DOUBLE *LfGDShrink, DOUBLE *L, \
		DOUBLE LfObj, DOUBLE *LfGrad, DOUBLE tau, INT lineSearchFlag, \
		DOUBLE eta, DIST_LABEL_TYPE problemType, INT *constraintMat, \
		DOUBLE *betaVec, DOUBLE *weights, INT numPoints, INT numConstraints, \
		DOUBLE *LfGD, DOUBLE *KLfGDDiff, DOUBLE *lVec, DOUBLE *Vr, DOUBLE *work, \
		INT lwork);

void nrkl_shrinkage(DOUBLE *KShrink, DOUBLE *KShrinkNorm, DOUBLE *K, \
				DOUBLE tau, INT numPoints, DOUBLE *lvec, DOUBLE *Vr, \
				DOUBLE *work, INT lwork);

void nrkl_Q_func_mod(DOUBLE *QObj, DOUBLE *K, DOUBLE *LfGD, DOUBLE LfObj, \
			DOUBLE *LfGrad, DOUBLE delta, INT numPoints, DOUBLE *KLfGDDiff);

void frkl_pgd(DOUBLE *K, DIST_LABEL_TYPE problemType, INT *constraintMat, \
			DOUBLE *betaVec, DOUBLE *weights, DOUBLE kappa, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT stepFlag, INT numPoints, \
			INT numConstraints);

void frkl_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, \
			DIST_LABEL_TYPE problemType, INT *constraintMat, DOUBLE *betaVec, \
			DOUBLE *weights, DOUBLE kappa, INT numPoints, INT numConstraints);

void kl_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, \
			DIST_LABEL_TYPE problemType, INT *constraintMat, DOUBLE *betaVec, \
			DOUBLE *weights, INT numPoints, INT numConstraints);

void kl_target_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, \
			INT *constraintMat, DOUBLE *targetVec, DOUBLE *weights, \
			INT numPoints, 	INT numConstraints);

void kl_relational_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, \
			INT *constraintMat, DOUBLE *marginVec, DOUBLE *weights, \
			INT numPoints, INT numConstraints);

void kl_sqrhinge_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, \
		INT *constraintMat, DOUBLE *marginVec, DOUBLE *weights, INT numPoints, \
		INT numConstraints);

void kl_bound_obj_grad(DOUBLE *obj, DOUBLE *grad, DOUBLE *K, INT *constraintMat, \
			DOUBLE *boundVec, DOUBLE *weights, INT numPoints, \
			INT numConstraints);

//void semisupervised_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *W, \
//		DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, DOUBLE *classLabels, DOUBLE *D, DOUBLE *DDt, \
//		DOUBLE *mu1p, DOUBLE *mu2p, INT N, INT K, INT numSamples, INT numTasks, INT derivFlag, INT regularizationFlag, \
//		DOUBLE *MD, DOUBLE *ObjMat, DOUBLE *MDDt, DOUBLE *derivTemp);


#endif /* __KERNEL_LEARNING_H__ */
