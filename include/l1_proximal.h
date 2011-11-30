/*
 * l1_proximal.h
 *
 *  Created on: Apr 19, 2011
 *      Author: igkiou
 */

#ifndef __L1_PROXIMAL_H__
#define __L1_PROXIMAL_H__

#include "useblas.h"
#ifdef USE_CUDA
#include "usecuda.h"
#endif

void l1qp_ista(DOUBLE *A, DOUBLE *X, DOUBLE *D, DOUBLE *lambda, DOUBLE delta, \
			INT numIters, DOUBLE tolerance, INT N, INT K, \
			INT numSamples, INT numRepeats);

void l1kernel_ista(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE *lambda, \
			DOUBLE delta, INT numIters, DOUBLE tolerance, INT K, \
			INT numSamples, INT numRepeats);

void l1_ista_inner(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE tau, \
			DOUBLE delta, INT numIters, DOUBLE tolerance, INT K, DOUBLE *AOld, \
			DOUBLE *KDDA);

void l1qp_fista(DOUBLE *A, DOUBLE *X, DOUBLE *D, DOUBLE *lambda, DOUBLE delta0, \
			INT numIters, DOUBLE tolerance, INT lineSearchFlag, DOUBLE eta, \
			INT N, INT K, INT numSamples, INT numRepeats);

void l1kernel_fista(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE *lambda, \
			DOUBLE delta0, INT numIters, DOUBLE tolerance, INT lineSearchFlag, \
			DOUBLE eta, INT K, INT numSamples, INT numRepeats);

void l1_fista_inner(DOUBLE *A, DOUBLE *KDX, DOUBLE *KDD, DOUBLE tau, \
			DOUBLE delta0, INT numIters, DOUBLE tolerance, INT lineSearchFlag, \
			DOUBLE eta, INT K, DOUBLE *AOld, DOUBLE *KDDA, DOUBLE *L, \
			DOUBLE *LfGrad, DOUBLE *LfGD, DOUBLE *ALfGDDiff);

void l1_line_search(DOUBLE *delta, DOUBLE *LfGDShrink, DOUBLE *L, \
		DOUBLE LfObj, DOUBLE *LfGrad, DOUBLE tau, INT lineSearchFlag, \
		DOUBLE eta, DOUBLE *KDX, DOUBLE *KDD, INT K, DOUBLE *LfGD, \
		DOUBLE *ALfGDDiff, DOUBLE *KDDA);

void l1_shrinkage(DOUBLE *AShrink, DOUBLE *AShrinkNorm, DOUBLE *A, \
				DOUBLE tau, INT K);

void l1_Q_func_mod(DOUBLE *QObj, DOUBLE *A, DOUBLE *LfGD, DOUBLE LfObj, \
			DOUBLE *LfGrad, DOUBLE delta, INT K, DOUBLE *ALfGDDiff);

void l1_proximal(DOUBLE *X, DOUBLE *norm, DOUBLE tau, INT N);

#ifdef USE_CUDA
void l1_proximal_cuda(CUHANDLE handle, CUDOUBLE *X, CUDOUBLE *h_norm, \
						CUDOUBLE tau, CUINT N);
#endif

void qp_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *A, DOUBLE *X, DOUBLE *D, \
		INT N, INT K, DOUBLE *res);

void kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *A, DOUBLE *KXX, \
		DOUBLE *KDX, DOUBLE *KDD, INT K, DOUBLE *KDDA);

void kernel_alt_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *A, DOUBLE *KDX, \
		DOUBLE *KDD, INT K, DOUBLE *KDDA);

void qp_lipschitz(DOUBLE *Lf, DOUBLE *D, INT N, INT K, DOUBLE *KDD, \
				INT precomputedKernelFlag);

#endif /* __L1_PROXIMAL_H__ */
