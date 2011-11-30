/*
 * l1_featuresign.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __L1_FEATURESIGN_H__
#define __L1_FEATURESIGN_H__

#include "useblas.h"
#include "exponential.h"

//void l1qp_featuresign_sub_serial(DOUBLE *x, DOUBLE *A, DOUBLE *b, DOUBLE lambda, INT K, \
//						DOUBLE *grad, INT *actset, DOUBLE *xa, DOUBLE *ba, \
//						DOUBLE *Aa, DOUBLE *signa, DOUBLE *vect, DOUBLE *xnew, INT *idxset, \
//						INT *sset, DOUBLE *vecti, DOUBLE *bi, DOUBLE *xnewi, DOUBLE *Ai, \
//						DOUBLE *xmin, DOUBLE *d, DOUBLE *t, DOUBLE *xs);

//void l1qp_featuresign_serial(DOUBLE *S, DOUBLE *X, DOUBLE *D, DOUBLE *lambdap, INT N, INT K, \
//						INT numSamples, DOUBLE *regparamp);

void l1_featuresign_sub(DOUBLE *x, DOUBLE *A, DOUBLE *b, DOUBLE lambda, INT K, \
						DOUBLE *grad, INT *actset, DOUBLE *xa, DOUBLE *ba, \
						DOUBLE *Aa, DOUBLE *signa, DOUBLE *vect, DOUBLE *xnew, INT *idxset, \
						INT *sset, DOUBLE *vecti, DOUBLE *bi, DOUBLE *xnewi, DOUBLE *Ai, \
						DOUBLE *xmin, DOUBLE *d, DOUBLE *t, DOUBLE *xs, INT init);

void l1qp_featuresign(DOUBLE *S, DOUBLE *X, DOUBLE *D, DOUBLE *lambdap, INT N, INT K, \
						INT numSamples, DOUBLE *regparamp, DOUBLE *KDD, DOUBLE *KDX);

void l1kernel_featuresign(DOUBLE *S, DOUBLE *KDX, DOUBLE *KDD, DOUBLE *lambdap, \
						INT K, INT numSamples, DOUBLE *regparamp);

void l1exp_featuresign_sub(DOUBLE *s, DOUBLE *x, DOUBLE *Dt, DOUBLE lambda, INT N, INT K, EXPONENTIAL_TYPE family, DOUBLE regparam, \
						DOUBLE *grad, INT *actset, DOUBLE *xa, DOUBLE *ba, \
						DOUBLE *Aa, DOUBLE *signa, DOUBLE *vect, DOUBLE *xnew, INT *idxset, \
						INT *sset, DOUBLE *vecti, DOUBLE *bi, DOUBLE *xnewi, DOUBLE *Ai, \
						DOUBLE *xmin, DOUBLE *d, DOUBLE *t, DOUBLE *xs, DOUBLE *Ds, DOUBLE *xtilde, \
						DOUBLE *Dttilde, DOUBLE *A, DOUBLE *b, DOUBLE *shat, \
						DOUBLE *aPrime, DOUBLE *aDoublePrime, DOUBLE *deriv);

void l1exp_obj_subgrad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *s, DOUBLE *Dt, DOUBLE *x, INT N, INT K, \
						DOUBLE lambda, EXPONENTIAL_TYPE family, INT derivFlag, DOUBLE *Ds, DOUBLE *aPrime);

void l1exp_featuresign(DOUBLE *S, DOUBLE *X, DOUBLE *D, DOUBLE *lambdap, INT N, INT K, \
				INT numSamples, EXPONENTIAL_TYPE family, DOUBLE *regparamp);

void exp_irls_sub(DOUBLE *s, DOUBLE *x, DOUBLE *Dt, INT N, INT K, EXPONENTIAL_TYPE family, DOUBLE regparam, \
						DOUBLE *Ds, DOUBLE *xtilde, DOUBLE *Dttilde, DOUBLE *A, DOUBLE *shat, \
						DOUBLE *aPrime, DOUBLE *aDoublePrime, DOUBLE *deriv);

void exp_irls(DOUBLE *S, DOUBLE *X, DOUBLE *D, INT N, INT K, \
				INT numSamples, EXPONENTIAL_TYPE family, DOUBLE *regparamp);

#endif /* __L1_FEATURESIGN_H__ */
