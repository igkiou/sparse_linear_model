/*
 * matrix_dictionary.h
 *
 *  Created on: May 3, 2011
 *      Author: igkiou
 */

#ifndef __MATRIX_DICTIONARY_H__
#define __MATRIX_DICTIONARY_H__

#include "useblas.h"
#ifdef USE_CUDA
#include "usecuda.h"
#endif

#ifdef USE_CUDA
void matrix_dictionary_learning_lowrank_apg_cuda(CUHANDLE handle, \
				CUDOUBLE *h_B, CUDOUBLE *h_XAt, CUDOUBLE *h_AAt, CUDOUBLE mu, \
				CUDOUBLE kappa, CUDOUBLE tolerance, CUDOUBLE delta,	\
				CUINT numIters, CUDOUBLE eta, CUINT initFlag, CUINT M, CUINT N, \
				CUINT K, CUINT numSamples);
#endif

void matrix_dictionary_learning_lowrank_apg_parallel(DOUBLE *B, DOUBLE *XAt, \
				DOUBLE *AAt, DOUBLE mu, DOUBLE kappa, DOUBLE tolerance, \
				DOUBLE delta,INT numIters, DOUBLE eta, INT initFlag, INT M, \
				INT N, INT K, INT numSamples);

void matrix_dictionary_learning_lowrank_apg(DOUBLE *B, DOUBLE *XAt, DOUBLE *AAt, \
				DOUBLE mu, DOUBLE kappa, DOUBLE tolerance, DOUBLE delta,\
				INT numIters, DOUBLE eta, INT initFlag, INT M, INT N, INT K, \
				INT numSamples);

void operator_dictionary_learning_lowrank_weighted_apg_parallel(DOUBLE *B, \
				DOUBLE *XWsqYAt, DOUBLE *AAt, DOUBLE *YtWsqY, DOUBLE mu, \
				DOUBLE kappa, DOUBLE tolerance, DOUBLE delta, INT numIters, \
				DOUBLE eta, INT initFlag, INT M, INT N, INT F, INT K, \
				INT numSamples);

void operator_dictionary_learning_lowrank_weighted_apg(DOUBLE *B, DOUBLE *XWsqYAt, \
				DOUBLE *AAt, DOUBLE *YtWsqY, DOUBLE mu, DOUBLE kappa, \
				DOUBLE tolerance, DOUBLE delta, INT numIters, DOUBLE eta, \
				INT initFlag, INT M, INT N, INT F, INT K, INT numSamples);

void matrix_dictionary_hard_thresholding_parallel(DOUBLE *Dc, DOUBLE *norm, INT rank, INT M, INT N, INT K);

void matrix_dictionary_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *D, DOUBLE *X, DOUBLE *A, DOUBLE mu, \
		INT M, INT N, INT K, INT numSamples, INT derivFlag, DOUBLE *res);

void matrix_dictionary_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *D, DOUBLE *X, DOUBLE *A, DOUBLE *Ksq, \
		DOUBLE mu, INT M, INT N, INT K, INT numSamples, INT derivFlag, DOUBLE *res, DOUBLE *Dt, DOUBLE *derivTemp);

#endif /* __MATRIX_DICTIONARY_H__ */
