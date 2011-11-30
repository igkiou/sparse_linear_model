/*
 * pca.h
 *
 *  Created on: Oct 12, 2011
 *      Author: igkiou
 */

#ifndef __PCA_H__
#define __PCA_H__

#include "useblas.h"

#ifdef USE_CUDA
void robust_pca_apg_cuda(CUHANDLE handle, CUDOUBLE *h_B, CUDOUBLE *h_A, \
				CUDOUBLE *h_D, CUDOUBLE mu, CUDOUBLE lambda, CUDOUBLE kappa, \
				CUINT numIters, CUDOUBLE tolerance, CUDOUBLE delta, \
				CUDOUBLE eta, CUINT initFlag, CUINT M, CUINT N);
#endif

void robust_pca_apg(DOUBLE *B, DOUBLE *A, DOUBLE *D, DOUBLE mu, DOUBLE lambda, \
				DOUBLE kappa, INT numIters, DOUBLE tolerance, DOUBLE delta, \
				DOUBLE eta, INT initFlag, INT M, INT N);

void robust_pca_apg_gesdd(DOUBLE *B, DOUBLE *A, DOUBLE *D, DOUBLE mu, DOUBLE lambda, \
				DOUBLE kappa, INT numIters, DOUBLE tolerance, DOUBLE delta, \
				DOUBLE eta, INT initFlag, INT M, INT N);

#ifdef USE_CUDA
void robust_weighted_operator_pca_apg_cuda(CUHANDLE handle, CUDOUBLE *h_B, \
				CUDOUBLE *h_A, CUDOUBLE *h_D, CUDOUBLE *h_Y, CUDOUBLE *h_W, \
				CUDOUBLE mu, CUDOUBLE lambda, CUDOUBLE kappa, CUINT numIters, \
				CUDOUBLE tolerance, CUDOUBLE delta, CUDOUBLE eta, \
				CUINT initFlag, CUINT M, CUINT N, CUINT K);
#endif

void robust_weighted_operator_pca_apg(DOUBLE *B, DOUBLE *A, DOUBLE *D, \
				DOUBLE *Y, DOUBLE *W, DOUBLE mu, DOUBLE lambda, \
				DOUBLE kappa, INT numIters, DOUBLE tolerance, DOUBLE delta, \
				DOUBLE eta, INT initFlag, INT M, INT N, INT K);

void robust_weighted_operator_pca_apg_gesdd(DOUBLE *B, DOUBLE *A, DOUBLE *D, \
				DOUBLE *Y, DOUBLE *W, DOUBLE mu, DOUBLE lambda, \
				DOUBLE kappa, INT numIters, DOUBLE tolerance, DOUBLE delta, \
				DOUBLE eta, INT initFlag, INT M, INT N, INT K);

void robust_operator_pca_apg(DOUBLE *B, DOUBLE *A, DOUBLE *D, DOUBLE *Y, \
				DOUBLE mu, DOUBLE lambda, DOUBLE kappa, INT numIters, \
				DOUBLE tolerance, DOUBLE delta, DOUBLE eta, INT initFlag, \
				INT M, INT N, INT K);

#endif /* __PCA_H__ */
