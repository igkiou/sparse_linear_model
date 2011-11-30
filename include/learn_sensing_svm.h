/*
 * learn_sensing_svm.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __LEARN_SENSING_SVM_H__
#define __LEARN_SENSING_SVM_H__

#include "useblas.h"

void huber_obj_grad_multitask(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, \
						DOUBLE *weights, DOUBLE *bias, INT M, INT N, INT numSamples, INT numTasks, INT derivFlag);

void huber_obj_grad_for(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, \
						DOUBLE *bias, INT M, INT N, INT numSamples, INT derivFlag, DOUBLE *Ypred, DOUBLE *Ytemp);

void huber_average_obj_grad_for(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, INT numPool, INT derivFlag, DOUBLE *Ypred, DOUBLE *Ytemp);

void huber_average_obj_grad_multitask(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, INT numPool, INT numTasks, INT derivFlag);

void huber_obj_grad_multitask_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, \
						DOUBLE *weights, DOUBLE *bias, INT M, INT N, INT numSamples, INT numTasks, \
						DOUBLE *Ypred, DOUBLE *Ytemp, DOUBLE *derivtemp);

void huber_obj_grad_tensor_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, \
						DOUBLE *weights, DOUBLE *bias, DOUBLE *wXtensor, INT M, INT N, INT numSamples, \
						INT initFlag, DOUBLE *Ypred, DOUBLE *Ytemp);

void huber_obj_grad_for_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, DOUBLE *Ypred, DOUBLE *Ytemp);

void minimize_huber(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *Xtrain, DOUBLE *Y, DOUBLE *weights, \
						DOUBLE *bias, INT M, INT N, INT numSamples, INT numTasks);

void square_obj_grad_multitask(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, \
						DOUBLE *weights, DOUBLE *bias, INT M, INT N, INT numSamples, INT numTasks, INT derivFlag);

void square_obj_grad_for(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, \
						DOUBLE *bias, INT M, INT N, INT numSamples, INT derivFlag, DOUBLE *Ypred, DOUBLE *Ytemp);

void square_obj_grad_multitask_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, \
						DOUBLE *weights, DOUBLE *bias, INT M, INT N, INT numSamples, INT numTasks, \
						DOUBLE *Ypred, DOUBLE *Ytemp, DOUBLE *derivtemp);

void square_obj_grad_tensor_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, \
						DOUBLE *weights, DOUBLE *bias, DOUBLE *wXtensor, INT M, INT N, INT numSamples, \
						INT initFlag, DOUBLE *Ypred, DOUBLE *Ytemp);

void square_obj_grad_for_serial(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *weights, DOUBLE *bias, \
					INT M, INT N, INT numSamples, DOUBLE *Ypred, DOUBLE *Ytemp);

void reconstruction_obj_grad_sparse(DOUBLE *obj, DOUBLE *deriv, DOUBLE *D, DOUBLE *X, DOUBLE *Aval, INT *Aindx, INT *Apntrb, \
					INT *Apntre, INT N, INT K, INT numSamples, INT derivFlag, DOUBLE *Dt, DOUBLE *Xt, DOUBLE *Errt, \
					DOUBLE *derivt);

void reconstruction_obj_grad_dense(DOUBLE *obj, DOUBLE *deriv, DOUBLE *D, DOUBLE *X, DOUBLE *A, \
					INT N, INT K, INT numSamples, INT derivFlag, DOUBLE *Err);


#endif /* __LEARN_SENSING_SVM_H__ */
