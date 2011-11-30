/*
 * svm_optimization.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __SVM_OPTIMIZATION_H__
#define __SVM_OPTIMIZATION_H__

#include "useblas.h"

void squaredhinge_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *wb, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
		DOUBLE *kernelMatrix, INT M, INT numSamples, INT biasFlag, INT derivFlag, INT regularizationFlag, \
		DOUBLE *Ypred, DOUBLE *KX);

void huberhinge_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *wb, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
		DOUBLE *kernelMatrix, INT M, INT numSamples, INT biasFlag, INT derivFlag, INT regularizationFlag, \
		DOUBLE *Ypred, DOUBLE *KX);

void huberhinge_obj_grad_alt(DOUBLE *obj, DOUBLE *deriv, DOUBLE *w, DOUBLE *X, DOUBLE *Y, INT taskLabel, \
		DOUBLE *lambdap, INT M, INT numSamples, INT derivFlag, INT regularizationFlag, DOUBLE *Ypred);

void multihuberhinge_obj_grad_memory(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
				DOUBLE *classLabels, INT M, INT numSamples, INT numTasks, INT derivFlag, INT regularizationFlag);

void multihuberhinge_obj_grad_speed(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
		 	 DOUBLE *classLabels, INT M, INT numSamples, INT numTasks, INT derivFlag, INT regularizationFlag, \
		 	 DOUBLE *Ypred);

void cramersinger_approx_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
					INT N, INT numSamples, INT numTasks, INT derivFlag, DOUBLE *oneVec, DOUBLE *expMat, DOUBLE *logArg);

void cramersinger_nuclear_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *gammap, \
					DOUBLE *rhop, DOUBLE *lambdap, INT N, INT numSamples, INT numTasks, INT derivFlag, \
					DOUBLE *oneVecSvdVec, DOUBLE *expMatVtMat, DOUBLE *logArgDerivVec, DOUBLE *dataBuffer, \
					DOUBLE *work, INT lwork);

void cramersinger_frobenius_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, DOUBLE *gammap, \
					DOUBLE *lambdap, INT N, INT numSamples, INT numTasks, INT derivFlag, \
					DOUBLE *oneVec, DOUBLE *expMat, DOUBLE *logArg);

void multihuberhinge_nuclear_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *W, DOUBLE *X, DOUBLE *Y, \
					DOUBLE *rhop, DOUBLE *lambdap, DOUBLE *classLabels, INT N, INT numSamples, INT numTasks, \
					INT derivFlag, DOUBLE *YpredVtMat, DOUBLE *svdVec, DOUBLE *derivVec, DOUBLE *dataBuffer, \
					DOUBLE *work, INT lwork);

void pegasos_svm_sub(DOUBLE *weights, DOUBLE *bias, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
		INT *taskLabel, INT N, INT numSamples, INT biasFlag, INT numIters, INT batchSize, INT returnAverageFlag, \
		INT *batch, DOUBLE *deriv, DOUBLE *weightsAverage);

void pegasos_binary_svm(DOUBLE *weights, DOUBLE *bias, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, \
		INT N, INT numSamples, INT biasFlag, INT numIters, INT batchSize, INT returnAverageFlag);

void pegasos_svm_vl(DOUBLE *model, DOUBLE *X, DOUBLE *Y, INT N, INT numSamples,
				DOUBLE *lambdap, INT numIters);

void pegasos_multiclass_svm(DOUBLE *weights, DOUBLE *bias, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, DOUBLE *classLabels, \
		INT N, INT numSamples, INT numTasks, INT biasFlag, INT numIters, INT batchSize, INT returnAverageFlag);

void pegasos_multiclass_svm_alt(DOUBLE *weights, DOUBLE *bias, DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, DOUBLE *classLabels, \
		INT N, INT numSamples, INT numTasks, INT biasFlag, INT numIters, INT batchSize, INT returnAverageFlag);

void squaredhinge_kernel_obj_grad_sub(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *wb, \
			DOUBLE *X, DOUBLE *Y, INT taskLabel, DOUBLE *lambdap, INT M, INT numSamples, INT derivFlag, \
			INT regularizationFlag, DOUBLE *Ypred, DOUBLE *KX);

void squaredhinge_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *wb, \
			DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, INT M, INT numSamples, INT derivFlag, \
			INT regularizationFlag, DOUBLE *Ypred, DOUBLE *KX);

void multisquaredhinge_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *W, DOUBLE *X, DOUBLE *Y, \
			DOUBLE *lambdap, DOUBLE *classLabels, INT M, INT numSamples, INT numTasks, INT derivFlag, INT regularizationFlag);

void huberhinge_kernel_obj_grad_sub(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *wb, \
			DOUBLE *X, DOUBLE *Y, INT taskLabel, DOUBLE *lambdap, INT M, INT numSamples, INT derivFlag, \
			INT regularizationFlag, DOUBLE *Ypred, DOUBLE *KX);

void huberhinge_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *wb, \
			DOUBLE *X, DOUBLE *Y, DOUBLE *lambdap, INT M, INT numSamples, INT derivFlag, \
			INT regularizationFlag, DOUBLE *Ypred, DOUBLE *KX);

void multihuberhinge_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *kernelMatrix, DOUBLE *W, DOUBLE *X, DOUBLE *Y, \
			DOUBLE *lambdap, DOUBLE *classLabels, INT M, INT numSamples, INT numTasks, INT derivFlag, INT regularizationFlag);

#endif /* __SVM_OPTIMIZATION_H__ */
