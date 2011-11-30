/*
 * l2kernel_learn_basis.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __L2KERNEL_LEARN_BASIS_H__
#define __L2KERNEL_LEARN_BASIS_H__

#include "useblas.h"
#include "kernel_gram.h"

void basis_kernel_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *D, DOUBLE *X, DOUBLE *A, INT N, INT K, INT numSamples, \
				KERNEL_TYPE kernelType, DOUBLE *pparam1, DOUBLE *pparam2, INT derivFlag, DOUBLE *KDD, DOUBLE *KDX, DOUBLE *KDDDX, \
				DOUBLE *ak, DOUBLE *normMat1, DOUBLE *oneVec, DOUBLE *tempMat1, DOUBLE *tempMat2);

void minimize_kernel_basis(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *Xdata, DOUBLE *Acodes, INT N, INT K, INT numSamples, \
				KERNEL_TYPE kernelType, DOUBLE *pparam1, DOUBLE *pparam2);

#endif /* __L2KERNEL_LEARN_BASIS_H__ */
