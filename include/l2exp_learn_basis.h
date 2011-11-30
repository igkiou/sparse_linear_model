/*
 * l2exp_learn_basis.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __L2EXP_LEARN_BASIS_H__
#define __L2EXP_LEARN_BASIS_H__

#include "useblas.h"
#include "exponential.h"

void basis_exp_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *D, DOUBLE *X, DOUBLE *S, INT N, \
					INT K, INT numSamples, EXPONENTIAL_TYPE family, INT derivFlag, DOUBLE *DS, DOUBLE *aVal, \
					DOUBLE *aPrime);

void l2exp_learn_basis_gradient_projection_backtracking(DOUBLE *D, DOUBLE *X, DOUBLE *S, DOUBLE *Dinit, \
							INT N, INT K, INT numSamples, EXPONENTIAL_TYPE family);

void l2exp_learn_basis_gradient_backtracking(DOUBLE *D, DOUBLE *X, DOUBLE *S, DOUBLE *Dinit, \
							INT N, INT K, INT numSamples, EXPONENTIAL_TYPE family);


#endif /* __L2EXP_LEARN_BASIS_H__ */
