/*
 * l2ls_learn_basis.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __L2LS_LEARN_BASIS_H__
#define __L2LS_LEARN_BASIS_H__

#include "useblas.h"

void dual_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *dualLambda, DOUBLE *SSt, DOUBLE *SXt, DOUBLE *SXtXSt, DOUBLE trXXt, \
					DOUBLE c, INT N, INT K, INT derivFlag, DOUBLE *SStLambda, DOUBLE *tempMatrix);

void minimize_dual(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *SSt, DOUBLE *SXt, DOUBLE *SXtXSt, DOUBLE trXXt, \
					DOUBLE c, INT N, INT K);

void l2ls_learn_basis_dual(DOUBLE *Dopt, DOUBLE *Dorig, DOUBLE *X, DOUBLE *S, DOUBLE l2norm, INT length, INT N, INT K, INT numSamples);


#endif /* __L2LS_LEARN_BASIS_H__ */
