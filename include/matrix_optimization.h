/*
 * matrix_optimization.h
 *
 *  Created on: Mar 25, 2011
 *      Author: igkiou
 */

#ifndef __MATRIX_OPTIMIZATION_H__
#define __MATRIX_OPTIMIZATION_H__

#include "useblas.h"

void abs_smooth_obj_grad(DOUBLE *objVec, DOUBLE *derivVec, DOUBLE *xVec, DOUBLE *rp, \
					INT N, INT derivFlag);

void nuclear_approx_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, DOUBLE *rp, \
					INT M, INT N, INT derivFlag, DOUBLE *svdVec, DOUBLE *vtMat, \
					DOUBLE *dataBuffer, DOUBLE *derivVec, DOUBLE *work, INT lwork);

void nuclear_proximal(DOUBLE *X, DOUBLE *norm, DOUBLE tau, INT M, INT N, DOUBLE *sv, \
		DOUBLE *svecsmall, DOUBLE *sveclarge, DOUBLE *work, INT lwork);

void nuclear_hard_thresholding(DOUBLE *X, DOUBLE *norm, INT rank, INT M, INT N, DOUBLE *sv, \
		DOUBLE *svecsmall, DOUBLE *sveclarge, DOUBLE *work, INT lwork);

void nuclear_psd_proximal(DOUBLE *X, DOUBLE *norm, DOUBLE tau, INT M, DOUBLE *eigv, \
		DOUBLE *eigvec, DOUBLE *work, INT lwork);

void nuclear_psd_hard_thresholding(DOUBLE *X, DOUBLE *norm, INT rank, INT M, DOUBLE *eigv, \
		DOUBLE *eigvec, DOUBLE *work, INT lwork);

//void nuclear_psd_hard_thresholding(DOUBLE *X, DOUBLE *norm, INT rank, INT M, DOUBLE *eigv, \
//		DOUBLE *eigvec, DOUBLE *work, INT lwork, INT *iwork, INT liwork);

#endif /* __MATRIX_OPTIMIZATION_H__ */
