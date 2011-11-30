/*
 * learn_sensing.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __LEARN_SENSING_H__
#define __LEARN_SENSING_H__

#include "useblas.h"

void random_sensing(DOUBLE *Phi, INT M, INT N);

void learn_sensing(DOUBLE *Phi, DOUBLE *D, INT M, INT N, INT K);

void minimize(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *DDt2, DOUBLE *DDt3, DOUBLE *VL, DOUBLE *L, INT M, INT N);

void eig_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, DOUBLE *DDt2, DOUBLE *DDt3, DOUBLE *VL, \
						DOUBLE *L, INT M, INT N, INT derivFlag, DOUBLE *Gamma, DOUBLE *ObjMat, DOUBLE *PhiDDt2, \
						DOUBLE* DDt3temp);

void eig_lap_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *XLXt, DOUBLE *DDt2, DOUBLE *DDt3,\
					DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N, INT numSamples, INT derivFlag,\
					DOUBLE *PhiXLXt, DOUBLE *PhiXLXtPhit, DOUBLE *Gamma, DOUBLE *ObjMat, DOUBLE *PhiDDt2, DOUBLE *DDt3temp);

void eig_lsqr_obj_grad_smalldata(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *X, DOUBLE *Y, DOUBLE *DDt2, DOUBLE *DDt3,\
					DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N, INT numSamples, INT derivFlag,\
					DOUBLE *Err, DOUBLE *Gamma, DOUBLE *ObjMat, DOUBLE *PhiDDt2, DOUBLE* DDt3temp);

void eig_lsqr_obj_grad_largedata(DOUBLE *obj, DOUBLE *deriv, DOUBLE *Phi, DOUBLE *XXt, DOUBLE *YXt, DOUBLE trYYt, DOUBLE *DDt2,\
					DOUBLE *DDt3, DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N, INT derivFlag,\
					DOUBLE *AXXt, DOUBLE *AXXtAt, DOUBLE *Gamma, DOUBLE *ObjMat, DOUBLE *PhiDDt2, DOUBLE* DDt3temp);

void minimize_eig_lap(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *XLXt, DOUBLE *DDt2, DOUBLE *DDt3, \
					DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N, INT numSamples);

void minimize_eig_lsqr_largedata(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *XXt, DOUBLE *YXt, DOUBLE trYYt,\
					DOUBLE *DDt2, DOUBLE *DDt3, DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N);

void minimize_eig_lsqr_smalldata(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *Xdata, DOUBLE *Y, DOUBLE *DDt2,\
					DOUBLE *DDt3, DOUBLE *VL, DOUBLE *L, DOUBLE alphaReg, DOUBLE betaReg, INT M, INT N, INT numSamples);

void orth_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, INT M, INT N, INT derivFlag, DOUBLE *PhiPhit);

void minimize_orth(DOUBLE *Xopt, DOUBLE *Xorig, INT length, INT M, INT N);

#endif /* __LEARN_SENSING_H__ */
