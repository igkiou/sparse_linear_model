/*
 * learn_mahalanobis.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __LEARN_MAHALANOBIS_H__
#define __LEARN_MAHALANOBIS_H__

#include "useblas.h"

void mahalanobis_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, DOUBLE *DDt2, DOUBLE *DDt3, DOUBLE *VL, \
						DOUBLE *L, INT N, INT derivFlag, DOUBLE *GtG, DOUBLE *ObjMat, DOUBLE *MDDt2);

void minimize_mahalanobis(DOUBLE *Xopt, DOUBLE *Xorig, INT length, DOUBLE *DDt2, DOUBLE *DDt3, DOUBLE *VL, \
						DOUBLE *L, INT N);

void mahalanobis_unweighted_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, DOUBLE *D, DOUBLE *DDt, \
					INT N, INT K, INT derivFlag, DOUBLE *MD, DOUBLE *ObjMat, DOUBLE *MDDt);

void mahalanobis_ml_obj_grad(DOUBLE *obj, DOUBLE *deriv, DOUBLE *X, DOUBLE *XXt, DOUBLE *XAtAXt, \
					DOUBLE *normAAtSq, INT N, INT derivFlag, DOUBLE *MXXt, DOUBLE *MXXtSq, DOUBLE *MXAtAXt);

#endif /* __LEARN_MAHALANOBIS_H__ */
