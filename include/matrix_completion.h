/*
 * matrix_completion.h
 *
 *  Created on: Nov 20, 2011
 *      Author: igkiou
 */

#ifndef __MATRIX_COMPLETION_H__
#define __MATRIX_COMPLETION_H__

void convertObservationMat(INT *observedInds, DOUBLE *observedVals, \
						DOUBLE *matlabObservationMat, INT numObservations, \
						INT M, INT N);

void matrix_completion_apg(DOUBLE *B, INT *observedInds, DOUBLE *observedVals,
						DOUBLE mu, DOUBLE kappa, INT numIters, DOUBLE tolerance, \
						DOUBLE delta, DOUBLE eta, INT initFlag, INT M, INT N, \
						INT numObserved);

void operator_completion_apg(DOUBLE *B, INT *observedInds, DOUBLE *observedVals,
						DOUBLE *Y, DOUBLE mu, DOUBLE kappa, INT numIters, \
						DOUBLE tolerance, DOUBLE delta, DOUBLE eta, \
						INT initFlag, INT M, INT N, INT K, INT numObserved);

#endif /* __MATRIX_COMPLETION_H__ */
