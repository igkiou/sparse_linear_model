/*
 * ksvd.h
 *
 *  Created on: Mar 11, 2011
 *      Author: igkiou
 */

#ifndef __KSVD_H__
#define __KSVD_H__

#include "useblas.h"

void optimize_atom(DOUBLE *D, DOUBLE* X, DOUBLE *Gamma, INT j, INT *unusedSigs, INT *replacedAtoms, INT N, INT K, INT numSamples, \
		INT *numUnusedp, INT *dataIndices, DOUBLE *gammaj, DOUBLE *Dj, DOUBLE *u, DOUBLE *s, DOUBLE *smallGamma, DOUBLE *smallX);

void clear_dictionary(DOUBLE *D, DOUBLE *X, DOUBLE *Gamma, DOUBLE muThresh, INT *unusedSigs, INT *replacedAtoms, \
		INT N, INT K, INT numSamples, INT numUnused, DOUBLE *smallGamma, DOUBLE *smallX, DOUBLE *errorVector, \
		INT *sortedUnusedSigs, INT *useCount, DOUBLE *Gj);

void optimize_atom_general(DOUBLE *D, DOUBLE *Dorig, DOUBLE* X, DOUBLE *Gamma, DOUBLE *extPhi, DOUBLE *cholPhiLambda, INT j, \
		INT *unusedSigs, INT *replacedAtoms, INT Neq, INT K, INT numSamples, INT numMeasurements, INT *numUnusedp, \
		INT *dataIndices, DOUBLE *gammaj, DOUBLE *Dj, DOUBLE *u, DOUBLE *s, DOUBLE *smallGamma, DOUBLE *smallX, \
		DOUBLE *atom);

void ksvd_update(DOUBLE *Deq, DOUBLE *D, DOUBLE *Xeq, DOUBLE *X, DOUBLE *Gamma, DOUBLE *extPhi, DOUBLE *cholPhiLambda, INT Norig, \
		INT K, INT numSamples, INT numMeasurements);

#endif /* __KSVD_H__ */
