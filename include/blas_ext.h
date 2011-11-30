/*
 * blasext.h
 *
 *  Created on: Nov 13, 2011
 *      Author: igkiou
 */

#ifndef __BLAS_EXT_H__
#define __BLAS_EXT_H__

#include "useblas.h"

void dimm(CHAR *side, CHAR *trans, INT *m, INT *n, DOUBLE *alpha, DOUBLE *a, DOUBLE *b, DOUBLE *beta, DOUBLE *c);
void diag(CHAR *trans, INT *m, INT *n, DOUBLE *a, DOUBLE *b);
DOUBLE trac(INT *m, INT *n, DOUBLE *a);
DOUBLE vsum(INT *n, DOUBLE *x, INT *incx);
DOUBLE vprd(INT *n, DOUBLE *x, INT *incx);
void geva(CHAR *trans, INT *n, INT *m, DOUBLE *alpha, DOUBLE *a, DOUBLE *b);
void gevm(CHAR *trans, INT *m, INT *n, DOUBLE *alpha, DOUBLE *a, DOUBLE *b);
void gesu(CHAR *trans, INT *n, INT *m, DOUBLE *alpha, DOUBLE *a, DOUBLE *beta, DOUBLE *b);
#ifdef USE_DOUBLE_PRECISION
#define ddimm dimm
#define ddiag diag
#define dtrac trac
#define dvsum vsum
#define dvprd vprd
#define dgeva geva
#define dgevm gevm
#define dgesu gesu
#elif defined(USE_SINGLE_PRECISION)
#define sdimm dimm
#define sdiag diag
#define strac trac
#define svsum vsum
#define svprd vprd
#define sgeva geva
#define sgevm gevm
#define sgesu gesu
#endif

#endif /* __BLAS_EXT_H__ */
