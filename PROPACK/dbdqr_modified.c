/* Stephen Becker, 11/12/08
 * Re-writing the fortran dbdqr.f
 * so that I can compile it on computers without
 * a fortran compiler
 *
 * This function calls dlartg, which is a LAPACK routine
 * that compuets plane rotations.  From the documentation,
 * "this is a slower, more accurate version of the BLAS1
 * routine DROTG" (with some syntax differenes).   */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
/*
#include <algorithm>
*/

#include <mkl.h>
#include <omp.h>
#include "mex.h"
#include "matrix.h"
#include "../useblas.h"
#include "../sparse_classification.h"

void dbdqr(INT *n, DOUBLE *d, DOUBLE *e, DOUBLE *c1, DOUBLE *c2) {
	INT i;
	DOUBLE cs, sn, r;

	if (*n < 2) {
		return;
	}
	/* in the fortran version, the loop over i is
	 * from 1 to n-1.  Fortran is 1-based, so I want
	 * to loop from 0 to n-2 in C */
	for (i = 0 ; i < *n-1 ; i++) {
		LARTG(d + i, e + i, &cs, &sn, &r);
		d[i] = r;
		e[i] = sn * d[i + 1];
		d[i + 1] = cs * d[i + 1]; /* need i < n-1 for this */
	}

	/* take care of i = n-1 case */
	i = *n - 1;
	LARTG(d + i, e + i, &cs, &sn, &r);
	d[i] = r;
	e[i] = 0.0;
	*c1 = sn;
	*c2 = cs;
}
                
