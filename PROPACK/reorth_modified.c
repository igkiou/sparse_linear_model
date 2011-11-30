/* Stephen Becker, 3/7/09
 * Re-writing the fortran reorth.c
 * so that I can compile it on computers without
 * a fortran compiler
 *
 * This calls some BLAS functions
 * There is a different naming convention on windows,
 * so when compiling this on windows, define the symbol "WINDOWS"
 * (In the mex compiler, this can be done with the -DWINDOWS flag)
 *
 * See reorth.f for details on calling this
 * reorth.f written by R.M. Larsen for PROPACK
 *
 * March 2009
 * Not working!!!
 *
 * Note: the fortran verion doesn't work for classical gram-schmidt,
 * only for mgs!  So not surprising that this doesn't work
 *
 * Nov 9, 2009
 * dnrm2 wasn't working at all
 * Problem: use INT instead of INT
 * See blas.h in /opt/matlab/extern/include
 * Should be working great now
 *
 * Dec 4, 2009
 * not all versions of matlab have blas.h (or even libmwblas)
 *
 * */

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

/* Modified Gram Schmidt re-orthogonalization */
void MGS(INT *n, INT *k, DOUBLE *V, INT *ldv, DOUBLE *vnew, DOUBLE *index) {

	INT i, j, idx;
	INT LDV = *ldv;
	DOUBLE s;
	for (i = 0; i < *k; ++i) {
		idx = (INT) index[i] - 1;  /* -1 because MATLAB uses 1-based indices */
		s = 0.0;

		for (j=0; j<*n; ++j) {
			/*s += V[j,idx]*vnew[j]; */ 
			s += V[idx *LDV + j] * vnew[j];
		}

		for (j = 0; j < *n; ++j) {
			/* vnew[j] -= s*V[j,idx]; */ /* Fortran is row-indexed */
			vnew[j] -= s*V[idx * LDV + j];
		}
	}
}

void reorth(INT *n, INT *k, DOUBLE *V, INT *ldv, DOUBLE *vnew, DOUBLE *normv, DOUBLE *index,
		DOUBLE *alpha, DOUBLE *work, INT *iflag, INT *nre) {

	INT i;
	INT one = 1; 
	INT N = (INT) *n;
	INT K = (INT) *k;
	INT LDV = (INT) *ldv;
	CHAR Transpose = 'T';
	CHAR Normal = 'N';
	DOUBLE normv_old;
	INT MAXTRY = 4;

	DOUBLE oneD = 1.0;
	DOUBLE nOneD = -1.0; 
	DOUBLE zero = 0.0;

	INT workflag = 0;
	if (work == NULL) {
		work = mxMalloc(* k * sizeof(DOUBLE));
		workflag = 1;
	}
	
/*
	void (*GEMVPtr)(CHAR *, INT *, INT *, DOUBLE *, DOUBLE *, INT *, 
		  DOUBLE *, INT *, DOUBLE *, DOUBLE *, INT *) = GEMV; 
	DOUBLE (*NRM2Ptr)(INT*, DOUBLE *, INT *) = NRM2; 
*/

	/* Hack: if index != 1:k, we do MGS to avoid reshuffling */
	if (*iflag == 1) {
		for (i = 0; i< *k; ++i){
			if (index[i] != (i + 1)) {
				*iflag = 0;
				break;
			}
		}
	}
	normv_old = 0;
	*nre = 0;  
	*normv = 0.0;

	while ((*normv < *alpha * normv_old) || (*nre == 0)) {
		if (*iflag == 1) {
			/* CGS */
			GEMV(&Transpose, &N, &K, &oneD, V, &LDV, vnew, &one, &zero, work, &one); 
			GEMV(&Normal, &N, &K, &nOneD, V, &LDV, work, &one, &oneD, vnew, &one); 
		} else {
			/* MGS */
			MGS(n, k, V, ldv, vnew, index);
		}
		normv_old = *normv; 
		/* following line works! */
		*normv = NRM2(&N, vnew, &one);    

		/* following line does not work: */
/*
		*normv = dnrm2Ptr( (INT *)n, vnew, &one );
*/

		*nre = *nre + 1;

		if (*nre > MAXTRY) {
			/* vnew is numerically in span(V) --> return vnew as all zeros */
			*normv = 0.0;
			for (i = 0; i< *n ; ++i)
				vnew[i] = 0.0;
			return;
		}
	}
	
	if (workflag == 1) {
		mxFree(work);
	}
}


