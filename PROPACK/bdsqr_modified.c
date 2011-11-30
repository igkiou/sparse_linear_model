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

void bdsqr_caller(INT M, DOUBLE *d, DOUBLE *bnd, DOUBLE *e, DOUBLE *wrk) {

/*
	INT eflag = 0;
	if (e == NULL) {
		e = mxMalloc(m * sizeof(DOUBLE)); 
		eflag = 1;
	}
*/
	
	INT wrkflag = 0;
	if (wrk == NULL) {
		wrk = mxMalloc(4 * M * sizeof(DOUBLE)); 
		wrkflag = 1;
	}
	
	if (wrk == NULL) {
		mexErrMsgTxt("Failed to allocate memory");
	}
	
	memset((void *) bnd, 0, M * sizeof(DOUBLE));
	
	INT Zero = 0; 
	INT One = 1; 
	INT Info;
  	DOUBLE dummy = 1; 

	/* Reduce to upper m-by-m upper bidiagonal */
	BDQR(&M, d, e, &bnd[M - 1], &dummy);	
  
	/* Compute singular values and last row of U */
	BDSQR("U", &M, &Zero, &One, &Zero, d, e, &dummy, &One, \
       bnd, &One, &dummy, &One, wrk, &Info);  
	
	/* Check exit status of dbdsqr */
	if (Info < 0) {
		mexErrMsgTxt("DBDSQR was called with illegal arguments");
	} else if (Info > 0) {
		mexWarnMsgTxt("DBDSQR: singular values did not converge");
	}
	
/*
	if (eflag == 1) {
		mxFree(e);
	}
*/
	
	if (wrkflag == 1) {
		mxFree(wrk);
	}
}
