/*
-------------------------------------------------------------------------
   GATEWAY ROUTINE FOR CALLING REORTH FROM MATLAB.

   REORTH   Reorthogonalize a vector using iterated Gram-Schmidt

   [R_NEW,NORMR_NEW,NRE] = reorth(Q,R,NORMR,INDEX,ALPHA,METHOD)
   reorthogonalizes R against the subset of columns of Q given by INDEX. 
   If INDEX==[] then R is reorthogonalized all columns of Q.
   If the result R_NEW has a small norm, i.e. if norm(R_NEW) < ALPHA*NORMR,
   then a second reorthogonalization is performed. If the norm of R_NEW
   is once more decreased by  more than a factor of ALPHA then R is 
   numerically in span(Q(:,INDEX)) and a zero-vector is returned for R_NEW.

   If method==0 then iterated modified Gram-Schmidt is used.
   If method==1 then iterated classical Gram-Schmidt is used.

   The default value for ALPHA is 0.5. 
   NRE is the number of reorthogonalizations performed (1 or 2).

 References: 
  Aake Bjorck, "Numerical Methods for Least Squares Problems",
  SIAM, Philadelphia, 1996, pp. 68-69.

  J.~W. Daniel, W.~B. Gragg, L. Kaufman and G.~W. Stewart, 
  ``Reorthogonalization and Stable Algorithms Updating the
  Gram-Schmidt QR Factorization'', Math. Comp.,  30 (1976), no.
  136, pp. 772-795.

  B. N. Parlett, ``The Symmetric Eigenvalue Problem'', 
  Prentice-Hall, Englewood Cliffs, NJ, 1980. pp. 105-109

  Rasmus Munk Larsen, DAIMI, 1998.
-------------------------------------------------------------------------
  */

/* Modifications by Stephen Becker, srbecker@caltech.edu
 * Update, 3/7/09
 * Re-wrote reorth.f to reorth.c (in C) so that it's easier to compile
 * on Windows (since windows fortran compilers are not that common).
 *
 * When install on Windows, define the pre-processor definition "WINDOWS"
 * Then it will call the appropriate functions
 *
 * 11/9/09
 * Fixed bugs in reorth.c; it is now preferable to use this, as opposed
 * to reorth.f.  Should work with 64-bit systems.
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

/* Here comes the gateway function to be called by Matlab: */
void mexFunction(int nlhs, mxArray *plhs[], 
		 int nrhs, const mxArray *prhs[])
{
	if (nrhs != 6) {
		mexErrMsgTxt("reorth requires 6 input arguments");
	} else if (nlhs < 2) {
		mexErrMsgTxt("reorth requires at least 2 output arguments");
	}

	INT n = mxGetM(prhs[0]); /* get the dimensions of the input */
	INT k1 = mxGetN(prhs[0]); /* SRB: total possible number of columns */
	INT k = mxGetM(prhs[3]) * mxGetN(prhs[3]);  /* SRB: this is index */
	
	/* SRB, Nov 9 2009, adding support for the empty matrix input */
	DOUBLE *columnIndex;
	INT DEFAULT_INDEX = 0;
	INT i;
	if ((k == 0) || mxIsEmpty(prhs[3])) {
		columnIndex = (DOUBLE *) mxMalloc(k1 * sizeof(DOUBLE));
		k = k1;
		DEFAULT_INDEX = 1;
		for (i = 0; i < k1; ++i) {
			columnIndex[i] = (DOUBLE) i + 1;  /* MATLAB is 1-based */
		}
	} else {
		columnIndex = mxGetPr(prhs[3]);
	}

	/* Create/allocate return argument, a 1x1 real-valued Matrix */
	plhs[0] = mxCreateNumericMatrix(n, 1, MXPRECISION_CLASS, mxREAL);
	plhs[1] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	if (nlhs > 2) {
		plhs[2] = mxCreateNumericMatrix(1, 1, MXPRECISION_CLASS, mxREAL);
	}

	DOUBLE *work = mxMalloc(k * sizeof(DOUBLE));

	memcpy((void *) mxGetPr(plhs[0]), (void *) mxGetPr(prhs[1]), n * sizeof(DOUBLE));
	memcpy((void *) mxGetPr(plhs[1]), (void *) mxGetPr(prhs[2]), sizeof(DOUBLE));
	INT imethod = (INT) mxGetScalar(prhs[5]);

	INT LDV = n;
	INT inre = 0;

	reorth(&n, &k, mxGetPr(prhs[0]), &LDV, mxGetPr(plhs[0]), 
	  mxGetPr(plhs[1]), columnIndex, mxGetPr(prhs[4]), 
	  work, &imethod, &inre);
	
	if (nlhs > 2) {
		*(mxGetPr(plhs[2])) = (DOUBLE) inre * k;
	}

	mxFree(work);
	if (DEFAULT_INDEX) {
		mxFree(columnIndex);
	}
}



