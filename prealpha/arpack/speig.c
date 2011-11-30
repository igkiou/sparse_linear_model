/* {{{1 GNU General Public License

Program Tops - a stack-based computing environment
Copyright (C) 1999-2005  Dale R. Williamson

Author and copyright holder of speig.c: Al Danial <al.danial@gmail.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software Foundation,
Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
}}}1 */

#if defined(ARPACK)
/* headers {{{1 */
#include <stdio.h>
#include <stdlib.h>        /* free    */
#include <string.h>        /* strncmp */
#include <math.h>          /* fabs    */

#include "stk.h"
#include "main.h"
#include "exe.h"           /* xmain */
#include "lapack.h"        /* dcopy, dgttrf, dgttrs, daxpy, dscal, dnrm2 */
#include "sparse.h"
#include "tag.h"
#include "spsolve.h"

#include "ctrl.h"
#include "inpo.h"
#include "mem.h"

#ifdef FORT_UDSC
#define DSAUPD dsaupd_
#define DSEUPD dseupd_
#define DMOUT  dmout_ 
#else
#define DSAUPD dsaupd
#define DSEUPD dseupd
#define DMOUT  dmout
#endif
void   DSAUPD ();
void   DSEUPD ();
void   DMOUT  ();

void dsaupd_error(int info);
void mat_vec_av (int n, double v[], double w[]);
void mat_vec_mv ( int n, double *v, double *w);
void print_int(char *title, int n, int *array);
void print_mat(char *title, int nR, int nC, double *mat, int nR_desired);
/* headers 1}}} */

/* words */
int  speig_near_shift()   /* (hA_sp hB_sp sigma n ---  hEval hEvec nBelow) {{{1 */     
/* 
 * man entry:  speig_near_shift {{{2
 * (hA_sp hB_sp sigma n ---  hEval hEvec nBelow) Returns the n eigenpairs closest to the shift value sigma for the real symmetric general eigenvalue problem [A]{x} = lambda [B]{x}.  nBelow is the number of eigenvalues below sigma.  [A] and [B] must be real, symmetric, sparse matrices.  The solution is obtained by calling the ARPACK subroutines dsaupd and dseupd.
 * category: math::matrix::sparse
 * related: sparse, spadd, spmult, solve, splu, spfbs, spdiag, eig, zggev
 * 2}}}
 */ 
{
/* based on ARPACK/EXAMPLES/SYM/dsdrv4.f */
int DEBUG = 0;
/* 2=solve demo problem */
/*
#define DEBUG2 2
*/

/*
#define MAXN   256
#define MAXNEV 10
#define MAXNCV 25
#define LDF    MAXN
*/
    /*
    int     maxn=MAXN, maxnev=MAXNEV, maxncv=MAXNCV, ldv=LDF;
    double  v[ldv*maxncv], 
            workl[maxncv*(maxncv+8)],
            workd[3*maxn], 
            d[maxncv*2], 
            resid[maxn], 
            ad[maxn], 
            adl[maxn], 
            adu[maxn], 
            adu2[maxn];
    int     ipiv[maxn], select[maxncv];
     */
    int     maxn, maxnev, maxncv, ldv;
    double  *v, *workl, *workd, *d, *resid;
    int     *ipiv, *select;

    int     iparam[11], c_ipntr[11], fort_ipntr[11];
    
    char   *bmat         = "G"  ,
           *which        = "LM" ,
           *All          = "All";
    int     ido;
    #define FBS_S_MULT_B -1     /* ido == -1: do  y <= inv(A - s*B)*B*x */
    #define FBS_S         1     /* ido ==  1: do  y <= inv(A - s*B)*x   */
    #define MULT_B        2     /* ido ==  2: do  y <= B*x              */
    int     n, nev, ncv, lworkl, info, j, ierr = 0,
            nconv = 0, maxitr, ishfts, mode, iterating, 
            Xvec_bytes, Xvec_cols, nBTot, nBytes, sturm_count = 0;

    int     rvec;           /* arg to dseupd as a Fortran logical */
    double  sigma, tol, zero = 0.0, *Xvec, *U_diagonal;

    SparseMatrix A, B;
#ifdef DEBUG2
    double  *ad, *adl, *adu, *adu2;
    char   *no_trans     = "Notranspose",
           *ritz_message = "Ritz values and relative residuals                                                          ";
    double  r1, r2, djm, h;
    double  one = 1.0, four = 4.0, six = 6.0, two = 2.0;
    int     int_1 = 1, int_2 = 2, int_6 = 6, int_m6 = -6;
#endif
    

    /*
    N   = dimension of [A] and [B]
    NEV = desired number of eigenvalues/eigenvectors
    NCV = number of Lanczos vectors to generate at each iteration
    */

    /* type check the inputs {{{2 */
    if (!popint(&nev)) return 0;
    if (!popd(&sigma)) return 0;  /* need sigma both on the stack and as */
    pushd(sigma);                 /* a local variable to pass to DSEUPD  */

    if (!is_sparse(tos-1)) {
        stkerr(" speig_near_shift: [B] ",SPARSENOT);
        return 0;
    }
    if (!is_symm(tos-1)) {
        stkerr(" speig_near_shift: [B] "," not symmetric");
        return 0;
    }
    if (is_complex(tos-1)) {
        stkerr(" speig_near_shift: [B] "," cannot be complex");
        return 0;
    }
    if (!is_sparse(tos-2)) {
        stkerr(" speig_near_shift: [A] ",SPARSENOT);
        return 0;
    }
    if (!is_symm(tos-2)) {
        stkerr(" speig_near_shift: [A] "," not symmetric");
        return 0;
    }
    if (is_complex(tos-2)) {
        stkerr(" speig_near_shift: [A] "," cannot be complex");
        return 0;
    }

    B = sparse_overlay(tos-1);
    A = sparse_overlay(tos-2);

    if ((A.H[COLS] != B.H[COLS]) || 
        (A.H[ROWS] != B.H[ROWS])) {
        stkerr(" speig_near_shift: ",MATSNOTC);
        return 0;
    }
    n   = A.H[ROWS];
    ncv = MIN(nev + 10, n);
    Xvec_cols  = 1;  /* can't do block Arnoldi until UMFPACK FBS does >1 col */
    Xvec_bytes = n * Xvec_cols * sizeof(double);
    /* 2}}} */

    /* allocate memory for work arrays {{{2 */
    maxn   = n;
    maxnev = nev;
    maxncv = ncv;
    ldv    = n;
    nBTot  = 0; /* total number of bytes malloc'ed for working arrays */
if (DEBUG) {
gprintf("speig_near_shift nBTot=%d\n", nBTot);
gprintf("n=%d  nev=%d  ncv=%d  ldv=%d  maxn=%d  maxnev=%d  maxncv=%d\n",
n, nev, ncv, ldv, maxn, maxnev, maxncv);
}

    nBytes = maxn * sizeof(int);   nBTot += nBytes;
    if ((ipiv = (int *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (ipiv): ",MEMNOT);
        return 0;
    }
    /* select[] is passed to dseupd in place of a Fortran array of logical's */
    nBytes = maxncv * sizeof(int);   nBTot += nBytes;
    if ((select = (int *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (select): ",MEMNOT);
        return 0;
    }

    nBytes = ldv*maxncv * sizeof(double);   nBTot += nBytes;
    if ((v = (double *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (v): ",MEMNOT);
        return 0;
    }
    nBytes = maxncv*(maxncv + 8) * sizeof(double);   nBTot += nBytes;
    if ((workl = (double *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (workl): ",MEMNOT);
        return 0;
    }
    nBytes = 3*maxn * sizeof(double);   nBTot += nBytes;
    if ((workd = (double *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (workd): ",MEMNOT);
        return 0;
    }
    nBytes = maxncv*2 * sizeof(double);   nBTot += nBytes;
    if ((d = (double *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (d): ",MEMNOT);
        return 0;
    }
    nBytes = maxn * sizeof(double);   nBTot += nBytes;
    if ((resid = (double *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (resid): ",MEMNOT);
        return 0;
    }
#ifdef DEBUG2
    nBytes = maxn * sizeof(double);   nBTot += nBytes;
    if ((ad = (double *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (ad): ",MEMNOT);
        return 0;
    }
    nBytes = maxn * sizeof(double);   nBTot += nBytes;
    if ((adl = (double *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (adl): ",MEMNOT);
        return 0;
    }
    nBytes = maxn * sizeof(double);   nBTot += nBytes;
    if ((adu = (double *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (adu): ",MEMNOT);
        return 0;
    }
    nBytes = maxn * sizeof(double);   nBTot += nBytes;
    if ((adu2 = (double *) malloc(nBytes)) == NULL) {
        stkerr(" speig_near_shift (adu2): ",MEMNOT);
        return 0;
    }
#endif
/* 2}}} */

gprintf("speig_near_shift nBTot=%d\n", nBTot);

    /* shift the matrices together */
    pushstr("over 4 revn 3 revn swap");
    xmain(0);                       /* ( A B sigma    --- B A B   sigma ) */
    pushstr("-1.0 * "); xmain(0);  /* ( B A B  sigma --- B A B  -sigma ) */
    pushstr(" * + "  ); xmain(0);  /* ( B A B -sigma --- B A-sigma*B   ) */
    pushstr("splu "  ); xmain(0);  /* ( B A-sigma*B hLU ) */
    pushstr("dup "   ); xmain(0);  /* ( B A-sigma*B hLU hLU ) */
/*
return 1;
*/
    pushstr("spdiag "); xmain(0);  /* ( B A-sigma*B hLU U_diagonal) */
    if (is_factor_lu(tos)) {
        stkerr(" speig_near_shift: "," expected U diagonal, found U factor");
        return 0;
    }
    U_diagonal = tos->mat;
    for (j = 0; j < tos->row*tos->col; j++) {
        if (U_diagonal[j] < 0.0) {
            ++sturm_count;
        }
    }
    drop(); /* remove U diagonal from stack */
    /*
    if (!popint(&sturm_count)) return 0;
    */
    
#ifdef DEBUG2
    n     =  100;
    nev   =   4;
    ncv   =  10;
    sigma = 0.0;
#endif
    if ( n > maxn ) {
        printf(" ERROR with _SDRV4: N is greater than MAXN \n");
        return 0;
    } else if ( nev  >  maxnev ) { 
        printf(" ERROR with _SDRV4: NEV is greater than MAXNEV \n");
        return 0;
    } else if ( ncv  >  maxncv ) { 
        printf(" ERROR with _SDRV4: NCV is greater than MAXNCV \n");
        return 0;
    }
    
    lworkl = ncv*(ncv+8);
    tol    = zero ; /* make > 0 for faster, less accurate results */
    ido    = 0;     /* reverse communication flag                 */
    info   = 0;     /* 0 = use a random starting vector in DSAUPD */
    
    for (j = 0; j < 11; j++) {
        iparam[j]     = 0;
        c_ipntr[j]    = 0;
        fort_ipntr[j] = 0;
    }

    ishfts =   1;
    maxitr = 300;
    mode   =   3;
    iparam[ 0 ] = ishfts;   /* see ARPACK docs for DSAUPD for */
    iparam[ 2 ] = maxitr;   /* code control options           */
    iparam[ 6 ] = mode  ;
    
#ifdef DEBUG2
   /*-------------------------------------------------------%
    | Call LAPACK routine to factor the tridiagonal matrix  |
    | (A-SIGMA*M).  The matrix A is the 1-d discrete        |
    | Laplacian. The matrix M is the associated mass matrix |
    | arising from using piecewise linear finite elements   |
    | on the interval [0, 1].                               |
    %-------------------------------------------------------*/
    
    h  = one / ( (double) (n+1));
    r1 = (four / six) * h;
    r2 = (one / six) * h;
    for (j = 0; j < n; j++) {
        ad[j]  =  two/h - sigma * r1;
        adl[j] = -one/h - sigma * r2;
    }
    
    DCOPY  (&n, adl, &int_1, adu, &int_1);
    DGTTRF (&n, adl, ad, adu, adu2, &ipiv, &ierr);
/*
print_mat("adu",  maxn, 1, adu , n);
print_mat("adu2", maxn, 1, adu2, n);
*/
    if (ierr  !=  0) { 
        printf(" Error with _gttrf in _SDRV4.\n");
        return 0;
    }
#endif
    
   /*-------------------------------------------%
    | M A I N   L O O P (Reverse communication) |
    %-------------------------------------------*/
 
    iterating = 1;
    while (iterating) {
 
       /*---------------------------------------------%
        | Repeatedly call the routine DSAUPD and take |
        | actions indicated by parameter IDO until    |
        | either convergence is indicated or maxitr   |
        | has been exceeded.                          |
        %---------------------------------------------*/
        
        DSAUPD (&ido, bmat, &n, which, &nev, &tol, resid,
                &ncv, v, &ldv, iparam, fort_ipntr, workd, workl,
                &lworkl, &info );
        if (info) {
            dsaupd_error(info);
            if ((info != 1) && (info != 2)) return 0;
        }
        
        /* Need both 0-based and 1-based values of the pointer array 
         * for C and Fortran codes respectively.
         */
        for (j = 0; j < 11; j++) {
            c_ipntr[j] = fort_ipntr[j];
            if (fort_ipntr[j])
                --c_ipntr[j];
        }
if (DEBUG) {
print_int("dsaupd c_ipntr  1:", 11, c_ipntr );
print_int("dsaupd fort_ipntr  1:", 11, fort_ipntr );
/*
print_int("dsaupd iparam 1:", 11, iparam);
print_mat("dsaupd v:", ldv, 4, v, n);
*/
}
        switch (ido) {
        case FBS_S_MULT_B:         /* ido == -1 */
        
           /*--------------------------------------------%
            | Perform  y <--- OP*x = inv[A-SIGMA*M]*M*x  |
            | to force the starting vector into the      |
            | range of OP.  The user should supply       |
            | his/her own matrix vector multiplication   |
            | routine and a linear system solver here.   |
            | The matrix vector multiplication routine   |
            | takes workd(ipntr(1)) as the input vector. |
            | The final result is returned to            |
            | workd(ipntr(2)).                           |
            %--------------------------------------------*/

            pushstr("2 pick"); xmain(0); 
            /* ( B A-sigma*B hLU --- B A-sigma*B hLU B) */

            if (!matstk(n, Xvec_cols, "_Xvec")) return 0;
            Xvec = tos->mat;
            memcpy(Xvec, &workd[c_ipntr[ 0 ]], Xvec_bytes);
            /* ( B A-sigma*B hLU B --- B A-sigma*B hLU B Xvec ) */

            pushstr("* spfbs"); xmain(0); 
            /* ( B A-sigma*B hLU hX )   where X = inv(A-sigma*B)*B*Xvec */
            memcpy(&workd[c_ipntr[ 1 ]], tos->mat, Xvec_bytes);
            drop();  /* ( B A-sigma*B hLU ) */

#ifdef DEBUG2
            mat_vec_mv(n, &workd[c_ipntr[ 0 ]], &workd[c_ipntr[ 1 ]] );
            
            DGTTRS (no_trans,     &n, &int_1, adl, ad, adu, adu2, ipiv, 
                   &workd[c_ipntr[ 1 ]], &n, &ierr);
            if (ierr) { 
                printf(" \n");
                printf(" Error with _gttrs in _SDRV4. \n");
                printf("\n");
                return 0;
            }
#endif
            break;
            
        case FBS_S:                /* ido ==  1 */
 
           /*-----------------------------------------%
            | Perform y <-- OP*x = inv[A-sigma*M]*M*x |
            | M*x has been saved in workd(ipntr(3)).  |
            | the user only needs the linear system   |
            | solver here that takes workd(ipntr(3)   |
            | as input, and returns the result to     |
            | workd(ipntr(2)).                        | 
            %-----------------------------------------*/

            if (!matstk(n, Xvec_cols, "_Xvec")) return 0;
            Xvec = tos->mat;
            memcpy(Xvec, &workd[c_ipntr[ 2 ]], Xvec_bytes);
            /* ( B A-sigma*B hLU Xvec ) */

            pushstr("spfbs"); xmain(0); 
            /* ( B A-sigma*B hLU hX )   where X = inv(A-sigma*B)*Xvec */
            memcpy(&workd[c_ipntr[ 1 ]], tos->mat, Xvec_bytes);
            drop();  /* ( B A-sigma*B hLU ) */
            
#ifdef DEBUG2
            DCOPY  (&n, &workd[c_ipntr[2]], &int_1, &workd[c_ipntr[1]], &int_1);
            DGTTRS (no_trans, &n, &int_1, adl, ad, adu, adu2, ipiv, 
                   &workd[c_ipntr[1]], &n, &ierr);
            if (ierr) { 
               printf(" \n");
               printf(" Error with _gttrs in _SDRV4.\n");
               printf(" \n"); 
               return 0;
            }
#endif
            break;
            
        case MULT_B:               /* ido ==  2 */
 
           /*-----------------------------------------%
            |          Perform  y <--- M*x            |
            | Need the matrix vector multiplication   |
            | routine here that takes workd(ipntr(1)) |
            | as the input and returns the result to  |
            | workd(ipntr(2)).                        |
            %-----------------------------------------*/

            pushstr("2 pick"); xmain(0); 
            /* ( B A-sigma*B hLU --- B A-sigma*B hLU B) */

            if (!matstk(n, Xvec_cols, "_Xvec")) return 0;
            Xvec = tos->mat;
            memcpy(Xvec, &workd[c_ipntr[ 0 ]], Xvec_bytes);
            /* ( B A-sigma*B hLU B --- B A-sigma*B hLU B Xvec ) */

            pushstr("*"); xmain(0); 
            /* ( B A-sigma*B hLU hX )   where X = B*Xvec */
            memcpy(&workd[c_ipntr[ 1 ]], tos->mat, Xvec_bytes);
            drop();  /* ( B A-sigma*B hLU ) */
            
#ifdef DEBUG2
            mat_vec_mv (n, &workd[c_ipntr[0]], &workd[c_ipntr[1]]);
#endif
            break;
            
        default:   /* finished iterations; either converged or hit an error */
            iterating = 0;
        }
    }
 
    drop(); /* remove LU        */
    drop(); /* remove A-sigma*B */
    drop(); /* remove B         */
    
   /*-------------------------------------------%
    | No fatal errors occurred.                 |
    | Post-Process using DSEUPD.                |
    |                                           |
    | Computed eigenvalues may be extracted.    |
    |                                           |
    | Eigenvectors may also be computed now if  |
    | desired.  (indicated by rvec = .true.)    |
    %-------------------------------------------*/
    
    rvec = 1; /* 1 == .true. ;    0 == .false. */
 
if (DEBUG) {
nconv = iparam[4];
print_mat("before dseupd v:", ldv, nconv, v, n);
print_mat("before dseupd d", maxncv, 2, d, nconv);
print_mat("before dseupd workl", maxncv, 4, workl, maxncv);
print_mat("before dseupd workd", 3, 4, workd, 3);
print_mat("before dseupd resid", n, 1, resid, n);
printf(  "sigma %e; n %d; nev %d; ncv %d; ldv %d; lworkl %d; ierr %d\n", 
          sigma, n, nev, ncv, ldv, lworkl, ierr);
print_int("before dseupd c_ipntr ", 11, c_ipntr );
print_int("before dseupd fort_ipntr ", 11, fort_ipntr );
print_int("before dseupd iparam ", 11, iparam );
}
    DSEUPD (&rvec, All, select, d, v, &ldv, &sigma,
            bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, 
            iparam, fort_ipntr, workd, workl, &lworkl, &ierr );
if (DEBUG) {
printf("after dseupd ldv=%d  nconv=%d  n=%d\n", ldv, nconv, n);
print_mat("after dseupd [v]", ldv, nconv, v, 5);
}
 
   /*----------------------------------------------%
    | Eigenvalues are returned in the first column |
    | of the two dimensional array D and the       |
    | corresponding eigenvectors are returned in   |
    | the first NEV columns of the two dimensional |
    | array V if requested.  Otherwise, an         |
    | orthogonal basis for the invariant subspace  |
    | corresponding to the eigenvalues in D is     |
    | returned in V.                               |
    %----------------------------------------------*/

    if (!matstk(nev, 1, "_Eval")) return 0;
    Xvec = tos->mat;
    memcpy(Xvec, d, nev*sizeof(double));

    if (!matstk(n, nev, "_Evec")) return 0;
    Xvec = tos->mat;
    for (j = 0; j < nev; j++) {
        memcpy(&Xvec[j*n], &v[j*ldv], n*sizeof(double));
    }
    pushd(sturm_count);
    
    if (ierr) { 
    
       /*------------------------------------%
        | Error condition:                   |
        | Check the documentation of DSEUPD. |
        %------------------------------------*/
        
        printf(" \n");
        printf(" Error with _seupd, info = %d\n", ierr);
        printf(" Check the documentation of _seupd \n");
        printf(" \n");
    
    } else {
    
        nconv = iparam[4];
if (DEBUG) print_mat("before av/mv/daxpy v", ldv, nconv, v, n);
        for (j = 0; j < nconv; j++) {
    
           /*---------------------------%
            | Compute the residual norm |
            |                           |
            |   ||  A*x - lambda*x ||   |
            |                           |
            | for the NCONV accurately  |
            | computed eigenvalues and  |
            | eigenvectors.  (iparam(5) |
            | indicates how many are    |
            | accurate to the requested |
            | tolerance)                |
            %---------------------------*/
    
#ifdef DEBUG2
            mat_vec_av(n, &v[ldv*j],  workd);
            mat_vec_mv(n, &v[ldv*j], &workd[n]);
            djm = -d[j];
            DAXPY (&n, &djm, &workd[n], &int_1, workd, &int_1);
            d[j+maxncv] = DNRM2(&n, workd, &int_1);
            d[j+maxncv] = d[j+maxncv] / fabs(d[j]);
#endif
    
        }
    
if (DEBUG) print_mat("after av/mv/daxpy d", maxncv, 2, d, nconv);
#ifdef DEBUG2
        DMOUT (&int_6, &nconv, &int_2, d, &maxncv, &int_m6,
              ritz_message);
#endif
    }
    
if (DEBUG) {
nc();
gprintf("speig_near_shift "); nc();
gprintf("  Size of the matrix is %d", n); nc();
gprintf("  The number of Ritz values requested is %d", nev); nc();
gprintf("  The number of Arnoldi vectors generated (NCV) is %d", ncv); nc();
gprintf("  What portion of the spectrum: %s", which); nc();
gprintf("  The number of converged Ritz values is %d", nconv ); nc();
gprintf("  The number of Implicit Arnoldi update iterations taken is %d", 
         iparam[2]); nc();
gprintf("  The number of OP*x is %d", iparam[8]); nc();
gprintf("  The convergence criterion is %e", tol); nc();
}

    free(ipiv);
    free(select);
    free(v);
    free(workl);
    free(workd);
    free(d);
    free(resid);
#ifdef DEBUG2
    free(ad);
    free(adl);
    free(adu);
    free(adu2);
#endif
    
    return 1; 
}
/* 1}}} */
 
void dsaupd_error(int info) { /* {{{1 */
    switch (info) {
        case 0:
            break; /* Normal exit. */
        case 1:
            break; /*
                      Maximum number of iterations taken.
                      All possible eigenvalues of OP has been found. IPARAM(5)  
                      returns the number of wanted converged Ritz values.
                    */
        case 2:
            break; /*
                      No longer an informational error. Deprecated starting
                      with release 2 of ARPACK.
                    */
        case 3:
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("No shifts could be applied during a cycle of the ");
            nc();
            gprintf("Implicitly restarted Arnoldi iteration. One possibility ");
            nc();
            gprintf("is to increase the size of NCV relative to NEV. ");
            nc();
            gprintf("See remark 4 in ARPACK/SRC/dsaupd.f.");
            nc();
            break;
        case -1:
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("N must be positive.");
            nc();
            break;
        case -2:
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("NEV must be positive.");
            nc();
            break;
        case -3:  
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("NCV must be > NEV and <= N.");
            nc();
            break;
        case -4:  
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("The maximum number of Arnoldi update iterations allowed");
            nc();
            gprintf("must be greater than zero.");
            nc();
            break;
        case -5:  
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.");
            nc();
            break;
        case -6:  
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("BMAT must be one of 'I' or 'G'.");
            nc();
            break;
        case -7:  
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("Length of private work array WORKL is not sufficient.");
            nc();
            break;
        case -8:  
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("Error return from trid. eigenvalue calculation;");
            nc();
            gprintf("Informatinal error from LAPACK routine dsteqr.");
            nc();
            break;
        case -9:  
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("Starting vector is zero.");
            nc();
            break;
        case -10: 
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("IPARAM(7) must be 1,2,3,4,5.");
            nc();
            break;
        case -11: 
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("IPARAM(7) = 1 and BMAT = 'G' are incompatable.");
            nc();
            break;
        case -12: 
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("IPARAM(1) must be equal to 0 or 1.");
            nc();
            break;
        case -13: 
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("NEV and WHICH = 'BE' are incompatable.");
            nc();
            break;
        case -9999: 
            gprintf("speig_near_shift/dsaupd info=%d", info); nc();
            gprintf("Could not build an Arnoldi factorization.");
            nc();
            gprintf("IPARAM(5) returns the size of the current Arnoldi");
            nc();
            gprintf("factorization. The user is advised to check that");
            nc();
            gprintf("enough workspace and array storage has been allocated.");
            nc();
        default:
            break;
    }
    return;
}
/* 1}}} */
/* Matrix vector multiplication routines used for debugging the ARPACK
 * interface.  They are C versions of subroutines mv and av from
 * ARPACK/EXAMPLES/SYM/dsdrv4.f
 */
void mat_vec_mv ( int n, double *v, double *w) /* {{{1 */
{
      int         j;
      double h, one = 1.0, four = 4.0, six = 6.0;
      int int_1 = 1;
 
      w[0] =  four*v[0] + v[1];
      /*
      w(1) =  four*v(1) + v(2);
      do 100 j = 2,n-1;
      */
      for (j = 1; j < n-1; j++) {
         w[j] = v[j-1] + four*v[j] + v[j+1] ;
      }
      j = n-1;
      w[j] = v[j-1] + four*v[j] ;
/*
c     Scale the vector w by h.
*/
      h = one / ( six*((double) (n+1)) );
      DSCAL(&n, &h, w, &int_1);
}
/* 1}}} */
void mat_vec_av (int n, double v[], double w[]) /* {{{1 */
{
      int     j;
      double  h, one = 1.0, two = 2.0, np1;
      int int_1 = 1;
 
      w[0] =  two*v[0] - v[1];
      for (j = 1; j < n-1; j++) {
         w[j] = - v[j-1] + two*v[j] - v[j+1]; 
      }
      j = n-1;
      w[j] = - v[j-1] + two*v[j]; 
/*
c     Scale the vector w by (1/h)
*/
      h = one / ( (double) (n+1) );
      np1 = (double) n + 1;
      DSCAL(&n, &np1, w, &int_1);
}
/* 1}}} */
void print_mat(char *title, int nR, int nC, double *mat, int nR_desired) { /* {{{1 */
    int r, c;
    printf("%s\n", title);
    for (r = 0; r < nR_desired; r++) {
        printf("%3d ", r+1);
        for (c = 0; c < nC; c++) {
            printf("% 16.6e ", mat[nR*c + r]);
        }
        printf("\n");
    }
}
/* 1}}} */
void print_int(char *title, int n, int *array) { /* {{{1 */
    int r;
    printf("%s\n", title);
    for (r = 0; r < n; r++) {
        printf("%4d ", array[r]);
    }
    printf("\n");
}
/* 1}}} */
#endif
