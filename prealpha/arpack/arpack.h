
//enum _NAUPD_ERRORS {
//	0: "Normal exit.",
//	1: "Maximum number of iterations taken. "
//	   "All possible eigenvalues of OP has been found.",
//	2: "No longer an informational error. Deprecated starting with "
//	   "release 2 of ARPACK.",
//	3: "No shifts could be applied during a cycle of the Implicitly "
//	   "restarted Arnoldi iteration. One possibility is to increase "
//	   "the size of NCV relative to NEV. ",
//	-1: "N must be positive.",
//	-2: "NEV must be positive.",
//	-3: "NCV must be greater than NEV and less than or equal to N.",
//	-4: "The maximum number of Arnoldi update iterations allowed "
//		"must be greater than zero.",
//	-5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
//	-6: "BMAT must be one of 'I' or 'G'.",
//	-7: "Length of private work array WORKL is not sufficient.",
//	-8: "Error return from trid. eigenvalue calculation; "
//		"Informational error from LAPACK routine dsteqr .",
//	-9: "Starting vector is zero.",
//	-10: "IPARAM(7) must be 1,2,3,4,5.",
//	-11: "IPARAM(7) = 1 and BMAT = 'G' are incompatable.",
//	-12: "IPARAM(1) must be equal to 0 or 1.",
//	-13: "NEV and WHICH = 'BE' are incompatable. ",
//	-9999: "Could not build an Arnoldi factorization. "
//		   "IPARAM(5) returns the size of the current Arnoldi "
//		   "factorization. The user is advised to check that "
//		   "enough workspace and array storage has been allocated.",
//}

enum _NAUPD_ERRORS {
	NORMAL = 0,
	MAXITER = 1,
	DEPREC = 2,
	NOSHIFTS = 3,
	NPOS = -1,
	NEVPOS = -2,
	NCVREL = -3,
	MAXITERPOS = -4,
	WHICHTYPE = -5,
	BMATTYPE = -6,
	WORKLSM = -7,
	STEQRERROR = -8,
	INITVECZERO = -9,
	IPARAM7TYPE = -10,
	IPARAM7BMAT = -11,
	NEVWHICH = -13,
	ARNOLDIERROR = -9999
};

//_NEUPD_ERRORS = {
//    0: "Normal exit.",
//    1: "The Schur form computed by LAPACK routine dlahqr "
//       "could not be reordered by LAPACK routine dtrsen. "
//       "Re-enter subroutine dneupd  with IPARAM(5)NCV and "
//       "increase the size of the arrays DR and DI to have "
//       "dimension at least dimension NCV and allocate at least NCV "
//       "columns for Z. NOTE: Not necessary if Z and V share "
//       "the same space. Please notify the authors if this error"
//       "occurs.",
//    -1: "N must be positive.",
//    -2: "NEV must be positive.",
//    -3: "NCV-NEV >= 2 and less than or equal to N.",
//    -5: "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'",
//    -6: "BMAT must be one of 'I' or 'G'.",
//    -7: "Length of private work WORKL array is not sufficient.",
//    -8: "Error return from calculation of a real Schur form. "
//        "Informational error from LAPACK routine dlahqr .",
//    -9: "Error return from calculation of eigenvectors. "
//        "Informational error from LAPACK routine dtrevc.",
//    -10: "IPARAM(7) must be 1,2,3,4.",
//    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
//    -12: "HOWMNY = 'S' not yet implemented",
//    -13: "HOWMNY must be one of 'A' or 'P' if RVEC = .true.",
//    -14: "DNAUPD  did not find any eigenvalues to sufficient "
//         "accuracy.",
//    -15: "DNEUPD got a different count of the number of converged "
//         "Ritz values than DNAUPD got.  This indicates the user "
//         "probably made an error in passing data from DNAUPD to "
//         "DNEUPD or that the data was modified before entering "
//         "DNEUPD",
//}

enum _NEUPD_ERRORS {
	NORMAL = 0,
	SCHUR = 1,
	NPOS = -1,
	NEVPOS = -2,
	NCVNEV = -3,
	WHICHTYPE = -5,
	BMATTYPE = -6,
	WORKSM = -7,
	LAHQRERROR = -8,
	TREVCERROR = -9,
	IPARAM7TYPE = -10,
	IPARAM7BMAT = -11,
	HOWMNYERROR = -12,
	HOWMNYTYPE = -13,
	NAUPDACC = -14,
	NEUPDNAUPD = -15
};

//_SEUPD_ERRORS = {
//    0: "Normal exit.",
//    -1: "N must be positive.",
//    -2: "NEV must be positive.",
//    -3: "NCV must be greater than NEV and less than or equal to N.",
//    -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
//    -6: "BMAT must be one of 'I' or 'G'.",
//    -7: "Length of private work WORKL array is not sufficient.",
//    -8: ("Error return from trid. eigenvalue calculation; "
//         "Information error from LAPACK routine dsteqr."),
//    -9: "Starting vector is zero.",
//    -10: "IPARAM(7) must be 1,2,3,4,5.",
//    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
//    -12: "NEV and WHICH = 'BE' are incompatible.",
//    -14: "DSAUPD  did not find any eigenvalues to sufficient accuracy.",
//    -15: "HOWMNY must be one of 'A' or 'S' if RVEC = .true.",
//    -16: "HOWMNY = 'S' not yet implemented",
//    -17: ("DSEUPD  got a different count of the number of converged "
//          "Ritz values than DSAUPD  got.  This indicates the user "
//          "probably made an error in passing data from DSAUPD  to "
//          "DSEUPD  or that the data was modified before entering  "
//          "DSEUPD.")
//}

enum _SEUPD_ERRORS {
    NORMAL = 0,
    NPOS = -1,
    NEVPOS = -2,
    NCVNEV = -3,
    WHICHTYPE = -5,
    BMATTYPE = -6,
    WORKLSM = -7,
    STEQRERROR = -8,
    INITVECZERO = -9,
    IPARAM7TYPE = -10,
    IPARAM7BMAT = -11,
    NEVWHICH = -12,
    SAUPDACC = -14,
    HOWMNYTYPE = -15,
    HOWMNYERROR = -16,
    SEUPDSAUPD = -17
};

// accepted values of parameter WHICH in _SEUPD
//enum _SEUPD_WHICH {
//	LM, SM, LA, SA, BE
//};

// accepted values of parameter WHICH in _NAUPD
//enum _NAUPD_WHICH {
//	LM, SM, LR, SR, LI, SI
//};

enum WHICH_TYPE {
	LM, SM, LR, SR, LI, SI, LA, SA, BE
};


//int simarpack(double *Amat, int n, int nev, int ncv, double tol, char* which, \
//		int maxitr, int printlevel, double* evals, double* evecs, int* nconv, \
//		int* nummatvec, int *count)
//
//	double *v, *workl, *workd, *d, *resid;
//	int *select, iparam[11], ipntr[11];
//	char bmat[2]="I",all[4]="All"; // ARPACK params we only solve simple eigenvalue problems
//	int ldv, ido, lworkl, info, ierr;
//	int mode, ishfts, rvec,i,incx=1;
//	double sigma,zero,minevalest;
//	double alpha=1.0,beta=0.0;
//
//	// Internal ARPACK parameters and vars
//	ldv=n;zero=0.0;lworkl=ncv*(ncv+10);
//	info=0;ido=0;ishfts =1;mode=1;rvec=1;
//	iparam[1-1]=ishfts;iparam[3-1]=maxitr;iparam[7-1]=mode;
//	*nummatvec=0;
//
//	// Allocate memory for arpack workspace
//	v = (double*)calloc(ldv*ncv, sizeof(double));
//	workl = (double*)calloc(lworkl, sizeof(double));
//	workd = (double*)calloc(3*n, sizeof(double));
//	d = (double*)calloc(ncv*2, sizeof(double));
//	resid = (double*)calloc(n, sizeof(double));
//	select = (int*)calloc(ncv, sizeof(int));
//
//	// ARPACK loop
//	*count=0;
//	do {
//		(*count)++;
//		dsaupd(&ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
//		//if (iparam[3-1]>1000) printf("ARPACK dsaupd number iters: %d.\n", iparam[3-1]);
//		if(ido == -1 || ido == 1) {
//			// *** Matrix vector product, X is at workd+ipntr[1-1]-1 and the product Y is at workd+ipntr[2-1]-1 ***
//			cblas_dgemv(CblasColMajor,CblasNoTrans,n,n,alpha,Amat,n,workd + ipntr[1-1] - 1,incx,beta,workd + ipntr[2-1] - 1,incx);
//			*nummatvec = *nummatvec + 1;}
//		else if(info != 0) {
//			if(printlevel > 0) printf("ARPACK dsaupd error (info = %d).\n", info);}}
//	while(ido != 99);
//	// Post processing
//	dseupd(&rvec, all, select, d, v, &ldv, &sigma, bmat, &n, which, &nev, &tol,\
		resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &ierr);

//ncv >= 2*nev, at least nev + 1.
//ldv >= n
//MKL_INT lworkl = ncv * (ncv + 10); //ARPACK documentation uses +8
//MKL_INT ldv = n;
//DOUBLE *v = z = (DOUBLE *) MALLOC(ldv * ncv * sizeof(DOUBLE)); //ARPACK documentation proposed size, v is eigenvectors
//DOUBLE *workl = (DOUBLE *) MALLOC(lworkl * sizeof(DOUBLE)); //ARPACK documentation uses +8
//DOUBLE *workd = (DOUBLE *) MALLOC(3 * n * sizeof(DOUBLE)); //ARPACK documentation proposed size
//DOUBLE *d = w = (DOUBLE *) MALLOC(2 * ncv * sizeof(DOUBLE)); //ARPACK documentation proposed size, d is eigenvalues
//DOUBLE *resid = (DOUBLE *) MALLOC(n * sizeof(DOUBLE)); //ARPACK documentation proposed size, initialization, alternatively set to previous iter
//MKL_INT *select = (MKL_INT *) MALLOC(ncv * sizeof(MKL_INT)); //ARPACK documentation says just MALLOC(1*sizeof(MKL_INT))
//INFO should be set to 0 if zero init, otherwise to 1. On return, it is an error
//ABSTOL is set to 0 in driver file
//rvec is jobz and should be 0 or 1
void dsyevar(char *jobz, WHICH_TYPE *which, MKL_INT *nev, MKL_INT *n, double *a, \
		double *w, double *z, double *abstol, ARPACK_ERROR_TYPE *info, MKL_INT *nconv, \
		MKL_INT *ncv, MKL_INT *ldv, MKL_INT *maxiter, DOUBLE *v, DOUBLE *workl, \
		MKL_INT *lworkl, DOUBLE *workd, DOUBLE *d, DOUBLE *resid, MKL_INT *select) {

	MKL_INT N = *n;
	char bmat[2] = "I";
	int iparam[11], ipntr[11];
	iparam[1 - 1] = 1; //exact shifts, use always
	iparam[3 - 1] = maxiter;
	iparam[7 - 1] = 1; //set to 1 for regular mode

	char uplo = 'U';
	MKL_INT SYMVN = N;
	DOUBLE alpha = 1.0;
	MKL_INT SYMVLDA = N;
	MKL_INT incx = 1;
	MKL_INT incy = 1;
	DOUBLE beta = 0.0;

	char all[4]= "All";
	char rvec = (*jobz == 'V');
	MKL_INT *ierr;
	DOUBLE *sigma = NULL;

	int ido = 0;
	do {
		dsaupd(&ido, bmat, n, which, nev, abstol, resid, ncv, z, ldv, iparam, ipntr, workd, workl, lworkl, info);

		if ((ido == -1) || (ido == 1)) {
			dsymv(&uplo, &SYMVN, &alpha, a, &SYMVLDA, workd + ipntr[1 - 1] - 1, &incx, &beta, workd + ipntr[2 - 1] - 1, &incy);
		}
	} while ((ido != 99) && (*info == 0)); // fix INFO and IDO conditions

	if (*info < 0) {
		*nconv = -1;
		dsyevar_error(info);
	} else {
		if (*info > 0) {
			dsyevar_error(info);
		}
		dseupd(&rvec, all, select, w, z, ldv, sigma, bmat, n, which, nev, abstol,\
				resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, ierr);
		if (*ierr != 0) {
			*nconv = -1;
			dseupd_error(ierr);
		} else {
			*nconv = iparam[5 - 1];
		}
	}
}

void dsyevasp(char *jobz, WHICH_TYPE *which, MKL_INT *nev, MKL_INT *n, double *a, \
		double *w, double *z, double *abstol, ARPACK_ERROR_TYPE *info, MKL_INT *nconv, \
		MKL_INT *ncv, MKL_INT *ldv, MKL_INT *maxiter, DOUBLE *v, DOUBLE *workl, \
		MKL_INT *lworkl, DOUBLE *workd, DOUBLE *d, DOUBLE *resid, MKL_INT *select, DOUBLE *sigma) {

	MKL_INT N = *n;

	char bmat[2] = "I";
	int iparam[11], ipntr[11];
	iparam[1 - 1] = 1; //exact shifts, use always
	iparam[3 - 1] = maxiter;
	iparam[7 - 1] = 3; //set to 1 for regular mode


	char uplo = 'U';
	MKL_INT SYMVN = N;
	DOUBLE alpha = 1.0;
	MKL_INT SYMVLDA = N;
	MKL_INT incx = 1;
	MKL_INT incy = 1;
	DOUBLE beta = 0.0;

	char all[4]= "All";
	char rvec = (*jobz == 'V');
	MKL_INT *ierr;

	CHAR uplo = 'U';
	MKL_INT POTRSN = N;
	MKL_INT POTRSNRHS = 1;
	MKL_INT POTRSLDA = N;
	MKL_INT POTRSLDB = N;
	MKL_INT INFO;

	MKL_INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		a[iterN * N + iterN] = a[iterN * N + iterN] - *sigma;
	}
	dpotrf(&uplo, &POTRSN, a, &POTRSLDA, &INFO);
	if (INFO != 0) {
		printf("Error, INFO = %d. LAPACK POTRF error.", INFO);
	}

	int ido = 0;
	do {
		dsaupd(&ido, bmat, n, which, nev, abstol, resid, ncv, z, ldv, iparam, ipntr, workd, workl, lworkl, info);

		if ((ido == -1) || (ido == 1)) {

			memcpy((void *)(workd + ipntr[2 - 1] - 1), (void *) (workd + ipntr[1 - 1] - 1), N * sizeof(DOUBLE));
			dpotrs(&uplo, &POTRSN, &POTRSNRHS, a, &POTRSLDA, workd + ipntr[2-1] - 1, &POTRSLDB, &INFO);
			if (INFO != 0) {
				printf("Error, INFO = %d. LAPACK POTRS error.", INFO);
			}
		}
	} while ((ido != 99) && (*info == 0)); // fix INFO and IDO conditions

	if (*info < 0) {
		*nconv = -1;
		dsyevasp_error(info);
	} else {
		if (*info > 0) {
			dsyevasp_error(info);
		}
		dseupd(&rvec, all, select, w, z, ldv, sigma, bmat, n, which, nev, abstol,\
				resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, ierr);
		if (*ierr != 0) {
			*nconv = -1;
			dseupd_error(ierr);
		} else {
			*nconv = iparam[5 - 1];
		}
	}
}

void dsyevas(char *jobz, WHICH_TYPE *which, MKL_INT *nev, MKL_INT *n, double *a, \
		double *w, double *z, double *abstol, ARPACK_ERROR_TYPE *info, MKL_INT *nconv, \
		MKL_INT *ncv, MKL_INT *ldv, MKL_INT *maxiter, DOUBLE *v, DOUBLE *workl, \
		MKL_INT *lworkl, DOUBLE *workd, DOUBLE *d, DOUBLE *resid, MKL_INT *select, DOUBLE *sigma, \
		DOUBLE *workf, MKL_INT *lworkf, MKL_INT *ipiv) {

	MKL_INT N = *n;

	if(*lworkf == -1) {
		CHAR uplo = 'U';
		MKL_INT SYTRSN = N;
		MKL_INT SYTRSNRHS = 1;
		MKL_INT SYTRSLDA = N;
		MKL_INT SYTRSLDB = N;
		MKL_INT INFO;

		dsytrf(&uplo, &SYTRN, a, &SYTRLDA, ipiv, workf, lworkf, &INFO);
		return;
	}

	char bmat[2] = "I";
	int iparam[11], ipntr[11];
	iparam[1 - 1] = 1; //exact shifts, use always
	iparam[3 - 1] = maxiter;
	iparam[7 - 1] = 3; //set to 1 for regular mode

	char all[4]= "All";
	char rvec = (*jobz == 'V');
	MKL_INT *ierr;

	CHAR uplo = 'U';
	MKL_INT SYTRSN = N;
	MKL_INT SYTRSNRHS = 1;
	DOUBLE alpha = 1.0;
	MKL_INT SYTRSLDA = N;
	MKL_INT SYTRSLDB = N;
	DOUBLE beta = 0.0;
	MKL_INT INFO;

	MKL_INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		a[iterN * N + iterN] = a[iterN * N + iterN] - *sigma;
	}
	dsytrf(&uplo, &SYTRN, a, &SYTRLDA, ipiv, workf, lworkf, &INFO);
	if (INFO != 0) {
		printf("Error, INFO = %d. LAPACK SYTRF error.", INFO);
	}

	int ido = 0;
	do {
		dsaupd(&ido, bmat, n, which, nev, abstol, resid, ncv, z, ldv, iparam, ipntr, workd, workl, lworkl, info);

		if ((ido == -1) || (ido == 1)) {

			memcpy((void *)(workd + ipntr[2 - 1] - 1), (void *) (workd + ipntr[1 - 1] - 1), N * sizeof(DOUBLE));
			dsytrs(&uplo, &SYTRSN, &SYTRSNRHS, a, &SYTRSLDA, ipiv, workd + ipntr[2-1] - 1, &SYTRSLDB, &INFO);
			if (INFO != 0) {
				printf("Error, INFO = %d. LAPACK SYTRS error.", INFO);
			}
		}
	} while ((ido != 99) && (*info == 0)); // fix INFO and IDO conditions

	if (*info < 0) {
		*nconv = -1;
		dsyevas_error(info);
	} else {
		if (*info > 0) {
			dsyevas_error(info);
		}
		dseupd(&rvec, all, select, w, z, ldv, sigma, bmat, n, which, nev, abstol,\
				resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, ierr);
		if (*ierr != 0) {
			*nconv = -1;
			dseupd_error(ierr);
		} else {
			*nconv = iparam[5 - 1];
		}
	}
}

//ncv >= 2*nev, at least nev + 1.
//ldv >= n
//MKL_INT lworkl = 3 * ncv ^ 2 + 6 * ncv; //ARPACK documentation uses +8
//MKL_INT ldv = n;
//DOUBLE *v = z = (DOUBLE *) MALLOC(ldv * ncv * sizeof(DOUBLE)); //ARPACK documentation proposed size, v is eigenvectors
//DOUBLE *workl = (DOUBLE *) MALLOC(lworkl * sizeof(DOUBLE)); //ARPACK documentation uses +8
//DOUBLE *workd = (DOUBLE *) MALLOC(3 * n * sizeof(DOUBLE)); //ARPACK documentation proposed size
//DOUBLE *workev = (DOUBLE *) MALLOC(3 * ncv * sizeof(DOUBLE)); //ARPACK documentation proposed size
//DOUBLE *d = w = (DOUBLE *) MALLOC(3 * ncv * sizeof(DOUBLE)); //ARPACK documentation proposed size, d is eigenvalues
//DOUBLE *resid = (DOUBLE *) MALLOC(n * sizeof(DOUBLE)); //ARPACK documentation proposed size, initialization, alternatively set to previous iter
//MKL_INT *select = (MKL_INT *) MALLOC(ncv * sizeof(MKL_INT)); //ARPACK documentation says just MALLOC(1*sizeof(MKL_INT))
//INFO should be set to 0 if zero init, otherwise to 1. On return, it is an error
//ABSTOL is set to 0 in driver file
//rvec is jobz and should be 0 or 1
void dgeevar(char *jobz, WHICH_TYPE *which, MKL_INT *nev, MKL_INT *n, double *a, \
		double *w, double *z, double *abstol, ARPACK_ERROR_TYPE *info, MKL_INT *nconv, \
		MKL_INT *ncv, MKL_INT *ldv, MKL_INT *maxiter, DOUBLE *v, DOUBLE *workl, \
		MKL_INT *lworkl, DOUBLE *workd, DOUBLE *workev, DOUBLE *d, DOUBLE *resid, MKL_INT *select) {

	MKL_INT N = *n;
	char bmat[2] = "I";
	int iparam[11], ipntr[11];
	iparam[1 - 1] = 1; //exact shifts, use always
	iparam[3 - 1] = maxiter;
	iparam[7 - 1] = 1; //set to 1 for regular mode

	char trans = 'N';
	MKL_INT GEMVM = N;
	MKL_INT GEMVN = N;
	DOUBLE alpha = 1.0;
	MKL_INT GEMVLDA = N;
	MKL_INT incx = 1;
	MKL_INT incy = 1;
	DOUBLE beta = 0.0;

	char all = 'A';
	char rvec = (*jobz == 'V');
	MKL_INT *ierr;
	DOUBLE *sigmar = NULL;
	DOUBLE *sigmai = NULL;

	int ido = 0;
	do {
		dnaupd(&ido, bmat, n, which, nev, abstol, resid, ncv, z, ldv, iparam, ipntr, workd, workl, lworkl, info);

		if ((ido == -1) || (ido == 1)) {
			dgemv(&trans, &GEMVM, &GEMVN, &alpha, a, &GEMVLDA, workd + ipntr[1 - 1] - 1, &incx, &beta, workd + ipntr[2 - 1] - 1, &incy);
		}
	} while ((ido != 99) && (*info == 0)); // fix INFO and IDO conditions

	if (*info < 0) {
		*nconv = -1;
		dgeevar_error(info);
	} else {
		if (*info > 0) {
			dgeevar_error(info);
		}
		dneupd(&rvec, &all, select, w, &w[*ncv], z, ldv, sigmar, sigmai, workev, bmat, n, which, nev, abstol,\
				resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, ierr);
		if (*ierr != 0) {
			*nconv = -1;
			dneupd_error(ierr);
		} else {
			*nconv = iparam[5 - 1];
		}
	}
}

void dgeevas(char *jobz, WHICH_TYPE *which, MKL_INT *nev, MKL_INT *n, double *a, \
		double *w, double *z, double *abstol, ARPACK_ERROR_TYPE *info, MKL_INT *nconv, \
		MKL_INT *ncv, MKL_INT *ldv, MKL_INT *maxiter, DOUBLE *v, DOUBLE *workl, \
		MKL_INT *lworkl, DOUBLE *workd, DOUBLE *workev, DOUBLE *d, DOUBLE *resid, MKL_INT *select, \
		DOUBLE *sigmar, DOUBLE *sigmai, MKL_INT *ipiv) {

	MKL_INT N = *n;

	char bmat[2] = "I";
	int iparam[11], ipntr[11];
	iparam[1 - 1] = 1; //exact shifts, use always
	iparam[3 - 1] = maxiter;
	iparam[7 - 1] = 3; //set to 1 for regular mode

	char all = 'A';
	char rvec = (*jobz == 'V');
	MKL_INT *ierr;

	CHAR trans = 'N';
	MKL_INT GETRSN = N;
	MKL_INT GETRSNRHS = 1;
	DOUBLE alpha = 1.0;
	MKL_INT GETRSLDA = N;
	MKL_INT GETRSLDB = N;
	DOUBLE beta = 0.0;
	MKL_INT INFO;

	MKL_INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		a[iterN * N + iterN] = a[iterN * N + iterN] - *sigma;
	}
	dgetrf(&GETRM, &GETRN, a, &GETRLDA, ipiv, &INFO);
	if (INFO != 0) {
		printf("Error, INFO = %d. LAPACK GETRF error.", INFO);
	}

	int ido = 0;
	do {
		dnaupd(&ido, bmat, n, which, nev, abstol, resid, ncv, z, ldv, iparam, ipntr, workd, workl, lworkl, info);

		if ((ido == -1) || (ido == 1)) {

			memcpy((void *)(workd + ipntr[2 - 1] - 1), (void *) (workd + ipntr[1 - 1] - 1), N * sizeof(DOUBLE));
			dgetrs(&trans, &GETRN, &GETRNRHS, a, &GETRLDA, ipiv, workd + ipntr[2 - 1] -1, &GETRLDB, &INFO);
			if (INFO != 0) {
				printf("Error, INFO = %d. LAPACK GETRS error.", INFO);
			}
		}
	} while ((ido != 99) && (*info == 0)); // fix INFO and IDO conditions

	if (*info < 0) {
		*nconv = -1;
		dgeevas_error(info);
	} else {
		if (*info > 0) {
			dgeevas_error(info);
		}
		dneupd(&rvec, all, select, w, &w[*ncv], z, ldv, sigmar, sigmai, workev, bmat, n, which, nev, abstol,\
				resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, ierr);
		if (*ierr != 0) {
			*nconv = -1;
			dneupd_error(ierr);
		} else {
			*nconv = iparam[5 - 1];
		}
	}
}

void dgesvr(char *jobz, WHICH_TYPE *which, MKL_INT *nev, MKL_INT *n, double *a, \
		double *w, double *z, double *abstol, ARPACK_ERROR_TYPE *info, MKL_INT *nconv, \
		MKL_INT *ncv, MKL_INT *ldv, MKL_INT *maxiter, DOUBLE *v, DOUBLE *workl, \
		MKL_INT *lworkl, DOUBLE *workd, DOUBLE *workev, DOUBLE *d, DOUBLE *resid, MKL_INT *select, \
		DOUBLE *sigmar, DOUBLE *sigmai, MKL_INT *ipiv) {

	MKL_INT N = *n;

	char bmat[2] = "I";
	int iparam[11], ipntr[11];
	iparam[1 - 1] = 1; //exact shifts, use always
	iparam[3 - 1] = maxiter;
	iparam[7 - 1] = 3; //set to 1 for regular mode

	char all = 'A';
	char rvec = (*jobz == 'V');
	MKL_INT *ierr;

	CHAR trans = 'N';
	MKL_INT GETRSN = N;
	MKL_INT GETRSNRHS = 1;
	DOUBLE alpha = 1.0;
	MKL_INT GETRSLDA = N;
	MKL_INT GETRSLDB = N;
	DOUBLE beta = 0.0;
	MKL_INT INFO;

	MKL_INT iterN;
	for (iterN = 0; iterN < N; ++iterN) {
		a[iterN * N + iterN] = a[iterN * N + iterN] - *sigma;
	}
	dgetrf(&GETRM, &GETRN, a, &GETRLDA, ipiv, &INFO);
	if (INFO != 0) {
		printf("Error, INFO = %d. LAPACK GETRF error.", INFO);
	}

	int ido = 0;
	do {
		dnaupd(&ido, bmat, n, which, nev, abstol, resid, ncv, z, ldv, iparam, ipntr, workd, workl, lworkl, info);

		if ((ido == -1) || (ido == 1)) {

			memcpy((void *)(workd + ipntr[2 - 1] - 1), (void *) (workd + ipntr[1 - 1] - 1), N * sizeof(DOUBLE));
			dgetrs(&trans, &GETRN, &GETRNRHS, a, &GETRLDA, ipiv, workd + ipntr[2 - 1] -1, &GETRLDB, &INFO);
			if (INFO != 0) {
				printf("Error, INFO = %d. LAPACK GETRS error.", INFO);
			}
		}
	} while ((ido != 99) && (*info == 0)); // fix INFO and IDO conditions

	if (*info < 0) {
		*nconv = -1;
		dgeevas_error(info);
	} else {
		if (*info > 0) {
			dgeevas_error(info);
		}
		dneupd(&rvec, all, select, w, &w[*ncv], z, ldv, sigmar, sigmai, workev, bmat, n, which, nev, abstol,\
				resid, ncv, v, ldv, iparam, ipntr, workd, workl, lworkl, ierr);
		if (*ierr != 0) {
			*nconv = -1;
			dneupd_error(ierr);
		} else {
			*nconv = iparam[5 - 1];
		}
	}
}
