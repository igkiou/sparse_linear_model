void cblas_dscal( int N, double alpha, double *X, int incX)
{
	dscal(&N,&alpha,X,&incX);
}

void cblas_dcopy(int N,double *X,int incX,double *Y,int incY)
{
	dcopy(&N,X,&incX,Y,&incY);
}

void cblas_dgemm(enum CBLAS_ORDER Order,enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
                 int M, int N, int K, double alpha, double *A, int lda,
                 double *B, int ldb, double beta, double *C, int ldc)
{
	char ta[1],tb[1];
	if (transA==111)
	{
		*ta='N';
	}
	else
	{
		*ta='T';
	};
	if (transB==111)
	{
		*tb='N';
	}
	else
	{
		*tb='T';
	};
	dgemm(ta,tb,&M,&N,&K,&alpha,A,&lda,B,&ldb,&beta,C,&ldc);
}

void cblas_dgemv(enum CBLAS_ORDER Order,enum CBLAS_TRANSPOSE transA,
                 int M, int N, double alpha, double *A, int lda,
                 double *B, int incB, double beta, double *C, int incC)
{
	char ta[1];
	if (transA==111)
	{
		*ta='N';
	}
	else
	{
		*ta='T';
	};
	dgemv(ta,&M,&N,&alpha,A,&lda,B,&incB,&beta,C,&incC);
}

void cblas_daxpy(int N,double alpha,double *X,int incX,double *Y,int incY)
{
	daxpy(&N,&alpha,X,&incX,Y,&incY);
}

void cblas_dger(enum CBLAS_ORDER Order,int m,int n,double alpha,double *x,int incx,double *y,int incy,double *A,int lda)
{
	dger(&m,&n,&alpha,x,&incx,y,&incy,A,&lda);
}

// Some useful functions ...
double doubsum(double *xmat, int n)
{
	int i;
	double res=0.0;
	for (i=0;i<n;i++){res+=xmat[i];};
	return res;
}

double doubdot(double *xvec, double *yvec, int n)
{
	int i;
	double res=0.0;
	for (i=0;i<n;i++){res+=xvec[i]*yvec[i];};
	return res;
}

int idxmax(double *xmat, int n)
{
	int i;
	int res=0;
	for (i=0;i<n;i++)
	{
		if (xmat[i]>xmat[res]) {res=i;}
	}
	return res;
}


double doubasum(double *xmat, int n)
{
	int i;
	double res=0.0;
	for (i=0;i<n;i++){res+=dabsf(xmat[i]);};
	return res;
}

double doubnorm2(double *xmat, int n)
{
	int i;
	double res=0.0;
	for (i=0;i<n;i++){res+=xmat[i]*xmat[i];};
	return sqrt(res);
}

double infnorm(double *xmat, int n)
{
	int i,j;
	double res=0.0,sum;
	
	for (j=0;j<n;j++){
		sum=0.0;
		for(i=0;i<n;i++)
			sum+=dabsf(xmat[j+i*n]);
		if(sum>=res) res=sum;
	}	
	return res;
}

double frobnorm(double *xmat, int n)
{
	int i,j;
	double res=0.0;
	
	for (i=0;i<n;i++)
		for (j=0;j<n;j++)
			res+=(xmat[i*n+j]*xmat[i*n+j]);
	
	return pow(res,.5);
}

double dsignf(double x)
{
	if (x>=0)
		return 1.0;
	else
		return -1.0;
}

double dminif(double x, double y)
{
	if (x>=y)
		return y;
	else
		return x;
}

double dmaxf(double x, double y)
{
	if (x>=y)
		return x;
	else
		return y;
}

int imaxf(int x, int y)
{
	if (x>=y)
		return x;
	else
		return y;
}

double dabsf(double x)
{
	if (x>=0)
		return x;
	else
		return -x;
}

void dispmat(double *xmat, int n, int m)
{
	int i,j;
	
	for (i=0; i<n; i++)
	{
		for (j=0;j<m;j++)
		{
			printf("%+.4f ",xmat[j*n+i]);
		}
		printf("\n");
	}
	printf("\n");
}

// do partial eig approximation of exp(bufmata)
// return fmu, and get dmax and numeigs from parameter references
double partial_eig(int n,int k,double mu,double eigcut,double *bufmata,
				   double *bufmatb,double *numeigs_matlab,double *evector_temp,
				   double *evector_store,double *eig,double *Dvec,double *gvec,
				   double *hvec,double *Vmat,double *Umat,double *workvec,int *count,
				   int addeigs, double perceigs,int check_for_more_eigs, int *arcount)
{
	int numeigs=(int)(*numeigs_matlab),nvls=0,h,i,incx=1,n2=n*n;
	int lwork,inflapack,indmax,check_other_eigs=0,neceigs=0;
	double alpha,beta,hs=0.0,dmax=0.0,fmu,buf,bufmata_shift=0.0;
	double *evector_index;
	char jobz[1],uplo[1];	
	double sum_sq_eigs=0.0; // variable for eigcut check, hs is the sum of the eigs
	double l2normbound,tol,minDvec;
	// Arpack parameters
	char which[2]="LA"; // Arpack: we want largest algebraic eigs...
	int ncv,info,nconv,nummatvec;
	int maxitr=500; 
	
	// TODO: find the optimal value for ncv
	if(numeigs<n-2 && (numeigs*1.0/n)<perceigs) {  
		// skip all this if we already know we want many eigs
		if (k==0 || numeigs>1 || (numeigs==1 && k%check_for_more_eigs==0)) check_other_eigs=1;
		bufmata_shift=frobnorm(bufmata,n);// Simple bound on largest magnitude eigenvalue
		tol=eigcut/bufmata_shift; // Tolerance parameter for eigs
		ncv=4*numeigs; // TODO: Should get a better size estimate here (cf. arapck manual)
		info=simarpack(bufmata,n,numeigs,ncv,tol,which,maxitr,1,eig,evector_temp,&nconv,&nummatvec,arcount);
		if (info!=0) {
			mexPrintf("DSPCA: Not all eigs converged in Arpack. iter=%d.\n",k);
			numeigs=n;} // Switch to full eig. decomposition.
		alpha=0.0;cblas_dscal(n2,alpha,evector_store,incx);
		evector_index=evector_store;  // Pointer to space for newest eigenvector
		cblas_dcopy(numeigs*n,evector_temp,incx,evector_index,incx); // TODO: simplify this, evector_temp not really required?		
		evector_index+=(numeigs*n);
		indmax=idxmax(eig,numeigs);dmax=eig[indmax];
		alpha=0.0;cblas_dscal(n,alpha,Dvec,incx);cblas_dscal(n,alpha,hvec,incx);
		nvls=numeigs;hs=0;minDvec=1.0;
		for(h=0;h<numeigs;h++) {  
			Dvec[h]=exp((eig[numeigs-1-h]-dmax)/mu);  // Dvec stores hvec, but in proper order.
			hvec[h]=exp((eig[h]-dmax)/mu);
			hs+=Dvec[h];
			sum_sq_eigs+=(Dvec[h]*Dvec[h]);
			minDvec=dminif(minDvec,Dvec[h]);
			l2normbound=(n-(h+1))*minDvec*pow(sum_sq_eigs,.5)/(hs*hs)+pow(n-(h+1),.5)*minDvec/hs;
			if (l2normbound<=eigcut && neceigs==0) neceigs=h+1;  // decreases number of eigenvalues if possible
		}
		for(h=0;h<numeigs;h++) { // Remove eigenvectors from matrix
			alpha=-eig[h];
			cblas_dger(CblasColMajor,n,n,alpha,evector_temp+h*n,incx,evector_temp+h*n,incx,bufmata,n);}
		while(check_other_eigs==1 && l2normbound>eigcut && nvls<n-2 && (nvls*1.0/n)<perceigs) {
			ncv=4*addeigs;	
			info=simarpack(bufmata,n,addeigs,ncv,tol,which,maxitr,0,eig,evector_temp,&nconv,&nummatvec,arcount);
			cblas_dcopy(addeigs*n,evector_temp,incx,evector_index,incx);		
			evector_index+=(addeigs*n);
			if (info!=0) {
				//mexPrintf("DSPCA: Not all eigs converged in Arpack.\n");
				numeigs=n;} // Switch to full eig. decomposition.
			for(h=0;h<addeigs;h++) {
				Dvec[nvls+h]=exp((eig[addeigs-1-h]-dmax)/mu);
				hvec[nvls+h]=exp((eig[h]-dmax)/mu);
				hs+=Dvec[nvls+h]; 
				sum_sq_eigs+=(Dvec[nvls+h]*Dvec[nvls+h]);
				minDvec=dminif(minDvec,Dvec[nvls+h]);
				l2normbound=(n-(nvls+h+1))*minDvec*pow(sum_sq_eigs,.5)/(hs*hs)+pow(n-(nvls+h+1),.5)*minDvec/hs;
				if (l2normbound<=eigcut && neceigs==0) neceigs=nvls+h+1;  // decreases number of eigenvalues if possible
			}
			for(h=0;h<addeigs;h++) { // Remove eigenvectors from matrix
				alpha=-eig[h];
				cblas_dger(CblasColMajor,n,n,alpha,evector_temp+h*n,incx,evector_temp+h*n,incx,bufmata,n);}	
			nvls+=addeigs;
		}
		numeigs=nvls;}
	if (numeigs>=n-2 || (numeigs*1.0/n)>=perceigs){ //just do a full eigenvalue decomposition if already done too many partial or if error in arpack
		*jobz='V';*uplo='U';lwork=3*n+n*n;
		dsyev(jobz,uplo,&n,Vmat,&n,Dvec,workvec,&lwork,&inflapack); // call LAPACK for full eig.
		// compute fmu(X) = mu*log(trace((exp(A+X)/mu)))-mu*log(n) reliably 
		indmax=idxmax(Dvec,n);dmax=Dvec[indmax];
		for (i=0;i<n;i++) {hvec[i]=exp((Dvec[i]-dmax)/mu);}
		buf=doubsum(hvec,n);
		fmu=dmax+mu*log(buf/n);
		// compute gradient of fmu w.r.t. X, which is the dual variable U 
		alpha=0.0;cblas_dscal(n2,alpha,bufmatb,incx);
		for (i=0;i<n;i++) {gvec[i]=hvec[i]/buf;bufmatb[i*n+i]=gvec[i];}
		alpha=1.0;beta=0.0;
		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,n,n,alpha,Vmat,n,bufmatb,n,beta,bufmata,n);
		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,n,n,n,alpha,bufmata,n,Vmat,n,beta,Umat,n);
		numeigs=n;neceigs=n;}
	else { // calculate fmu and Umat if partial eigenvalue approximation is satisfactory
		fmu=dmax+mu*log(hs)-mu*log(n);
		alpha=0.0;cblas_dscal(n2,alpha,Umat,incx);
		for(h=0;h<nvls;h++) { // Reconstruct U matrix from partial eig. decomposition
			alpha=hvec[h]/hs;
			cblas_dger(CblasColMajor,n,n,alpha,evector_store+h*n,incx,evector_store+h*n,incx,Umat,n);}}
	count[numeigs-1]++;
	if (info==0){*numeigs_matlab=neceigs;} // Return numeigs. 
	bufmata[0]=dmax;
	return fmu;
}


// Simple wrapper for calls to ARPACK 
int simarpack(double *Amat, int n, int nev, int ncv, double tol, char* which, int maxitr, int printlevel, double* evals, double* evecs, int* nconv, int* nummatvec, int *count)
{
	// From ARPACK examples. See dsvrv1.f, ... 
	double *v, *workl, *workd, *d, *resid;
	int *select, iparam[11], ipntr[11];
	char bmat[2]="I",all[4]="All"; // ARPACK params we only solve simple eigenvalue problems
	int ldv, ido, lworkl, info, ierr;
	int mode, ishfts, rvec,i,incx=1;
	double sigma,zero,minevalest;
	double alpha=1.0,beta=0.0;

	// Internal ARPACK parameters and vars
	ldv=n;zero=0.0;lworkl=ncv*(ncv+10);
	info=0;ido=0;ishfts =1;mode=1;rvec=1;
	iparam[1-1]=ishfts;iparam[3-1]=maxitr;iparam[7-1]=mode;
	*nummatvec=0;
	
	// Allocate memory for arpack workspace
	v = (double*)calloc(ldv*ncv, sizeof(double)); 
	workl = (double*)calloc(lworkl, sizeof(double));
	workd = (double*)calloc(3*n, sizeof(double));
	d = (double*)calloc(ncv*2, sizeof(double));
	resid = (double*)calloc(n, sizeof(double));
	select = (int*)calloc(ncv, sizeof(int));
		
	// ARPACK loop
	*count=0;
	do {
		(*count)++;
		dsaupd(&ido, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
		//if (iparam[3-1]>1000) printf("ARPACK dsaupd number iters: %d.\n", iparam[3-1]);
		if(ido == -1 || ido == 1) {
			// *** Matrix vector product, X is at workd+ipntr[1-1]-1 and the product Y is at workd+ipntr[2-1]-1 ***
			cblas_dgemv(CblasColMajor,CblasNoTrans,n,n,alpha,Amat,n,workd + ipntr[1-1] - 1,incx,beta,workd + ipntr[2-1] - 1,incx);
			*nummatvec = *nummatvec + 1;}
		else if(info != 0) {
			if(printlevel > 0) printf("ARPACK dsaupd error (info = %d).\n", info);}} 
	while(ido != 99);
	// Post processing
	dseupd(&rvec, all, select, d, v, &ldv, &sigma, bmat, &n, which, &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &ierr);
	if(ierr != 0) {
		if(printlevel > 0) printf("ARPACK dseupd error (ierr = %d)\n",ierr);
		info = ierr;}
	*nconv = iparam[5-1];
	if(printlevel > 0 && *nconv < nev) {
		printf("Warning: %d out of %d evals converged. dsaupd called %d times\n", *nconv, nev,*count);
		if(nev == 1) printf("Best estimate returned.\n");
		else printf("\n");}
	else{// Store eigensystem
		cblas_dcopy(nev,d,1,evals,1);
		cblas_dcopy(nev*n,v,1,evecs,1);
		if(nev == 1 && *nconv == 0) {
			minevalest = 1.0e10;
			for(i = 1; i <= ncv; i++) minevalest = -imaxf(-minevalest,-workl[ipntr[6-1]-2+i]);
			evals[0] = minevalest;}}

	// Free everything
	free(select);
	free(resid);
	free(d);
	free(workd);
	free(workl);
	free(v);
	// Return status
	if(*nconv < nev) return -1; // Not all eigenvalues were found
	return info;  // Return arapck info otherwise
}

// do partial eig approximation of exp(bufmata)
// return fmu, and get dmax and numeigs from parameter references
double partial_eig_matlab(int n,int k,double mu,double eigcut,double *bufmata,
				double *bufmatb,double *numeigs_matlab,double *evector_temp,
				double *evector_store,double *evalue,mxArray *input[4],mxArray *output[3],
				double *hvec,double *Vmat,double *Umat,double *workvec,int *count,
				double last_dmax,int addeigs, double perceigs,int check_for_more_eigs)
{
	int numeigs=(int)(*numeigs_matlab),nvls,h,i,incx=1,n2=n*n;
	int lwork,inflapack,indmax,check_other_eigs=0,neceigs=0;
	double alpha,beta,hs=0.0,dmax=1.0,fmu,buf,bufmata_shift=0.0,eigsflag;
	double *evector_index;
	char jobz[1],uplo[1],tolerance[100];	
	double sum_sq_eigs=0.0; // variable for eigcut check, hs is the sum of the eigs
	double l2normbound,minhvec;

	if(numeigs<n-2 && (numeigs*1.0/n)<perceigs) {  // skip all this if we already know we want many eigs
		if (k==0 || numeigs>1 || (numeigs==1 && k%check_for_more_eigs==0)) check_other_eigs=1;
		bufmata_shift=frobnorm(bufmata,n);// simple bound on largest magnitude eigenvalue
		sprintf(tolerance,"options.tol=%.15f;",eigcut/bufmata_shift); // this will be the tolerance parameter for eigs
		mexEvalString(tolerance);
		for (i=0;i<n;i++) bufmata[i*n+i]+=bufmata_shift;
		memcpy(mxGetPr(input[0]),bufmata,n*n*sizeof(double));
		memcpy(mxGetPr(input[1]),numeigs_matlab,sizeof(double));	
		mexCallMATLAB(3,output,4,input,"eigs");
		memcpy(evalue,mxGetPr(output[2]),sizeof(double));
		eigsflag=evalue[0];
		memcpy(evector_temp,mxGetPr(output[0]),numeigs*n*sizeof(double));
		alpha=0.0;cblas_dscal(n2,alpha,evector_store,incx);
		evector_index=evector_store;  // evector_index keeps track of place in evector_store to put newest eigenvector
		memcpy(evector_index,evector_temp,numeigs*n*sizeof(double));  // put the newest eigenvector in storage
		evector_index+=(numeigs*n);
		memcpy(evalue,mxGetPr(output[1]),numeigs*numeigs*sizeof(double));
		dmax=evalue[0];
		for (i=0;i<numeigs;i++) dmax=dmaxf(dmax,evalue[i*numeigs+i]); // get max eig from 1st call to eigs
		for(h=0;h<numeigs;h++) { // Remove eigenvectors from matrix
			alpha=-evalue[h*numeigs+h];
			cblas_dger(CblasColMajor,n,n,alpha,evector_temp+h*n,incx,evector_temp+h*n,incx,bufmata,n);
		}		
		alpha=0.0;cblas_dscal(n,alpha,hvec,incx);
		nvls=numeigs;
		hs=0;minhvec=1.0;
		for(h=0;h<numeigs;h++) {  
			hvec[h]=exp((evalue[h*numeigs+h]-dmax)/mu);
			hs+=hvec[h];
			sum_sq_eigs+=(hvec[h]*hvec[h]);
			minhvec=dminif(minhvec,hvec[h]);
			l2normbound=(n-(h+1))*minhvec*pow(sum_sq_eigs,.5)/(hs*hs)+pow(n-(h+1),.5)*minhvec/hs;
			if (l2normbound<=eigcut && neceigs==0) neceigs=h+1;  // decreases number of eigenvalues if possible
		}
		mxDestroyArray(output[0]); // free memory from MATLAB calls to eigs
		mxDestroyArray(output[1]);
		while(check_other_eigs==1 && l2normbound>eigcut && nvls<n-2 && (nvls*1.0/n)<perceigs) {
			symmetrize(bufmata,bufmatb,n);	// symmetrize A+X so no precision problems		
			cblas_dcopy(n2,bufmatb,incx,bufmata,incx);		
			memcpy(mxGetPr(input[0]),bufmata,n*n*sizeof(double));
			*numeigs_matlab=1.0*addeigs;	
			memcpy(mxGetPr(input[1]),numeigs_matlab,sizeof(double));	
			mexCallMATLAB(3,output,4,input,"eigs");
			if (eigsflag<0.5){
				memcpy(evalue,mxGetPr(output[2]),sizeof(double));
				eigsflag=evalue[0];
			}	
			memcpy(evector_temp,mxGetPr(output[0]),addeigs*n*sizeof(double));
			memcpy(evector_index,evector_temp,addeigs*n*sizeof(double));
			evector_index+=(addeigs*n);
			memcpy(evalue,mxGetPr(output[1]),addeigs*addeigs*sizeof(double));
			for(h=0;h<addeigs;h++) { // Remove eigenvectors from matrix
				alpha=-evalue[h*addeigs+h];
				cblas_dger(CblasColMajor,n,n,alpha,evector_temp+h*n,incx,evector_temp+h*n,incx,bufmata,n);
			}
			for(h=0;h<addeigs;h++) {
				hvec[nvls+h]=exp((evalue[h*addeigs+h]-dmax)/mu);
				hs+=hvec[nvls+h];
				sum_sq_eigs+=(hvec[nvls+h]*hvec[nvls+h]);
				minhvec=dminif(minhvec,hvec[nvls+h]);
				l2normbound=(n-(nvls+h+1))*minhvec*pow(sum_sq_eigs,.5)/(hs*hs)+pow(n-(nvls+h+1),.5)*minhvec/hs;
				if (l2normbound<=eigcut && neceigs==0) neceigs=nvls+h+1;  // decreases number of eigenvalues if possible
			}
			nvls+=addeigs;
			mxDestroyArray(output[0]); // free memory from MATLAB calls to eigs
			mxDestroyArray(output[1]);
			l2normbound=(n-nvls)*minhvec*pow(sum_sq_eigs,.5)/(hs*hs)+pow(n-nvls,.5)*minhvec/hs;
		}
		numeigs=nvls;
	}
	if (numeigs>=n-2 || (numeigs*1.0/n)>=perceigs){ //just do a full eigenvalue decomposition if already done many partial
		*jobz='V';*uplo='U';lwork=3*n+n*n;
		dsyev(jobz,uplo,&n,Vmat,&n,hvec,workvec,&lwork,&inflapack); // call LAPACK (most CPU time is here) (TODO: compute optimal lwork)
		// compute fmu(X) = mu*log(trace((exp(A+X)/mu)))-mu*log(n) reliably 
		indmax=idxmax(hvec,n);dmax=hvec[indmax];
		for (i=0;i<n;i++) {hvec[i]=exp((hvec[i]-dmax)/mu);}
		buf=doubsum(hvec,n);
		fmu=dmax+mu*log(buf/n);
		// compute gradient of fmu w.r.t. X, which is the dual variable U 
		alpha=0.0;cblas_dscal(n2,alpha,bufmatb,incx);
		for (i=0;i<n;i++) bufmatb[i*n+i]=hvec[i]/buf;
		alpha=1.0;beta=0.0;
		cblas_dgemm( CblasColMajor, CblasNoTrans, CblasNoTrans,n,n,n,alpha,Vmat,n,bufmatb,n,beta,bufmata,n);
		cblas_dgemm( CblasColMajor, CblasNoTrans, CblasTrans,n,n,n,alpha,bufmata,n,Vmat,n,beta,Umat,n);
		numeigs=n;neceigs=n;		
	} else { // calculate fmu and Umat if partial eigenvalue approximation is satisfactory
		fmu=(dmax-bufmata_shift)+mu*log(hs)-mu*log(n);
		alpha=0.0;cblas_dscal(n2,alpha,Umat,incx);
		for(h=0;h<nvls;h++) { // Reconstruct U matrix from partial eig. decomposition
			alpha=hvec[h]/hs;
			cblas_dger(CblasColMajor,n,n,alpha,evector_store+h*n,incx,evector_store+h*n,incx,Umat,n);
		}
	}
	if(eigsflag<0.5) *numeigs_matlab=neceigs; // to return from function
	
	if(numeigs<n) bufmata[0]=dmax-bufmata_shift; // to return from function
	else bufmata[0]=dmax;
	bufmata[1]=eigsflag;
	if (eigsflag>0.50) {
		mexPrintf("ERROR: Not all eigenvalues converged in calling Matlab eigs function in partial eigenvalue approximation.  Perhaps maxit not set high enough\n");
	}
	count[numeigs-1]++;
	return fmu;
}

// symmetrize a matrix xmat and return in matrix ymat
void symmetrize(double *xmat,double *ymat,int n)
{
	int i,j,incx=1,n2=n*n;
	double alpha=0.0;

	cblas_dscal(n2,alpha,ymat,incx);	
	for(i=0;i<n;i++)
		for(j=0;j<n;j++)
			ymat[n*i+j]=(xmat[n*i+j]+xmat[n*j+i])/2;	
}
