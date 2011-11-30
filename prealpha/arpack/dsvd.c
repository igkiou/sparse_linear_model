void dsvd() {
	INT maxm;
	INT maxn;
	INT maxnev;
	INT maxncv;
	INT ldv;
	INT ldu;
	
      integer          
      parameter       (maxm = 500, maxn=250, maxnev=10, maxncv=25, 
     &                 ldu = maxm, ldv=maxn )
//
//%--------------%
//| Local Arrays |
//%--------------%
//
      Double precision
     &                 v(ldv,maxncv), u(ldu, maxnev), 
     &                 workl(maxncv*(maxncv+8)), workd(3*maxn), 
     &                 s(maxncv,2), resid(maxn), ax(maxm)
      logical          select(maxncv)
      integer          iparam(11), ipntr(11)
//
//%---------------%
//| Local Scalars |
//%---------------%
//
      character        bmat*1, which*2
      integer          ido, m, n, nev, ncv, lworkl, info, ierr,
     &                 j, ishfts, maxitr, mode1, nconv
      logical          rve//
      Double precision      
     &                 tol, sigma, temp
//
//%------------%
//| Parameters |
//%------------%
//
      Double precision
     &                 one, zero
      parameter        (one = 1.0D+0, zero = 0.0D+0)
//
//%-----------------------------%
//| BLAS & LAPACK routines used |
//%-----------------------------%
//
      Double precision           
     &                 dnrm2
      external         dnrm2, daxpy, dcopy, dscal
//
//%-----------------------%
//| Executable Statements |
//%-----------------------%
//
//%-------------------------------------------------%
//| The following include statement and assignments |
//| initiate trace output from the internal         |
//| actions of ARPACK.  See debug.doc in the        |
//| DOCUMENTS directory for usage.  Initially, the  |
//| most useful information will be a breakdown of  |
//| time spent in the various stages of computation |
//| given by setting msaupd = 1.                    |
//%-------------------------------------------------%
//
      include 'debug.h'
      ndigit = -3
      logfil = 6
      msgets = 0
      msaitr = 0 
      msapps = 0
      msaupd = 1
      msaup2 = 0
      mseigt = 0
      mseupd = 0
//
//%-------------------------------------------------%
//| The following sets dimensions for this problem. |
//%-------------------------------------------------%
//
      m = 500
      n = 100
//
//%------------------------------------------------%
//| Specifications for ARPACK usage are set        | 
//| below:                                         |
//|                                                |
//|    1) NEV = 4 asks for 4 singular values to be |  
//|       computed.                                | 
//|                                                |
//|    2) NCV = 20 sets the length of the Arnoldi  |
//|       factorization                            |
//|                                                |
//|    3) This is a standard problem               |
//|         (indicated by bmat  = 'I')             |
//|                                                |
//|    4) Ask for the NEV singular values of       |
//|       largest magnitude                        |
//|         (indicated by which = 'LM')            |
//|       See documentation in DSAUPD for the      |
//|       other options SM, BE.                    | 
//|                                                |
//| Note: NEV and NCV must satisfy the following   |
//|       conditions:                              |
//|                 NEV <= MAXNEV,                 |
//|             NEV + 1 <= NCV <= MAXNCV           |
//%------------------------------------------------%
//
      nev   = 4
      ncv   = 10 
      bmat  = 'I'
      which = 'LM'
//
      if ( n .gt. maxn ) then
         print *, ' ERROR with _SVD: N is greater than MAXN '
         go to 9000
      else if ( m .gt. maxm ) then
         print *, ' ERROR with _SVD: M is greater than MAXM '
         go to 9000
      else if ( nev .gt. maxnev ) then
         print *, ' ERROR with _SVD: NEV is greater than MAXNEV '
         go to 9000
      else if ( ncv .gt. maxncv ) then
         print *, ' ERROR with _SVD: NCV is greater than MAXNCV '
         go to 9000
      end if
//
//%-----------------------------------------------------%
//| Specification of stopping rules and initial         |
//| conditions before calling DSAUPD                    |
//|                                                     |
//|           abs(sigmaC - sigmaT) < TOL*abs(sigmaC)    |
//|               computed   true                       |
//|                                                     |
//|      If TOL .le. 0,  then TOL <- macheps            |
//|              (machine precision) is used.           |
//|                                                     |
//| IDO  is the REVERSE COMMUNICATION parameter         |
//|      used to specify actions to be taken on return  |
//|      from DSAUPD. (See usage below.)                |
//|                                                     |
//|      It MUST initially be set to 0 before the first |
//|      call to DSAUPD.                                | 
//|                                                     |
//| INFO on entry specifies starting vector information |
//|      and on return indicates error codes            |
//|                                                     |
//|      Initially, setting INFO=0 indicates that a     | 
//|      random starting vector is requested to         |
//|      start the ARNOLDI iteration.  Setting INFO to  |
//|      a nonzero value on the initial call is used    |
//|      if you want to specify your own starting       |
//|      vector (This vector must be placed in RESID.)  | 
//|                                                     |
//| The work array WORKL is used in DSAUPD as           | 
//| workspace.  Its dimension LWORKL is set as          |
//| illustrated below.                                  |
//%-----------------------------------------------------%
//
      lworkl = ncv*(ncv+8)
      tol = zero 
      info = 0
      ido = 0
//
//%---------------------------------------------------%
//| Specification of Algorithm Mode:                  |
//|                                                   |
//| This program uses the exact shift strategy        |
//| (indicated by setting IPARAM(1) = 1.)             |
//| IPARAM(3) specifies the maximum number of Arnoldi |
//| iterations allowed.  Mode 1 of DSAUPD is used     |
//| (IPARAM(7) = 1). All these options can be changed |
//| by the user. For details see the documentation in |
//| DSAUPD.                                           |
//%---------------------------------------------------%
//
      ishfts = 1
      maxitr = n
      mode1 = 1
//
      iparam(1) = ishfts
//           
      iparam(3) = maxitr
//             
      iparam(7) = mode1
//
//%------------------------------------------------%
//| M A I N   L O O P (Reverse communication loop) |
//%------------------------------------------------%
//
 10   continue
//
//   %---------------------------------------------%
//   | Repeatedly call the routine DSAUPD and take | 
//   | actions indicated by parameter IDO until    |
//   | either convergence is indicated or maxitr   |
//   | has been exceeded.                          |
//   %---------------------------------------------%
//
         call dsaupd ( ido, bmat, n, which, nev, tol, resid, 
     &                 ncv, v, ldv, iparam, ipntr, workd, workl,
     &                 lworkl, info )
//
         if (ido .eq. -1 .or. ido .eq. 1) then
//
//      %---------------------------------------%
//      | Perform matrix vector multiplications |
//      |              w <--- A*x       (av())  |
//      |              y <--- A'*w      (atv()) |
//      | The user should supply his/her own    |
//      | matrix vector multiplication routines |
//      | here that takes workd(ipntr(1)) as    |
//      | the input, and returns the result in  |
//      | workd(ipntr(2)).                      |
//      %---------------------------------------%
//
            call av (m, n, workd(ipntr(1)), ax) 
            call atv (m, n, ax, workd(ipntr(2)))
//
//      %-----------------------------------------%
//      | L O O P   B A C K to call DSAUPD again. |
//      %-----------------------------------------%
//
            go to 10
//
         end if 
//
//%----------------------------------------%
//| Either we have convergence or there is |
//| an error.                              |
//%----------------------------------------%
//
      if ( info .lt. 0 ) then
//
//   %--------------------------%
//   | Error message. Check the |
//   | documentation in DSAUPD. |
//   %--------------------------%
//
         print *, ' '
         print *, ' Error with _saupd, info = ', info
         print *, ' Check documentation in _saupd '
         print *, ' '
//
      else 
//
//   %--------------------------------------------%
//   | No fatal errors occurred.                  |
//   | Post-Process using DSEUPD.                 |
//   |                                            |
//   | Computed singular values may be extracted. |  
//   |                                            |
//   | Singular vectors may also be computed now  |
//   | if desired.  (indicated by rvec = .true.)  | 
//   |                                            |
//   | The routine DSEUPD now called to do this   |
//   | post processing                            | 
//   %--------------------------------------------%
//      
         rvec = .true.
//
         call dseupd ( rvec, 'All', select, s, v, ldv, sigma, 
     &        bmat, n, which, nev, tol, resid, ncv, v, ldv, 
     &        iparam, ipntr, workd, workl, lworkl, ierr )
//
//   %-----------------------------------------------%
//   | Singular values are returned in the first     |
//   | column of the two dimensional array S         |
//   | and the corresponding right singular vectors  | 
//   | are returned in the first NEV columns of the  |
//   | two dimensional array V as requested here.    |
//   %-----------------------------------------------%
//
         if ( ierr .ne. 0) then
//
//      %------------------------------------%
//      | Error condition:                   |
//      | Check the documentation of DSEUPD. |
//      %------------------------------------%
//
            print *, ' '
            print *, ' Error with _seupd, info = ', ierr
            print *, ' Check the documentation of _seupd. '
            print *, ' '
//
         else
//
            nconv =  iparam(5)
            do 20 j=1, nconv
//
               s(j,1) = sqrt(s(j,1))
//
//         %-----------------------------%
//         | Compute the left singular   |
//         | vectors from the formula    |
//         |                             |
//         |     u = Av/sigma            |
//         |                             |
//         | u should have norm 1 so     |
//         | divide by norm(Av) instead. |
//         %-----------------------------%
//
               call av(m, n, v(1,j), ax)
               call dcopy(m, ax, 1, u(1,j), 1)
               temp = one/dnrm2(m, u(1,j), 1)
               call dscal(m, temp, u(1,j), 1)
//
//         %---------------------------%
//         |                           |
//         | Compute the residual norm |
//         |                           |
//         |   ||  A*v - sigma*u ||    |
//         |                           |
//         | for the NCONV accurately  |
//         | computed singular values  |
//         | and vectors.  (iparam(5)  |
//         | indicates how many are    |
//         | accurate to the requested |
//         | tolerance).               |
//         | Store the result in 2nd   |
//         | column of array S.        |
//         %---------------------------%
//
               call daxpy(m, -s(j,1), u(1,j), 1, ax, 1)
               s(j,2) = dnrm2(m, ax, 1)
//
 20         continue
//
//      %-------------------------------%
//      | Display computed residuals    |
//      %-------------------------------%
//
            call dmout(6, nconv, 2, s, maxncv, -6,
     &                'Singular values and direct residuals')
         end if
//
//   %------------------------------------------%
//   | Print additional convergence information |
//   %------------------------------------------%
//
         if ( info .eq. 1) then
            print *, ' '
            print *, ' Maximum number of iterations reached.'
            print *, ' '
         else if ( info .eq. 3) then
            print *, ' ' 
            print *, ' No shifts could be applied during implicit',
     &               ' Arnoldi update, try increasing NCV.'
            print *, ' '
         end if      
//
         print *, ' '
         print *, ' _SVD '
         print *, ' ==== '
         print *, ' '
         print *, ' Size of the matrix is ', n
         print *, ' The number of Ritz values requested is ', nev
         print *, ' The number of Arnoldi vectors generated',
     &            ' (NCV) is ', ncv
         print *, ' What portion of the spectrum: ', which
         print *, ' The number of converged Ritz values is ', 
     &              nconv 
         print *, ' The number of Implicit Arnoldi update',
     &            ' iterations taken is ', iparam(3)
         print *, ' The number of OP*x is ', iparam(9)
         print *, ' The convergence criterion is ', tol
         print *, ' '
//
      end if
//
//%-------------------------%
//| Done with program dsvd. |
//%-------------------------%
//
 9000 continue
//
      end
c 
c ------------------------------------------------------------------
//matrix vector subroutines
//
//The matrix A is derived from the simplest finite difference 
//discretization of the integral operator 
//
//                f(s) = integral(K(s,t)x(t)dt).
// 
//Thus, the matrix A is a discretization of the 2-dimensional kernel 
//K(s,t)dt, where
//
//            K(s,t) =  s(t-1)   if 0 .le. s .le. t .le. 1,
//                      t(s-1)   if 0 .le. t .lt. s .le. 1.
//
//Thus A is an m by n matrix with entries
//
//            A(i,j) = k*(si)*(tj - 1)  if i .le. j,
//                     k*(tj)*(si - 1)  if i .gt. j
//
//where si = i/(m+1)  and  tj = j/(n+1)  and k = 1/(n+1).
// 
c-------------------------------------------------------------------
//
      subroutine av (m, n, x, w)
//
//computes  w <- A*x
//
      integer          m, n, i, j
      Double precision
     &                 x(n), w(m), one, zero, h, k, s, t
      parameter        ( one = 1.0D+0, zero = 0.0D+0 ) 
//
      h = one / dble(m+1)
      k = one / dble(n+1)
      do 5 i = 1,m
         w(i) = zero
 5    continue
      t = zero
// 
      do 30 j = 1,n
         t = t+k
         s = zero
         do 10 i = 1,j
           s = s+h
           w(i) = w(i) + k*s*(t-one)*x(j)
 10      continue 
         do 20 i = j+1,m
           s = s+h
           w(i) = w(i) + k*t*(s-one)*x(j) 
 20      continue
 30   continue      
//
      return
      end
//
c-------------------------------------------------------------------
//
      subroutine atv (m, n, w, y)
//
//computes  y <- A'*w
//
      integer         m, n, i, j
      Double precision
     &                w(m), y(n), one, zero,  h, k, s, t
      parameter       ( one = 1.0D+0, zero = 0.0D+0 )
//
      h = one / dble(m+1)
      k = one / dble(n+1)
      do 5 i = 1,n
         y(i) = zero
 5    continue
      t = zero
//
      do 30 j = 1,n
         t = t+k
         s = zero
         do 10 i = 1,j
           s = s+h
           y(j) = y(j) + k*s*(t-one)*w(i)
 10      continue
         do 20 i = j+1,m
           s = s+h
           y(j) = y(j) + k*t*(s-one)*w(i)
 20      continue
 30   continue
//
      return
      end 
//
