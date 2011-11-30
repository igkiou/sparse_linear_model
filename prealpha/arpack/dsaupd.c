/*
  This is an example program using the header file
  dsaupd.h, my implementation of the ARPACK sparse matrix
  solver routine.  See that header file for more
  documentation on its use.
  
  For this example, we will work with the four by four
  matrix H[i,j] = i^j + j^i.  The output of this program
  should be:

  Eigenvalue  0: 0.151995
  Eigenvector 0: (0.610828, -0.72048, 0.324065, -0.0527271)
  Eigenvalue  1: 1.80498
  Eigenvector 1: (0.7614, 0.42123, -0.481884, 0.103072)
  Eigenvalue  2: 17.6352
  Eigenvector 2: (-0.216885, -0.547084, -0.764881, 0.26195)
  
  Scot Shaw
  7 September 1999
*/

// Begin with some standard include files.

#include <math.h>
#include <stdio.h>
#include <fstream.h>

/*
  To use the routines from dsaupd.h, we need to define our
  own matrix multiplication routine.  Here I show one method
  that would exploit sparsity by storing only the non-zero
  matrix elements (though for this example we won't actually
  have a sparse matrix to work with).

  There is a routine for the multiplication, and a global
  variable T that will hold the non-zero matrix elements
  and their indicies.  The global integer tlen will hold the
  number of non-zero elements.  The matrix multiplication function
  needs to have the form below, and must be defined before
  the file dsaupd.h is included.
*/

void av(int n, double *in, double *out);
#include "dsaupd.h"

double **T;
int tlen;

void main()
{
  int n, nev, i, j;
  double **Evecs, *Evals;
  
  n = 4; // The order of the matrix

  /*
    Here we generate the matrix T for the multiplication.
    It helps if we know the number of non-zero matrix
    elements before we find them, so that we can save on
    storage space.  If not, generate enough storage space
    for T to cover an upper limit.
  */

  tlen = n*n;
  T = new double*[tlen];
  for (i=0; i<tlen; i++) T[i] = new double[3];

  tlen = 0;
  for (i=0; i<n; i++) for (j=0; j<n; j++) {
    T[tlen][0] = i;
    T[tlen][1] = j;
    T[tlen][2] = pow(i+1, j+1) + pow(j+1, i+1);
    tlen++;
  }

  /*
    We will calculate both the eigenvalues and the
    eigenvectors in this example.  To not calculate the
    vectors, just leave that argument out of the call to
    dsaupd.

    We will find only three of the eigenvalues, to make
    it more clear how the eigenvectors come out of the
    routine.  Besides that, the ARPACK routine won't
    calculate all of the eigenvalues.  I'm not sure if
    that is intentional, or if I am missing something.
  */

  nev = 3; // The number of values to calculate

  Evals = new double[nev];
  Evecs = new double*[n];
  for (i=0; i<n; i++) Evecs[i] = new double[nev];

  dsaupd(n, nev, Evals, Evecs);

  // Now we output the results.
  
  for (i=0; i<nev; i++) {
    cout << "Eigenvalue  " << i << ": " << Evals[i] << "\n";
      cout << "Eigenvector " << i << ": ("
      << Evecs[0][i] << ", "
      << Evecs[1][i] << ", "
      << Evecs[2][i] << ", "
      << Evecs[3][i] << ")\n";
  }
}


void av(int n, double *in, double *out)
{
  int i, j;

  for (i=0; i<n; i++) out[i] = 0;
  for (i=0; i<tlen; i++) out[(int)T[i][0]] += in[(int)T[i][1]] * T[i][2];
}




