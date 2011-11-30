% L1QP_FEATURESIGN_MEX Sparse coding using the feature-sign algorithm.
%
%	S = l1qp_featuresign_mex(X, D, lambda, beta) uses l1-regularization to
%	calculate the sparse codes of the signals X with respect to the
%	dictionary D.
%
%	Inputs:
%	X		- original signals, N x P matrix, where N is the signal
%			  dimension and P is the number of samples (each column is a
%			  N x 1 signal).
%	D		- dictionary used for the encoding, N x K matrix, where K is
%			  the number of atoms in the dictionary (each column contains
%			  one N x 1 atom).
%	lambda	- regularization parameter, scalar.
%	bias	- regularization parameter, scalar, used to stabilize the
%			  results of the unconstrained quadratic programming problem
%			  solved internally (optional, default value 10^-4, set to
%			  similarly small value).
%
%	Outputs:
%	S		- sparse codes calculated, K x P matrix, (each column is the
%			  K x 1 code of the corresponding signal). Codes are calculated
%			  by minimizing the criterion
%
%			  ||X - D * S||_fro^2 + lambda * sum(i=1:P) ||S(:,i)||_1
%
%			  using the feature-sign algorithm.
