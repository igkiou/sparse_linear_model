%MINIMIZE_MEX Uses conjugate gradients to minimize the eigenvalue form
% of Grammian-based objective function with respect to Phi. 
%
%	optPhi = minimize_huber(initPhi, length DDt2, DDt3, VL, L) returns
%	the value of the projection matrix Phi by optimizing the eigenvalue
%	form of Grammian-based objective function using conjugate gradients.
%
%	Inputs:
%	initPhi	- projection matrix used for initialization of the conjugate
%			  gradient optimization algorithm,M x N matrix, where N is the
%			  original signal dimension and M is the number of projections
%			  (reduced dimension). 
%	length	- number of line searches before the conjugate gradient
%			  algorithm terminates.
%	DDt2	- N x N matrix equal to (D*D')^2, where D is the dictionary
%			  used for the encoding. D is an N x K matrix, where K is the
%			  number of atoms in the dictionary (each column contains one
%			  N x 1 atom).
%	DDt2	- N x N matrix equal to (D*D')^3.
%	VL		- N x N matrix equal to V * L, where [V L] = eig(D*D').
%	L		- N x 1 or 1 x N matrix equal to diag(L).
%
%	Outputs:
%	optPhi	- optimal projection matrix as returned by the conjugate 
%			  gradient optimization algorithm, M * N x 1 vector (i.e. it is
%			  returned in vectorized form).
%
%	This is an adapted mex implementation of Carl Rasmussen's minimize.m
%	optimization routine based on conjugate gradients. Line searches are
%	approximated using polynomial interpolation with Wolfe-Powel
%	conditions.  
