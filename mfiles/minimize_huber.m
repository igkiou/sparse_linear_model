% MINIMIZE_HUBER Uses conjugate gradients to minimize the multitask
% huberized hinge loss with respect to Phi.
%
%	optPhi = minimize_huber(initPhi, length X, Y, weights, bias) returns
%	the value of the projection matrix Phi by optimizing the multitask
%	huberized hinge loss using conjugate gradients.
%
%	Inputs:
%	initPhi	- projection matrix used for initialization of the conjugate
%			  gradient optimization algorithm, M x N matrix, where N is the
%			  original signal dimension and M is the number of projections
%			  (reduced dimension) (may also be provided as an M * N x 1
%			  vector).
%	length	- number of line searches before the conjugate gradient
%			  algorithm terminates.
%	X		- original data, N x P matrix, where P is the number of samples
%			  (each column is a signal).
%	Y		- labels for the classification task, P x T matrix, where P is
%			  the number of samples and T is the number of classification
%			  tasks. Labels must be -1 and +1 for negative and positive,
%			  respectively (each column contains the P labels for the
%			  respective classification task). 
%	weights - weights used by the linear SVM, M x T matrix (each column
%			  contains the M weights for the linear SVM used for the
%			  respective classification task).
%	bias	- bias terms used by the linear SVM, 1 x T or T x 1 matrix
%			  (each element is the bias term for the linear SVM used for
%			  the respective classification task).
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
