function [w, b] = li2nsvm_conj(X, Y, lambda, sigma, gamma, lossfun, usebias)

% USAGE: [w, b] = li2nsvm_lbfgs(X, Y, lambda, sigma, gamma,lossfun)
% linear SVM classifer with square- or huberized-hinge loss, trained via
% conjugate gradient.
%
% Input:
%        X --- N x d matrix, n data points in a d-dim space
%        Y --- N x 1 matrix, labels of n data points -1 or 1
%        lambda --- a positive coefficient for regulerization, related to
%                   learning rate
%        sigma --- a vector of weights of training examples. Its summation
%                  will be ensured to be N in the problem. Default setting is I. 
%                  This parameter enables to deal with classification with 
%                  unbalanced data. 
%        gamma --- a d x 1 weight vector of positive elements,
%                  used for modifying the regularization term into
%                  w'diag(gamma)w.
%        lossfun --- a string specifying the loss function used, 'square'
%                    for squared hinge loss, and 'huber' for huberized
%                    hinge loss (default 'square').
%		 usebias --- specify whether a bias term is used
%
% Output:
%        w --- d x 1 matrix, each column is a function vector
%        b --- the estimated bias term
%       
% Kai Yu, Aug. 2008

error(nargchk(3, 7, nargin));
if nargin < 7
    usebias = 1;
end
if nargin < 6
    lossfun = 'square';
end
if nargin < 5
    gamma = [];
end
if nargin < 4
    sigma  = [];
end

[N, D] = size(X); 

w0 = zeros(D,1);
b0 = 0;

% minimize options
verbose = 0;
numSearches = 100;

if (strcmp(lossfun, 'square')),
	gradfun = 'squaredhinge_obj_grad_mex';
elseif (strcmp(lossfun, 'huber')),
	gradfun = 'huberhinge_obj_grad_mex';
else
	error('Invalid loss function specification.');
end;

xstarbest = minimize([w0; b0], gradfun, numSearches, verbose, X', Y, lambda, usebias);
% [retval, xstarbest, xstarfinal, history] = lbfgs2([w0; b0], lbfgs_options, gradfun, [], X, Y, lambda, sigma, gamma, usebias);

w = xstarbest(1:D);
b = xstarbest(D+1);



