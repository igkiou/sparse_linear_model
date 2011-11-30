function [w, b] = li2nsvm_lbfgs(X, Y, lambda, kernelMatrix, lossfun, usebias)

% USAGE: [w, b] = li2nsvm_lbfgs(X, Y, lambda, sigma, kernelMatrix,lossfun)
% linear SVM classifer with square- or huberized-hinge loss, trained via
% lbfgs.
%
% Input:
%        X --- d x N matrix, N data points in a d-dim space
%        Y --- 1 x N matrix, labels of n data points -1 or 1
%        lambda --- a positive coefficient for regulerization, related to
%                   learning rate
%        kernelMatrix --- a d x d positive semidefinite matrix used for a
%					linear kernel.
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

error(nargchk(3, 6, nargin));
if nargin < 6
    usebias = 1;
end
if nargin < 5
    lossfun = 'square';
end
if nargin < 4
    kernelMatrix = [];
end

[D, N] = size(X); 

w0 = zeros(D,1);
b0 = 0;

wolfe = struct('a1',0.5,'a0',0.01,'c1',0.0001,'c2',0.9,'maxiter',10,'amax',1.1);
lbfgs_options = struct('maxiter', 30, ...
                       'termination', 1.0000e-004, ...
                       'xtermination', 1.0000e-004, ...
                       'm', 10, ...
                       'wolfe', wolfe, ...
                       'echo', false);
if (strcmp(lossfun, 'square')),
% 	gradfun = 'squaredhinge_obj_grad';
	gradfun = 'squaredhinge_obj_grad_mex';
elseif (strcmp(lossfun, 'huber')),
% 	gradfun = 'huberhinge_obj_grad';
	gradfun = 'huberhinge_obj_grad_mex';
else
	error('Invalid loss function specification.');
end;
          
% [retval, xstarbest, xstarfinal, history] = lbfgs2([w0; b0], lbfgs_options, gradfun, [], X, Y, lambda, kernelMatrix, usebias);
[retval, xstarbest, xstarfinal, history] = lbfgs2([w0; b0], lbfgs_options, gradfun, [], full(X), Y, lambda, kernelMatrix, usebias);

w = xstarbest(1:D);
b = xstarbest(D+1);



