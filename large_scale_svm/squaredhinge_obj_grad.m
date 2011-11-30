function [f, df] = squaredhinge_obj_grad(para, X, Y, lambda, kernelMatrix, usebias, useregularizer)

% USAGE: [f, df] = squaredhinge_obj_grad(para, X, Y, lambda, sigma, kernelMatrix)
% compute the cost and gradient of linear SVM classifer with square-hinge loss
%
% Input:
%        para --- (d+1) by 1 vector, first d elements giving linear
%        function weights w and the rest one elements giving the bias term
%        b
%        X --- d x N matrix, N data points in a d-dim space
%        Y --- 1 x N matrix, labels of n data points -1 or 1
%        lambda --- a positive coefficient for regulerization, related to
%                   learning rate
%        kernelMatrix --- a d x d positive semi-definite matrix used as linear kernel
%		 usebias --- specify whether a bias term is used
%
% Output:
%        f --- the cost function at current w and b
%        df --- (d + 1) dim gradient vector
%       
% Kai Yu, kai.yu@siemens.com
% Nov. 2005

error(nargchk(4, 7, nargin));
if nargin < 7
    useregularizer = 1;
end
if nargin < 6
    usebias = 1;
end
if nargin < 5
    kernelMatrix = [];
end

X = X';
Y = Y';
[N, D] = size(X);

w = para(1:D);
b = para(D+1);

if (useregularizer == 1),
	if isempty(kernelMatrix),
		lambdawgamma = w * lambda;
	else
		lambdawgamma = kernelMatrix * w * lambda;
	end;
else
	lambdawgamma = zeros(size(w));
end;
if (isempty(kernelMatrix)),
	Ypred = X*w + b;
else
	Ypred = X * kernelMatrix * w + b;
end;
active_idx = find( Ypred.*Y < 1 );  
if isempty(active_idx)
    dw = 2*lambdawgamma;
    db = 0;
    active_E = 0;
else
    active_X = X(active_idx, :);
    active_Y = Y(active_idx, :);
    active_E = Ypred(active_idx)- active_Y;
	if (isempty(kernelMatrix)),
		dw = 2*(active_E'*active_X)' + 2*lambdawgamma;  
	else
		dw = 2*(active_E'*active_X * kernelMatrix )' + 2*lambdawgamma;  
	end;
	if (usebias == 1),
		db = 2*sum(active_E);
	elseif (usebias == 0)
		db = 0;
	end;
end
df = [dw; db];
f = active_E'*active_E + w'*lambdawgamma;
