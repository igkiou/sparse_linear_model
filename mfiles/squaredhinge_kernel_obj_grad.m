function [f, df] = squaredhinge_kernel_obj_grad(kernelMatrix, para, X, Y, lambda, useregularizer)

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


error(nargchk(5, 6, nargin));
if nargin < 6,
    useregularizer = 1;
end

kernelMatrix = reshape(kernelMatrix, [size(X, 1) size(X, 1)]);

X = X';
Y = Y';
D = size(X, 2);

w = para(1:D);

if (useregularizer == 1),
	lambdawgamma = w * w' * lambda;
else
	lambdawgamma = zeros(size(kernelMatrix));
end;

Ypred = X * kernelMatrix * w;
active_idx = find( Ypred.*Y < 1 );  

if isempty(active_idx)
    dK = lambdawgamma;
    active_E = 0;
else
    active_X = X(active_idx, :);
    active_Y = Y(active_idx, :);
    active_E = Ypred(active_idx)- active_Y;
	dK = w * (2 * active_E' * active_X) / 2 + (2 * active_E' * active_X)' * w' / 2 + lambdawgamma;  
end
df = dK(:);
if (useregularizer == 1),
	f = lambda * w' * kernelMatrix * w;
else
	f = 0;
end;

if ~isempty(active_E),
	f = f + active_E' * active_E;
end;
