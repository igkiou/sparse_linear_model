function [f, df] = huberhinge_kernel_obj_grad(kernelMatrix, para, X, Y, lambda, useregularizer)

% USAGE: [f, df] = huberhinge_obj_grad(para, X, Y, lambda, sigma, kernelMatrix)
% compute the cost and gradient of linear SVM classifer with huberized
% hinge loss. The huberized hinge loss is defined as
% 
%        -4*y*f(x),             if y*f(x)<-1
%        (max(0,1-y*f(x)))^2,   if y*f(x)>-1
%
% Input:
%        para --- (d+1) by 1 vector, first d elements giving linear
%        function weights w and the rest one elements giving the bias term
%        b
%        X --- d x N matrix, N data points in a d-dim space
%        Y --- 1 x N matrix, labels of n data points -1 or 1
%        lambda --- a positive coefficient for regularization, related to
%                   learning rate
%        kernelMatrix --- a d x d positive semi-definite matrix used as linear kernel
%		 usebias --- specify whether a bias term is used
%
% Output:
%        f --- the cost function at current w and b
%        df --- (d + 1) dim gradient vector

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
Yprod = Ypred .* Y;
absYprod = abs(Yprod);
if isempty(find(Yprod < 1, 1))
    dK = lambdawgamma;
    active_quad_E = 0;
    active_lin_E = 0;
	active_lin_E_sec = 0;
else
	active_lin_idx = find(Yprod < -1);
% 	active_quad_idx = find((Yprod >= -1) & (Yprod <= 1));
	active_quad_idx = find(absYprod <= 1);
    active_lin_X = X(active_lin_idx, :);
    active_lin_Y = Y(active_lin_idx, :);
    active_quad_X = X(active_quad_idx, :);
    active_quad_Y = Y(active_quad_idx, :);
	active_quad_E = Ypred(active_quad_idx) - active_quad_Y;
	active_lin_E = - 4 * active_lin_Y;
	active_lin_E_sec = Ypred(active_lin_idx);
	dK = w * (2 * active_quad_E' * active_quad_X) / 2 + (2 * active_quad_E' * active_quad_X)' * w' / 2 ...
		+ w * (active_lin_E' * active_lin_X) / 2 + (active_lin_E' * active_lin_X)' * w' / 2 + lambdawgamma;  
end
df = dK(:);
if (useregularizer == 1),
	f = lambda * w' * kernelMatrix * w;
else
	f = 0;
end;

if ~isempty(active_quad_E),
	f = f + active_quad_E' * active_quad_E;
end;

if ~isempty(active_lin_E),
	f = f + active_lin_E' * active_lin_E_sec;
end;
