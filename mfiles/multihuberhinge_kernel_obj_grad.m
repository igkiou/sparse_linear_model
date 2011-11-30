function [f, df] = multihuberhinge_kernel_obj_grad(kernelMatrix, W, X, Y, lambda, classLabels, useregularizer)

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

error(nargchk(6, 7, nargin));
if nargin < 7,
    useregularizer = 1;
end

numTasks = length(classLabels);
f = 0;
df = zeros(size(X, 1) ^ 2, 1);
dfTemp = df;
fTemp = f;
Ytemp = Y;
taskLabel = classLabels(1);

for iterT = 1:numTasks,
	taskLabel = classLabels(iterT);
	Ytemp = double(Y == taskLabel);
	Ytemp(Ytemp == 0) = -1;
	[fTemp, dfTemp] = huberhinge_kernel_obj_grad(kernelMatrix, W(:, iterT), X, Ytemp, lambda, useregularizer);
	f = f + fTemp;
	df = df + dfTemp;
end;
