function [l dl] = huber_obj_grad_multiclass(Phi, X, Y, weights, bias)

% assumes sample x and weights are vector columns, and takes product 
% w' * Phi * x
%
% returns objective function l and gradient dl with respect to Phi

% 1 sample case
% YYpred = Y * Ypred;
% if (YYpred > 1),
% 	l = 0;
% 	dl = zeros(size(Phi))';
% elseif ((YYpred < 1) && (YYpred > -1)),
% 	l = (1 - YYpred) ^ 2;
% 	dl = 2 * (Ypred - Y) * weights * X';
% elseif (YYpred < -1),
% 	l = -4 * YYpred;
% 	dl = - 4 * Y * weights * X';
% end

[m cnum] = size(weights);
[n numSamples] = size(X);
cnum = size(Y, 2);

dl = zeros(m * n, 1);
l = 0;
dltemp = zeros(m * n, 1);
ltemp = 0;

for iter = 1:cnum,
	[ltemp dltemp] = huber_obj_grad(Phi, X, Y(:, iter), weights(:, iter), bias(iter));
	l = l + ltemp;
	dl = dl + dltemp;
end;
