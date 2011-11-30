function [l dl] = square_obj_grad(Phi, X, Y, weights, bias, wXtensor)

% assumes sample x and weights are vector columns, and takes product 
% w' * Phi * x
%
% returns objective function l and gradient dl with respect to Phi

% 1 sample case
% YYpred = Y * Ypred;
% if (YYpred > 1),
% 	l = 0;
% 	dl = zeros(size(Phi))';
% else
% 	l = (1 - YYpred) ^ 2;
% 	dl = 2 * (Ypred - Y) * weights * X';
% end

Y = Y';
m = length(weights);
[n numSamples] = size(X);
Phi = reshape(Phi, [m n]);

if (nargin < 6),
	Xt = repmat((X(:))', [m, 1]);
	Xtt = reshape(Xt, [m * n numSamples]);
	Wt = repmat(weights, [n 1]);
	wXtensor = bsxfun(@times, Wt, Xtt);
end;

Ypred = weights' * Phi * X + bias;
YYpred = Y .* Ypred;

LessOne = YYpred < 1;

lvec = zeros(1, numSamples);
lvec(LessOne) = (1 - YYpred(LessOne)) .^ 2;
l = sum(lvec) / numSamples;
% l1 = (1 - YYpred(LessOne)) * (1 - YYpred(LessOne))' / numSamples;

dlmat = zeros(numel(Phi), numSamples);
dlmat(:, LessOne) = 2 * bsxfun(@times, Ypred(LessOne) - Y(LessOne), wXtensor(:, LessOne));
dl = sum(dlmat, 2) / numSamples;
