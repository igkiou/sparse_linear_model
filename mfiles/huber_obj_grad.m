function [l dl] = huber_obj_grad(Phi, X, Y, weights, bias, wXtensor)

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

Y = Y';
m = length(weights);
[n numSamples] = size(X);
Phi = reshape(Phi, [m n]);

if (nargin < 6),
	Xt = xrepmat((X(:))', [m, 1]);
	Xtt = reshape(Xt, [m * n numSamples]);
	Wt = xrepmat(weights, [n 1]);
	wXtensor = bsxfun(@times, Wt, Xtt);
end;

Ypred = weights' * Phi * X + bias;
YYpred = Y .* Ypred;
absYYpred = abs(YYpred);

% GreatOne = YYpred > 1;
LessMOne = YYpred < -1;
LessOneGreatMOne = absYYpred <= 1;

lvec = zeros(1, numSamples);
lvec(LessMOne) = -4 * YYpred(LessMOne);
lvec(LessOneGreatMOne) = (1 - YYpred(LessOneGreatMOne)) .^ 2;
% lvec(LessOneGreatMOne) = (1 - YYpred(LessOneGreatMOne))' * (1 - YYpred(LessOneGreatMOne));
l = sum(lvec) / numSamples;

dlmat = zeros(numel(Phi), numSamples);
dlmat(:, LessMOne) = -4 * bsxfun(@times, Y(LessMOne), wXtensor(:, LessMOne));
dlmat(:, LessOneGreatMOne) = 2 * bsxfun(@times, Ypred(LessOneGreatMOne) - Y(LessOneGreatMOne), wXtensor(:, LessOneGreatMOne));
dl = sum(dlmat, 2) / numSamples;
