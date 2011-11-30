function [f df] = cramersinger_approx_obj_grad(W, X, y, lambda, numTasks) %#ok
%
% Make sure that classes in y are of the form 1:T, T being the number of
% classes. y is a vector of size 1 x S.

[N S] =  size(X);
uniqueLabels = unique(y);
T = length(uniqueLabels);
if (any(uniqueLabels' ~= 1:T)),
	error('Labels in y must be of the form 1:T, T being the number of classes.');
end;
W = reshape(W, [N T]);
S = size(X, 2);

WtX = W' * X;
corrOuts = WtX(sub2ind([T S], y', 1:S));

expMat = bsxfun(@minus, WtX, corrOuts);
expMat = exp(lambda * (expMat + 1));
logArg = sum(expMat, 1) + 1 - exp(lambda);
f = sum(log(logArg)) / S / lambda;

dfmult = bsxfun(@times, 1 ./ logArg, expMat);
dfmult(sub2ind([T S], y', 1:S)) = - 1 ./ logArg .* (logArg - 1);
df = X * dfmult';
df = df / S;
df = df(:);
% df = zeros(size(W));
% dftemp = zeros(size(W));
% for iterS = 1:S,
% 	dftemp = 1 / logArg(iterS) * X(:, iterS) * expMat(:, iterS)';
% 	dftemp(:, y(iterS)) = - 1 / logArg(iterS) * (logArg(iterS) - 1) * X(:, iterS);
% 	df = df + dftemp;
% end;
