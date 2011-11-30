function [f df] = multihuberhinge_obj_grad(W, X, Y, lambda, useregularizer)
% Y must be -1/1 as encoded by oneofc, and must be T x S. X is N x S, and W
% is N x T.

[signalDim numSamples] = size(X);
numTasks = size(Y, 1);
W = reshape(W, [signalDim numTasks]);
f = 0;
df = zeros(size(W));
dftemp = zeros(signalDim + 1, 1);
for iterT = 1:numTasks,
	[ftemp, dftemp] = huberhinge_obj_grad([W(:, iterT); 0], X, Y(iterT, :), lambda, [], [], 0, useregularizer);
	df(:, iterT) = dftemp(1:end - 1);
	f = f + ftemp;
end;
df = df(:);
