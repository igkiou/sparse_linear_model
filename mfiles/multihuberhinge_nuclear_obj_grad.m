function [f df] = multihuberhinge_nuclear_obj_grad(W, X, Y, rho, lambda)
% Y must be -1/1 as encoded by oneofc, and must be T x S. X is N x S, and W
% is N x T.

[signalDim numSamples] = size(X);
numTasks = size(Y, 1);
W = reshape(W, [signalDim numTasks]);
[f1 df1] = multihuberhinge_obj_grad(W, X, Y, lambda, 0);
[f2 df2] = nuclear_approx_obj_grad_mex(W, rho, signalDim, numTasks);
f = f1 + lambda * f2;
df = df1 + lambda * df2;
df = df(:);
