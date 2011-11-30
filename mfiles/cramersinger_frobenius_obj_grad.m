function [f df] = cramersinger_frobenius_obj_grad(W, X, Y, gamma, lambda)

[signalDim numSamples] = size(X);
numTasks = length(unique(Y));
W = reshape(W, [signalDim numTasks]);
[f1 df1] = cramersinger_approx_obj_grad_mex(W, X, Y, gamma, numTasks);
f2 = sum(W(:) .^ 2);
f = f1 + lambda * f2;
df = df1 + lambda * 2 * W;
df = df(:);
