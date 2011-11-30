function [f df] = cramersinger_nuclear_obj_grad(W, X, Y, gamma, rho, lambda)

[signalDim numSamples] = size(X);
numTasks = length(unique(Y));
W = reshape(W, [signalDim numTasks]);
[f2 df2] = nuclear_approx_obj_grad_mex(W, rho, signalDim, numTasks);
[f1 df1] = cramersinger_approx_obj_grad_mex(W, X, Y, gamma, numTasks);
f = f1 + lambda * f2;
df = df1 + lambda * df2;
df = df(:);
