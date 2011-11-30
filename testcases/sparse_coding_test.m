%%
numFeatures = 100;
numSamples = 300;
numAtoms = 256;
lambda = 0.1;

%%
X = randn(numFeatures, numSamples);
D = randn(numFeatures, numAtoms);
params.lambda = 0.1;

%%
A = l1qp_featuresign(X, D, lambda);
A = full(A);
B = l1qp_featuresign_mex(X, D, lambda);
C = mexLasso(X, D, params);
C = full(C);

fprintf('l1qp matmexdiff %g\n', norm(A - B, 'fro'));
fprintf('l1qp dataerror %g\n', norm(X - D * B, 'fro'));
fprintf('lasso dataerror %g\n', norm(X - D * C, 'fro'));
fprintf('l1qp sparsity %d\n', nnz(B));
fprintf('lasso sparsity %d\n', nnz(C));
fprintf('l1qp l1norm %g\n', sum(abs(B(:))));
fprintf('lasso l1norm %g\n', sum(abs(C(:))));
fprintf('l1qp objval %g\n', norm(X - D * B, 'fro') ^ 2 / 2 + lambda * sum(abs(B(:))));
fprintf('lasso objval %g\n', norm(X - D * C, 'fro') ^ 2 / 2 + lambda * sum(abs(C(:))));

KDX = D' * X;
KDD = D' * D;
E = l1kernel_featuresign(KDX, KDD, lambda);
E = full(E);
F = l1kernel_featuresign_mex(KDX, KDD, lambda);
fprintf('l1kernel matmexdiff %g\n', norm(E - F, 'fro'));
fprintf('l1 kernelqpdiff %g\n', norm(F - B, 'fro'));

%%
% TODO: Add ista and fista versions once I debug them properly.
