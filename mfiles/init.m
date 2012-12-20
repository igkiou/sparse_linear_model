A = randn(50,50);
A = A'*A;
[U V] = eig(A);
A = U*sqrt(V)*U';
v = randn(50,1);
L = chol(A)';
% dbmex on
% [L1 stat] = test_cholesky(L', v, 1);
