function [f, df] = nuclear_approx_obj_grad(X, r, M, N)

X = reshape(X, [M N]);
% [M N] = size(X);
MN = min(M, N);
[U S V] = svd(X);
[flvec dflvec] = abs_smooth_obj_grad(diag(S), r);
f = sum(flvec);
S(sub2ind([M N], 1:MN, 1:MN)) = dflvec;
df = U * S * V';
df = df(:);
