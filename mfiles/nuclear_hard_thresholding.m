function [Xr normValue] = nuclear_hard_thresholding(X, rank)

[U S V] = svd(X, 'econ');
threshSVD = diag(S);
threshSVD((rank + 1):end) = 0;
inds = 1:rank;
Xr = U(:, inds) * diag(threshSVD(inds)) * V(:, inds)';
normValue = sum(threshSVD(inds));

