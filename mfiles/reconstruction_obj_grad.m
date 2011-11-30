function [l dl] = reconstruction_obj_grad(D, X, A)

% assumes sample x are vector columns, and uses objective function
% ||X - D * A||_fro^2
%
% returns objective function l and gradient dl with respect to D. A should
% be provided as sparse.

n = size(X, 1);
k = size(A, 1);
D = reshape(D, [n k]);

Err = X - D * A;
l = norm(Err, 'fro');
dl = - 2 * Err * A';
dl = dl(:);
