function [obj deriv] = gram_obj_grad(X, D, Dt, Ig)
% objective function value and gradient of sensing matrix learning routine,
% for objective function ||Gram - I||_fro ^ 2.
%
%	Note: Inputs are chosen to optimize performance. Optional arguments
%	should be precalculated before the optimization algorithm begins
%	running and provided, to significantly reduce running speed.
%
%	Inputs
%	X: vectorized form of sensing matrix.
%	D: dictionary.
%	Dt: transpose of dictionary (optional).
%	Ig: identity matrix of size size(D, 2) (optional).
%
%	Outputs
%	obj: value of objective function.
%	deriv: vectorized form of gradient.
%

% original signal size
[n k] = size(D);
m = numel(X) / n;
Phi = reshape(X, [m n]);

if (nargin < 3),
	Dt = transpose(D);
end;
if (nargin < 4),
	Ig = eye(k);
end;

PhiD = Phi * D;
% G = Dt * (Phi' * Phi) * D;
K = PhiD' * PhiD - Ig;
obj = norm(K, 'fro') ^ 2;

if (nargout >= 2),
	deriv = 4 * PhiD * K * Dt;
	deriv = deriv(:);
end;
