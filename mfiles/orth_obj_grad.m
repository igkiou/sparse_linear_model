function [obj deriv] = orth_obj_grad(X, m, n)
% objective function value and gradient of orthogonality routine, for
% objective function ||I - Phi' * Phi||_fro ^ 2. 
%
%	Inputs
%	X: vectorized form of sensing matrix.
%	m: first dimension of sensing matrix.
%	n: second dimension of sensing matrix.
%
%	Outputs
%	obj: value of objective function.
%	deriv: vectorized form of gradient.
%

Phi = reshape(X, [m n]);
I = eye(m);
% mat = I - Phi * Phi';
obj = norm(I - Phi * Phi', 'fro') ^ 2;

if (nargout >= 2),
	deriv = 4 * (I - Phi * Phi') * Phi;
	deriv = deriv(:);
end;

% Phi = reshape(X, [m n]);
% I = eye(n);
% obj = norm(I - Phi' * Phi, 'fro') ^ 2;
% 
% if (nargout >= 2),
% 	deriv = 4 * Phi * (Phi' * Phi - I);
% 	deriv = deriv(:);
% end;
