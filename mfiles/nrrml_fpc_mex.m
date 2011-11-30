function [A, converged, violations] = nrrml_fpc_mex(X, fullConstraintMat,...
						tauList, varargin)
%NNRML_FPC    Nuclear-norm regularized metric learning using FPC.
%   A = NNRML_FPC(X, C, TAU) returns the NxN matrix K that is the solution
%   to the following optimization problem:
%
%   min_A tau * ||A||_* + 1 / 2 * sum((Trace(A * Dm) - bm) ^ 2)
%   s.t. A >= 0
%
%   where ||.||_* is the nuclear norm and of a matrix respectively, >=
%   denotes positive semi-definiteness, and 
%   D_{m} = (X(:, i_m) - X(:, j_m)) * (X(:, i_m) - X(:, j_m))',
%   d_{klm} = K(k_m, k_m) + K(l_m, l_m) - 2 * K(k_m, l_m),
%   i_m = C(m, 1), j_m = C(m, 2), beta_m = C(m, 5),
%	1 < i_m, j_m, k_m, l_m < N, 1 < m < M,
%   and M is the number of rows of C. 
%
%	NNRML_FPC uses the FPC algorithm to solve the above problem.
%
%   Ioannis Gkioulekas, Harvard University, August 2011
%   Last modification: August 18th, 2011.

A = nrml_fpc_mex(X, 'T', fullConstraintMat, tauList, varargin{:});
converged = - 1;

if (nargout >= 3),
	violations = getViolations(A, X, fullConstraintMat(:, 1:2), fullConstraintMat(:, 3), 'targets');
end;
