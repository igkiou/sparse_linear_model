function [K, converged, X, violations] = nmmds_svp_mex(numPoints, fullConstraintMat,...
						weights, rank, varargin)
%NMMDS    Non-metric MDS based on nuclear norm minimization.
%   K = NMMDS(C, N, TAU) returns the NxN matrix K that is the solution to
%   the following optimization problem:
%
%   min_K tau * ||K||_* + 1 / 2 * ||K||_{Fro}^2
%   s.t. d_{ijm} - d_{klm} < beta_m, 1 <= m <= M,
%        K >= 0
%
%   where ||.||_* and ||K||_{Fro} are the nuclear (or trace) norm and and
%   Frobenius norm of a matrix respectively, >= denotes positive
%   semi-definiteness, and the constraints are relational constraints,
%   where 
%   d_{ijm} = K(i_m, i_m) + K(j_m, j_m) - 2 * K(i_m, j_m),
%   d_{klm} = K(k_m, k_m) + K(l_m, l_m) - 2 * K(k_m, l_m),
%   i_m = C(m, 1), j_m = C(m, 2), k_m = C(m, 3), l_m = C(m, 4), 
%   1 < i_m, j_m, k_m, l_m < N, beta_m = C(m, 5),
%   and M is the number of rows of C. 
%
%	NMMDS uses a variant of Uzawa's algorithm to solve the above problem,
%	as proposed in [1], [2].
%
%   [K, SLACK] = NMMDS(C, N, TAU) further returns the amount of violation of
%   the constraints, in the case when the algorithm returns before
%   convergence, or a feasible point does not exist. A negative value
%   indicates that the algorithm terminated after reaching the maximum
%   number of iterations (non-convergence).
%
%   [K, SLACK, X] = NMMDS(C, N, TAU) also returns a Euclidean embedding
%   whose dimension is the same as the rank of K (in the worst case, equal
%   to N, the number of points). The embedding is produced following [3],
%   i.e.
%
%   [U S V] = svd(K); X = U * sqrt(S);
%
%   K = NMMDS(C, N, TAU, TOL, DELTA, MAXITER) allows specifying parameters
%   involved in the optimization algorithm, where TOL is a tolerance in the
%   aggregate amount of violation of the constraints, DELTA is the step
%   size of Uzawa's algorithm's iteration, and MAXITER is the maximum
%   number of iterations. Any subset of these values may be set to []
%   (empty), in which case the default values used are TOL = 10 ^ -5, DELTA
%   = 0.1, and MAXITER = 50000.
%
%   K = NMMDS(C, N, TAU, TOL, DELTA, MAXITER, RANKINCREMENT) can be used in
%   large problems, to perform partial instead of full eigendecompositions.
%   It requires having the package PROPACK, and its use is experimental.
%   Please refer to the code for more details if interested in using it.
%   Otherwise, either skip it or set it to a negative value (default value
%   is - 1).
%
%   [1] J. Cai, E. Candes, amd Z. Shen, "A singular value thresholding
%   algorithm for matrix completion," Arxiv preprint 2008.
%   [2] P. Jain, B. Kulis, I. Dhillon, "Inductive regularized learning of
%   kernel functions," NIPS 2010.
%   [3] Y. Bengio, J. Paiement, P. Vincent, O. Delalleau, N. Le Roux, and
%   M. Ouimet, "Out-of-sample extensions for lle, isomap, mds, eigenmaps,
%   and spectral clustering," NIPS 2003.
%
%   Ioannis Gkioulekas, Harvard University, July 2011
%   Last modification: August 2nd, 2011.
%   TODO: Add support for slack variables.

K = nrkl_svp_mex(numPoints, 'R', fullConstraintMat, weights, rank, varargin{:});

converged = - 1;

if (nargout >= 3),
	[U S V] = svd(K);
	X = U * sqrt(S);
end;

if (nargout >= 4),
	violations = getViolations(K, [], fullConstraintMat(:, 1:4), fullConstraintMat(:, 5));
end;
