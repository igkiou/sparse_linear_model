function [K, converged, X, violations] = nmmds_svt(numPoints, fullConstraintMat,...
							tau, tolerance, delta, numIters, rankIncrement)
%NMMDS_SVT  Non-metric MDS based on nuclear norm minimization, using SVT.
%   K = NMMDS_SVT(C, N, TAU) returns the NxN matrix K that is the solution
%   to the following optimization problem:
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
%	NMMDS_SVT uses a variant of Uzawa's algorithm to solve the above
%	problem, as proposed in [1], [2].
%
%   [K, SLACK] = NMMDS_SVT(C, N, TAU) further returns the amount of
%   violation of the constraints, in the case when the algorithm returns
%   before convergence, or a feasible point does not exist. A negative
%   value indicates that the algorithm terminated after reaching the
%   maximum number of iterations (non-convergence).
%
%   [K, SLACK, X] = NMMDS_SVT(C, N, TAU) also returns a Euclidean embedding
%   whose dimension is the same as the rank of K (in the worst case, equal
%   to N, the number of points). The embedding is produced following [3],
%   i.e.
%
%   [U S V] = svd(K); X = U * sqrt(S);
%
%   K = NMMDS_SVT(C, N, TAU, TOL, DELTA, MAXITER) allows specifying
%   parameters involved in the optimization algorithm, where TOL is a
%   tolerance in the aggregate amount of violation of the constraints,
%   DELTA is the step size of Uzawa's algorithm's iteration, and MAXITER is
%   the maximum number of iterations. Any subset of these values may be set
%   to [] (empty), in which case the default values used are TOL = 10 ^ -5,
%   DELTA = 0.1, and MAXITER = 50000.
%
%   K = NMMDS_SVT(C, N, TAU, TOL, DELTA, MAXITER, RANKINCREMENT) can be
%   used in large problems, to perform partial instead of full
%   eigendecompositions. It requires having the package PROPACK, and its
%   use is experimental. Please refer to the code for more details if
%   interested in using it. Otherwise, either skip it or set it to a
%   negative value (default value is - 1).
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

if ((nargin < 4) || isempty(tolerance)),
	tolerance = 10 ^ - 5;
end;

if ((nargin < 5) || isempty(delta)),
	delta = 0.1;
end;

if ((nargin < 6) || isempty(numIters)),
	numIters = 50000;
end;
										
if ((nargin < 7) || isempty(rankIncrement)),
	rankIncrement = - 1;
end;

rankEstimate = 10;
constraintMat = fullConstraintMat(:, 1:4);
betaVector = fullConstraintMat(:, 5);
numConstraints = length(betaVector);
K = zeros(numPoints, numPoints);
zetaVector = zeros(numConstraints, 1);

% Kicking by a few steps.
% dij - dkl < - 1, beta = - 1, NIPS10
minNegBeta = min(betaVector(betaVector < 0));
kappaInit = floor(tau / numPoints / abs(minNegBeta) / delta);
zetaVector = kappaInit * delta * max(- betaVector, 0);
% % dkl - dij > 1, beta = 1, Uzawa08
% maxPosBeta = max(betaVector(betaVector > 0));
% kappaInit = floor(tau / numPoints / maxPosBeta / delta);
% zetaVector = kappaInit * delta * max(betaVector, 0);

converged = 1;
for iter = 1:numIters,
	if (mod(iter, 10000) == 0),
		fprintf('Now running iter %d.\n', iter);
	end;
	K = zeros(numPoints, numPoints);
	for iterConstraint = 1:numConstraints,
		zeta = zetaVector(iterConstraint);
		% i
		coord1 = constraintMat(iterConstraint, 1);
		% j
		coord2 = constraintMat(iterConstraint, 2);
		% k
		coord3 = constraintMat(iterConstraint, 3);
		% l
		coord4 = constraintMat(iterConstraint, 4);
		% dij - dkl
		K(coord1, coord1) = K(coord1, coord1) + zeta;
		K(coord2, coord2) = K(coord2, coord2) + zeta;
		K(coord3, coord3) = K(coord3, coord3) - zeta;
		K(coord4, coord4) = K(coord4, coord4) - zeta;
		K(coord1, coord2) = K(coord1, coord2) - zeta;
		K(coord2, coord1) = K(coord2, coord1) - zeta;
		K(coord3, coord4) = K(coord3, coord4) + zeta;
		K(coord4, coord3) = K(coord4, coord3) + zeta;
% 		% dkl - dij
% 		K(coord1, coord1) = K(coord1, coord1) - zeta;
% 		K(coord2, coord2) = K(coord2, coord2) - zeta;
% 		K(coord3, coord3) = K(coord3, coord3) + zeta;
% 		K(coord4, coord4) = K(coord4, coord4) + zeta;
% 		K(coord1, coord2) = K(coord1, coord2) + zeta;
% 		K(coord2, coord1) = K(coord2, coord1) + zeta;
% 		K(coord3, coord4) = K(coord3, coord4) - zeta;
% 		K(coord4, coord3) = K(coord4, coord3) - zeta;
	end;
	[K foo rankEstimate] = nuclear_psd_proximal(K, tau, rankIncrement, rankEstimate);

	maxDivergence = 0;
	normDivergence = 0;
	for iterConstraint = 1:numConstraints,
		beta = betaVector(iterConstraint);
		coord1 = constraintMat(iterConstraint, 1);
		coord2 = constraintMat(iterConstraint, 2);
		coord3 = constraintMat(iterConstraint, 3);
		coord4 = constraintMat(iterConstraint, 4);
		dij = K(coord1, coord1) + K(coord2, coord2) - 2 * K(coord1, coord2);
		dkl = K(coord3, coord3) + K(coord4, coord4) - 2 * K(coord3, coord4);
		% dij - dkl < - 1, beta = - 1, NIPS10
		zetaVector(iterConstraint) = zetaVector(iterConstraint)...
									- delta * max(dij - dkl - beta, 0);
		if (dij - dkl - beta > 0),
			maxDivergence = dij - dkl - beta;
			normDivergence = normDivergence + (dij - dkl - beta) ^ 2;
		end;
% 		% dkl - dij > 1, beta = 1, Uzawa08
% 		zetaVector(iterConstraint) = max(zetaVector(iterConstraint)...
% 									+ delta * (beta + dij - dkl), 0);
% 		if (dkl - dij - beta < 0),
% 			maxDivergence = beta - dkl + dij;
% 			normDivergence = normDivergence + (beta + dij - dkl) ^ 2;
% 		end;
	end;
	if ((normDivergence < tolerance) || (maxDivergence < eps)),
		break
	end;
end;	

if (iter == numIters),
	converged = - 1;
end;

if (nargout >= 3),
	[U S V] = svd(K);
	X = U * sqrt(S);
end;

if (nargout >= 4),
	violations = getViolations(K, [], constraintMat, betaVector);
end;

