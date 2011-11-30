function [A, converged, violations] = nrrml_fpc(X, fullConstraintMat,...
						tauList, tolerance, delta, numIters, rankIncrement)
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

if ((nargin < 4) || isempty(tolerance)),
	tolerance = 10 ^ - 5;
end;

if ((nargin < 5) || isempty(delta)),
	delta = 0.00001;
end;

if ((nargin < 6) || isempty(numIters)),
	numIters = 100000;
end;
										
if ((nargin < 7) || isempty(rankIncrement)),
	rankIncrement = - 1;
end;

[N numPoints] = size(X);
rankEstimate = 10;
constraintMat = fullConstraintMat(:, 1:2);
betaVector = fullConstraintMat(:, 3);
numConstraints = length(betaVector);
A = zeros(N, N);
AOld = A;
numRepeats = length(tauList);

converged = 1;
for iterRepeat = 1:numRepeats,
	tau = tauList(iterRepeat);
	fprintf('Now running repeat %d out of %d, tau %g.\n', iterRepeat, numRepeats, tau);
	for iter = 1:numIters,
		if (mod(iter, 10000) == 0),
			fprintf('Now running iter %d.\n', iter);
		end;
		AOld = A;
		normAOld = norm(AOld, 'fro');
		objValue = 0;
		for iterConstraint = 1:numConstraints,
			beta = betaVector(iterConstraint);
			coord1 = constraintMat(iterConstraint, 1);
			coord2 = constraintMat(iterConstraint, 2);
			Dij = (X(:, coord1) - X(:, coord2)) * (X(:, coord1) - X(:, coord2))';
			alpha = trace(AOld * Dij) - beta;
			objValue = objValue + 0.5 * alpha ^ 2;
			A = A - delta * alpha * Dij;
		end;

		[A foo rankEstimate] = nuclear_psd_proximal(A, tau * delta, rankIncrement, rankEstimate);

		normA = norm(A, 'fro');
		normADiff  = norm(AOld - A, 'fro');
		if ((normA > eps) && (normADiff / max(1, normAOld) < tolerance)),
			break;
		end;
	end;	
end;

if (iter == numIters),
	converged = - 1;
end

if (nargout >= 3),
	violations = getViolations(A, X, constraintMat, betaVector, 'targets');
end;
