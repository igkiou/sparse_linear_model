function [K, converged, X, violations] = nmmds_apg(numPoints, fullConstraintMat,...
						tauList, tolerance, delta0, numIters, rankIncrement,...
						lineSearchFlag, eta, truncateFlag, gap)
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

if ((nargin < 4) || isempty(tolerance)),
	tolerance = 10 ^ - 5;
end;

if ((nargin < 5) || isempty(delta0)),
	delta0 = 1000;
end;

if ((nargin < 6) || isempty(numIters)),
	numIters = 50000;
end;
										
if ((nargin < 7) || isempty(rankIncrement)),
	rankIncrement = - 1;
end;
										
if ((nargin < 8) || isempty(lineSearchFlag)),
	lineSearchFlag = 0;
end;
										
if ((nargin < 9) || isempty(eta)),
	eta = 1.1;
end;
										
if ((nargin < 10) || isempty(truncateFlag)),
	truncateFlag = 0;
end;
										
if ((nargin < 11) || isempty(gap)),
	gap = 5;
end;

rankEstimate = 10;
constraintMat = fullConstraintMat(:, 1:4);
betaVector = fullConstraintMat(:, 5);
numRepeats = length(tauList);
K = zeros(numPoints, numPoints);
KOld = K;
t = 1;
tOld = t;

% eta > 1, as opposed to eta < 1 in Toh (and multiplying by 1 / eta)
% tau ~ mu
% delta ~ tau

converged = 1;
for iterRepeat = 1:numRepeats,
	tau = tauList(iterRepeat);
	fprintf('Now running repeat %d out of %d, tau %g.\n', iterRepeat, numRepeats, tau);
	delta = delta0;
	% TODO: change continuation to not change tau, or change mu on the run,
	% or cap delta, or some combination thereof. Same in mex.
% 	t = 1;
% 	tOld = t;
% 	KOld = K;
	for iter = 1:numIters,
		if (mod(iter, 10000) == 0),
			fprintf('Now running iter %d.\n', iter);
		end;
		a = (tOld - 1) / t;
		L = K + a * (K - KOld);
		[LfObj, LfGrad] = f_func(L, constraintMat, betaVector);
% 		delta = delta / eta;
% 		delta = delta0;

		[delta, LfGDShrink, rankEstimate] = line_search(L, LfObj, LfGrad,...
				tau, delta, lineSearchFlag, eta, rankIncrement, rankEstimate,...
				truncateFlag, gap, constraintMat, betaVector);
		KOld = K;
		K = LfGDShrink;
		tOld = t;
		t = (1 + sqrt(1 + 4 * tOld ^ 2)) / 2;
		
% 		[KfObj, KfGrad] = f_func(L, constraintMat, betaVector);
% 		S = delta * (L - K) + (KfGrad - LfGrad);
		normK = norm(K, 'fro');
		normKDiff  = norm(KOld - K, 'fro');
		normKOld = norm(KOld, 'fro');
% 		if ((norm(S,'fro') > eps) && (norm(S, 'fro') / delta / max(1, norm(K, 'fro')) < tolerance)),
		if ((normK > eps) && (normKDiff / max(1, normKOld) < tolerance)),
			break;
		end;
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

end

function [delta, LfGDShrink, rankEstimate] = line_search(L, LfObj, LfGrad,...
					tau, delta, lineSearchFlag, eta, rankIncrement,...
					rankEstimate, truncateFlag, gap, constraintMat, betaVector)

	while(1),
		LfGD = L - 1 / delta * LfGrad;
		[LfGDShrink, LfGDShrinkNorm, rankEstimate] = shrinkage(LfGD,...
			tau / delta, rankIncrement, rankEstimate, truncateFlag, gap);
		if (~lineSearchFlag),
			break;
		end;
		LfGDShrinkfObj = f_func(LfGDShrink, constraintMat, betaVector);
% 		LfGDShrinkFObj = F_func(LfGDShrinkfObj, LfGDShrinkNorm, tau);
% 		LQObj = Q_func(LfGDShrink, LfGDShrinkNorm, LfGD, LfObj, LfGrad, delta, tau);
		LQmodObj = Q_func_mod(LfGDShrink, LfGD, LfObj, LfGrad, delta);
% 		LQmodObj = Q_func_alt(LfGDShrink, L, LfObj, LfGrad, delta);
% 		if (LfGDShrinkFObj <= LQObj),
		if (LfGDShrinkfObj <= LQmodObj),
			break;
		end;
		delta = delta * eta;
% 		if (delta > 1)
% 			delta = 1;
% 			break;
% 		end;
	end;
	
end

function [KShrink, KShrinkNorm, rankEstimate] = shrinkage(K, tau, rankIncrement, rankEstimate, truncateFlag, gap)

% TODO: Add delay for truncation in early iterations.
% [KShrink KShrinkNorm rankEstimate] = nuclear_psd_proximal_truncate(K, tau, rankIncrement, rankEstimate, truncateFlag, gap);
[KShrink1 KShrinkNorm1 rankEstimate] = nuclear_psd_proximal(K, tau, rankIncrement, rankEstimate);
% [KShrink2 KShrinkNorm2] = nuclear_psd_proximal_mex(K, tau);
% if (norm(KShrink1-KShrink2, 'fro') > 10^-10)
% 	disp('problem');
% end;
KShrink = KShrink1; KShrinkNorm = KShrinkNorm1;

end

% NOTE: As KNorm is used both by Q and F, we can ignore it in their
% comparison. Q_func_mod does that, and is compared directly with f instead
% of F.
function QObj = Q_func_mod(K, LfGD, LfObj, LfGrad, delta)

% LfGD = L - 1 / delta * LfGrad;
QObj = delta / 2 * norm(K - LfGD, 'fro') ^ 2 + LfObj...
		- 1 / 2 / delta * norm(LfGrad, 'fro') ^ 2;
end

function QObj = Q_func_alt(K, L, LfObj, LfGrad, delta)

QObj = delta / 2 * norm(K - L, 'fro') ^ 2 + trace((K - L)' * LfGrad) + LfObj;

end

function QObj = Q_func(K, KNorm, LfGD, LfObj, LfGrad, delta, tau)

% LfGD = L - 1 / delta * LfGrad;
QObj = delta / 2 * norm(K - LfGD, 'fro') ^ 2 + tau * KNorm + LfObj...
		- 1 / 2 / delta * norm(LfGrad, 'fro') ^ 2;
end

function FObj = F_func(KfObj, KNorm, tau)

FObj = KfObj + tau * KNorm;

end

function [fObj, fGrad] = f_func(K, constraintMat, betaVector)

fObj = 0;
fGrad = zeros(size(K));
numConstraints = length(betaVector);
for iterConstraint = 1:numConstraints,
	beta = betaVector(iterConstraint);
	coord1 = constraintMat(iterConstraint, 1);
	coord2 = constraintMat(iterConstraint, 2);
	coord3 = constraintMat(iterConstraint, 3);
	coord4 = constraintMat(iterConstraint, 4);
	dij = K(coord1, coord1) + K(coord2, coord2) - 2 * K(coord1, coord2);
	dkl = K(coord3, coord3) + K(coord4, coord4) - 2 * K(coord3, coord4);
	% dij - dkl < - 1, beta = - 1, NIPS10
	if (dij - dkl - beta > 0),
		fObj = fObj + dij - dkl - beta;
		if (nargin >= 2),
			fGrad(coord1, coord1) = fGrad(coord1, coord1) + 1;
			fGrad(coord2, coord2) = fGrad(coord2, coord2) + 1;
			fGrad(coord3, coord3) = fGrad(coord3, coord3) - 1;
			fGrad(coord4, coord4) = fGrad(coord4, coord4) - 1;
			fGrad(coord1, coord2) = fGrad(coord1, coord2) - 1;
			fGrad(coord2, coord1) = fGrad(coord2, coord1) - 1;
			fGrad(coord3, coord4) = fGrad(coord3, coord4) + 1;
			fGrad(coord4, coord3) = fGrad(coord4, coord3) + 1;
		end;
	end;
end;

end
