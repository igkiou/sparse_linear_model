function [A, converged, violations] = nrml_apg(X, fullConstraintMat,...
						tauList, tolerance, delta0, numIters, rankIncrement,...
						lineSearchFlag, eta, truncateFlag, gap)
%NNRML_APG    Nuclear-norm regularized metric learning using FPC.
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

if ((nargin < 5) || isempty(delta0)),
	delta0 = 1000;
end;

if ((nargin < 6) || isempty(numIters)),
	numIters = 1000;
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

[N numPoints] = size(X);
rankEstimate = 10;
constraintMat = fullConstraintMat(:, 1:2);
betaVector = fullConstraintMat(:, 3);
numConstraints = length(betaVector);
A = zeros(N, N);
AOld = A;
numRepeats = length(tauList);
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
% 	AOld = A;
	for iter = 1:numIters,
		if (mod(iter, 10000) == 0),
			fprintf('Now running iter %d.\n', iter);
		end;
		a = (tOld - 1) / t;
		L = A + a * (A - AOld);
		[LfObj, LfGrad] = f_func(L, X, constraintMat, betaVector);
% 		delta = delta / eta;
% 		delta = delta0;

		[delta, LfGDShrink, rankEstimate] = line_search(L, LfObj, LfGrad,...
				tau, delta, lineSearchFlag, eta, rankIncrement, rankEstimate,...
				truncateFlag, gap, constraintMat, betaVector);
		AOld = A;
		A = LfGDShrink;
		tOld = t;
		t = (1 + sqrt(1 + 4 * tOld ^ 2)) / 2;
		
% 		[AfObj, AfGrad] = f_func(L, X, constraintMat, betaVector);
% 		S = delta * (L - A) + (AfGrad - LfGrad);
% 		if ((norm(S,'fro') > eps) && (norm(S, 'fro') / delta / max(1, norm(A, 'fro')) < tolerance)),
		normA = norm(A, 'fro');
		normAOld = norm(AOld, 'fro');
		normADiff = norm(ADiff, 'fro');
		if ((normA > eps) && (normADiff / max(1, normAOld) < tolerance)),
			break;
		end;
	end;
end;

if (iter == numIters),
	converged = - 1;
end;

if (nargout >= 3),
	violations = getViolations(A, X, constraintMat, betaVector, 'targets');
end;

end

function [delta, LfGDShrink, rankEstimate] = line_search(L, LfObj, LfGrad,...
					tau, delta, lineSearchFlag, eta, rankIncrement,...
					rankEstimate, truncateFlag, gap, X, constraintMat,...
					betaVector)

	while(1),
		LfGD = L - 1 / delta * LfGrad;
		[LfGDShrink, LfGDShrinkNorm, rankEstimate] = shrinkage(LfGD,...
			tau / delta, rankIncrement, rankEstimate, truncateFlag, gap);
		if (~lineSearchFlag),
			break;
		end;
		LfGDShrinkfObj = f_func(LfGDShrink, X, constraintMat, betaVector);
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

function [AShrink, AShrinkNorm, rankEstimate] = shrinkage(A, tau, rankIncrement, rankEstimate, truncateFlag, gap)

% TODO: Add delay for truncation in early iterations.
% [AShrink AShrinkNorm rankEstimate] = nuclear_psd_proximal_truncate(A, tau, rankIncrement, rankEstimate, truncateFlag, gap);
[AShrink1 AShrinkNorm1 rankEstimate] = nuclear_psd_proximal(A, tau, rankIncrement, rankEstimate);
% [AShrink2 AShrinkNorm2] = nuclear_psd_proximal_mex(A, tau);
% if (norm(AShrink1-AShrink2, 'fro') > 10^-10)
% 	disp('problem');
% end;
AShrink = AShrink1; AShrinkNorm = AShrinkNorm1;

end

% NOTE: As ANorm is used both by Q and F, we can ignore it in their
% comparison. Q_func_mod does that, and is compared directly with f instead
% of F.
function QObj = Q_func_mod(A, LfGD, LfObj, LfGrad, delta)

% LfGD = L - 1 / delta * LfGrad;
QObj = delta / 2 * norm(A - LfGD, 'fro') ^ 2 + LfObj...
		- 1 / 2 / delta * norm(LfGrad, 'fro') ^ 2;
end

function QObj = Q_func_alt(A, L, LfObj, LfGrad, delta)

QObj = delta / 2 * norm(A - L, 'fro') ^ 2 + trace((A - L)' * LfGrad) + LfObj;

end

function QObj = Q_func(A, ANorm, LfGD, LfObj, LfGrad, delta, tau)

% LfGD = L - 1 / delta * LfGrad;
QObj = delta / 2 * norm(A - LfGD, 'fro') ^ 2 + tau * ANorm + LfObj...
		- 1 / 2 / delta * norm(LfGrad, 'fro') ^ 2;
end

function FObj = F_func(AfObj, ANorm, tau)

FObj = AfObj + tau * ANorm;

end

function [fObj, fGrad] = f_func(A, X, constraintMat, betaVector)

fObj = 0;
fGrad = zeros(size(A));
numConstraints = length(betaVector);
for iterConstraint = 1:numConstraints,
	beta = betaVector(iterConstraint);
	coord1 = constraintMat(iterConstraint, 1);
	coord2 = constraintMat(iterConstraint, 2);
	Dij = (X(:, coord1) - X(:, coord2)) * (X(:, coord1) - X(:, coord2))';
	alpha = trace(AOld * Dij) - beta;
	fObj = fObj + 0.5 * alpha ^ 2;
	if (nargout >= 2),
		fGrad = fGrad + alpha * Dij;
	end;
end;

end
