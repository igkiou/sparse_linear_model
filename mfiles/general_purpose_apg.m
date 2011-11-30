function [X, converged, lossFuncObj] = general_purpose_apg(XDims,...
								lossFuncObjGrad, proximalOperator,...
								tauList, tolerance, delta0, numIters,...
								lineSearchFlag, eta)
%	LOSSFUNCOBJGRAD should accept a single argument and return the value of
%	the (smooth part of the) loss function, and its gradient at the input
%	argument.
%	PROXIMALOPERATOR should accept two arguments and return the solution to
%	the proximal problem and its "norm" (or obj. val for proximal problem).

if ((nargin < 5) || isempty(tolerance)),
	tolerance = 0.00000001;
end;

if ((nargin < 6) || isempty(delta0)),
	delta0 = 1000;
end;

if ((nargin < 7) || isempty(numIters)),
	numIters = 50000;
end;
										
if ((nargin < 8) || isempty(lineSearchFlag)),
	lineSearchFlag = 0;
end;
										
if ((nargin < 9) || isempty(eta)),
	eta = 1.1;
end;

numRepeats = length(tauList);
X = zeros(XDims);
XOld = X;
t = 1;
tOld = t;

% eta > 1, as opposed to eta < 1 in Toh (and multiplying by 1 / eta)
% tau ~ mu
% delta ~ tau

converged = 1;
for iterRepeat = 1:numRepeats,
	tau = tauList(iterRepeat);
% 	fprintf('Now running repeat %d out of %d, tau %g.\n', iterRepeat, numRepeats, tau);
	delta = delta0;
	% TODO: change continuation to not change tau, or change mu on the run,
	% or cap delta, or some combination thereof. Same in mex.
% 	t = 1;
% 	tOld = t;
% 	KOld = K;
	for iter = 1:numIters,
% 		if (mod(iter, 10000) == 0),
% 			fprintf('Now running iter %d.\n', iter);
% 		end;
		a = (tOld - 1) / t;
		L = X + a * (X - XOld);
		[LfObj, LfGrad] = lossFuncObjGrad(L);
% 		delta = delta / eta;
% 		delta = delta0;

		[delta, LfGDShrink] = line_search(L, LfObj, LfGrad,...
										tau, delta, lineSearchFlag, eta,...
										lossFuncObjGrad, proximalOperator);
		XOld = X;
		X = LfGDShrink;
				
		if (anynn(X)),
			warning('X has non-numeric (Inf or NaN values');
		end;
		
		tOld = t;
		t = (1 + sqrt(1 + 4 * tOld ^ 2)) / 2;
		
% 		[KfObj, KfGrad] = f_func(L, constraintMat, betaVector);
% 		S = delta * (L - K) + (KfGrad - LfGrad);
		normXOld = norm(XOld, 'fro');
		normX = norm(X, 'fro');
		normXDiff  = norm(XOld - X, 'fro');
% 		if ((norm(S,'fro') > eps) && (norm(S, 'fro') / delta / max(1, norm(K, 'fro')) < tolerance)),
		if ((normX > eps) && (normXDiff / max(1, normXOld) < tolerance)),
			break;
		end;
	end;
end;

if (iter == numIters),
	converged = - 1;
end;

if (nargout >= 3),
	lossFuncObj = lossFuncObjGrad(X);
end;

end

function [delta, LfGDShrink, rankEstimate] = line_search(L, LfObj, LfGrad,...
										tau, delta, lineSearchFlag, eta,...
										lossFuncObjGrad, proximalOperator)

	while(1),
		LfGD = L - 1 / delta * LfGrad;
		LfGDShrink = proximalOperator(LfGD, tau / delta);
		if (~lineSearchFlag),
			break;
		end;
		LfGDShrinkfObj = lossFuncObjGrad(LfGDShrink);
		LQmodObj = Q_func_mod(LfGDShrink, LfGD, LfObj, LfGrad, delta);
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
