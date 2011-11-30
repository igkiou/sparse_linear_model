function [X, converged, lossFuncObj] = general_purpose_fpc(XDims,...
								lossFuncObjGrad, proximalOperator,...
								tauList, tolerance, delta, numIters)
%	LOSSFUNCOBJGRAD should accept a single argument and return the value of
%	the (smooth part of the) loss function, and its gradient at the input
%	argument.
%	PROXIMALOPERATOR should accept two arguments and return the solution to
%	the proximal problem and its "norm" (or obj. val for proximal problem).

if ((nargin < 5) || isempty(tolerance)),
	tolerance = 0.00000001;
end;

if ((nargin < 6) || isempty(delta)),
	delta = 1000;
end;

if ((nargin < 7) || isempty(numIters)),
	numIters = 50000;
end;

X = zeros(XDims);
XOld = X;
numRepeats = length(tauList);

converged = 1;
for iterRepeat = 1:numRepeats,
	tau = tauList(iterRepeat);
% 	fprintf('Now running repeat %d out of %d, tau %g.\n', iterRepeat, numRepeats, tau);
	for iter = 1:numIters,
% 		if (mod(iter, 10000) == 0),
% 			fprintf('Now running iter %d.\n', iter);
% 		end;
		XOld = X;
		normXOld = norm(XOld, 'fro');
		[foo XGrad] = lossFuncObjGrad(XOld);
		X = XOld - delta * XGrad;
		X = proximalOperator(X, tau * delta);
		if (anynn(X)),
			warning('X has non-numeric (Inf or NaN values');
		end;

		normX = norm(X, 'fro');
		normXDiff  = norm(XOld - X, 'fro');
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
