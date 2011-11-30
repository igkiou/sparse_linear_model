function [X, converged, lossFuncObj] = general_purpose_svp(XDims,...
								lossFuncObjGrad, hardThresholdingOperator,...
								rank, tolerance, delta, numIters)
%	LOSSFUNCOBJGRAD should accept a single argument and return the value of
%	the (smooth part of the) loss function, and its gradient at the input
%	argument.
%	HARDTHRESHOLDINGOPERATOR should accept two arguments and return the
%	hard thresholded version of the first input and its "norm" (or obj. val
%	for proximal problem). 

if ((nargin < 5) || isempty(tolerance)),
	tolerance = 10 ^ - 5;
end;

if ((nargin < 6) || isempty(delta)),
	delta = 0.1;
end;

if ((nargin < 7) || isempty(numIters)),
	numIters = 50000;
end;

X = zeros(XDims);
XOld = X;

converged = 1;
for iter = 1:numIters,
	if (mod(iter, 10000) == 0),
		fprintf('Now running iter %d.\n', iter);
	end;
	XOld = X;
	normXOld = norm(XOld, 'fro');
	[foo XGrad] = lossFuncObjGrad(X);
	X = XOld - delta * XGrad;
	X = hardThresholdingOperator(X, rank);

	normX = norm(X, 'fro');
	normXDiff  = norm(XOld - X, 'fro');
	if ((normX > eps) && (normXDiff / max(1, normXOld) < tolerance)),
		break;
	end;
end;	

if (iter == numIters),
	converged = - 1;
end;

if (nargout >= 3),
	lossFuncObj = lossFuncObjGrad(X);
end;
