function A = group_lasso_ista(X, D, lambda, indexSets, setWeights, ...
							tolerance, delta0, numIters,...
							continuationFlag, varargin)
%	LOSSFUNCOBJGRAD should accept a single argument and return the value of
%	the (smooth part of the) loss function, and its gradient at the input
%	argument.
%	PROXIMALOPERATOR should accept two arguments and return the solution to
%	the proximal problem and its "norm" (or obj. val for proximal problem).

if ((nargin < 5) || (isempty(setWeights))),
	setWeights = ones(size(indexSets));
end;

if ((nargin < 6) || isempty(tolerance)),
	tolerance = 0.00000001;
end;

if ((nargin < 7) || isempty(delta0)),
	delta0 = 0.001;
end;

if ((nargin < 8) || isempty(numIters)),
	numIters = 50000;
end;
										
if ((nargin < 9) || isempty(continuationFlag)),
	continuationFlag = 0;
end;

if (continuationFlag == 1),
	lambdaList = getContinuationList(lambda, varargin{:});
else
	lambdaList = lambda;
end;

[N numSamples] = size(X);
K = size(D, 2);
if (size(D, 1) ~= N),
	error('Size of dictionary does not agree with size of samples.');
end;

if (delta0 < 0),	
	opts.disp = 0;
	delta0 = 2 * eigs(D'*D, 1, 'LA', opts);
end;

A = zeros(K, numSamples);
ADims = [K 1];
KXX = X' * X;
KDX = D' * X;
KDD = D' * D;
proximalOperator = @(s, L) group_lasso_proximal(s, L, indexSets, setWeights);
for iterSample = 1:numSamples,
% 	lossFuncObjGrad = @(a) qp_obj_grad(a, X(:, iterSample), D);
	lossFuncObjGrad = @(a) kernel_obj_grad(a, KXX(iterSample, iterSample),...
											KDX(:, iterSample), KDD);
    A(:, iterSample) = general_purpose_fpc(ADims,...
								lossFuncObjGrad, proximalOperator,...
								lambdaList, tolerance, delta0, numIters);
end;
