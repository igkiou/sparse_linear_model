function [Xr normValue] = group_lasso_proximal(X, tau, indexSets, setWeights)

if ((nargin < 4) || (isempty(setWeights))),
	setWeights = ones(size(indexSets));
end;

Xr = zeros(size(X));
numSets = length(indexSets);
normValue = 0;
for iterSet = 1:numSets,
	g = indexSets{iterSet};
	Xg = X(g);
	XgNorm = sqrt(sum(Xg .^ 2));
	weightSet = setWeights(iterSet);
	tauSet = tau * weightSet;
	if ((XgNorm < eps) || (XgNorm - tauSet < 0)),
		Xr(g) = 0;
	else
		Xr(g) = (XgNorm - tauSet) * Xg / XgNorm;
		normValue = normValue + (XgNorm - tauSet);
	end;
end;

% if (nargout >= 2),
% 	normValue = 0;
% 	for iterSet = 1:numSets,
% 		g = indexSets{iterSet};
% 		normValue = normValue + sqrt(sum(Xr(g) .^ 2));
% 	end;
% end;
