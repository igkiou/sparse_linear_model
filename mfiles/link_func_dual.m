function [aVal aPrime aDoublePrime aVec] = link_func_dual(X, family)

if (nargin < 2),
	family = 'P';
elseif ((family ~= 'P') && (family ~= 'p') && (family ~= 'B') && (family ~= 'b')),
	error('Unknown distribution.');
end;

if ((family == 'P') || (family == 'p')),
	tempVal = log(X);
	aVec = X .* tempVal - X;
	aVal = sum(aVec, 1);
	if (nargout >= 2),
		aPrime = tempVal;
	end;
	if (nargout >= 3),
		aDoublePrime = 1 ./ X;
	end;
elseif ((family == 'B') || (family == 'b')),
	tempVal = log(X);
	tempValSym = log(1 - X);
	aVec = X .* tempVal + (1 - X) .* tempValSym;
	aVal = sum(aVec, 1);
	if (nargout >= 2),
		aPrime = tempVal - tempValSym;
	end;
	if (nargout >= 3),
		aDoublePrime = 1 ./ (X .* (1 - X));
	end;
end;
