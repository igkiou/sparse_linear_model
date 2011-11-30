function [aVal aPrime aDoublePrime aVec] = link_func(X, family)

if (nargin < 2),
	family = 'P';
elseif ((family ~= 'P') && (family ~= 'p') && (family ~= 'B') && (family ~= 'b')),
	error('Unknown distribution.');
end;

if ((family == 'P') || (family == 'p')),
	tempVal = exp(X);
	aVec = tempVal;
	aVal = sum(aVec, 1);
	if (nargout >= 2),
		aPrime = tempVal;
	end;
	if (nargout >= 3),
		aDoublePrime = tempVal;
	end;
elseif ((family == 'B') || (family == 'b')),
	tempVal = exp(X);
	aVec = log(1 + tempVal);
	aVal = sum(aVec, 1);
	if (nargout >= 2),
		aPrime = tempVal ./ (tempVal + 1);
	end;
	if (nargout >= 3),
		aDoublePrime = aPrime - aPrime .^ 2;
	end;
end;
