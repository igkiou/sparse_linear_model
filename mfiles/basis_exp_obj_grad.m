function [objValue gradValue] = basis_exp_obj_grad(D, X, S, family)

if (nargin < 4),
	family = 'p';
end;

[N numSamples] = size(X);
K = size(S, 1);

D = reshape(D, [N K]);
DS = D * S;

if (nargout >= 2),
	gradValue = zeros(size(D));
	[aVal aPrime] = link_func(DS, family);
	aPrimeX = aPrime - X;
else
	aVal = link_func(DS, family);
end;

objValue = sum(aVal - sum(X .* DS));

if (nargout >= 2),
	for iter = 1:numSamples,
		gradValue = gradValue + aPrimeX(:, iter) * transpose(S(:, iter));
	end;
	gradValue = gradValue(:);
end;
