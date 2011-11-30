function [obj deriv] = l1exp_obj_subgrad(s, Dt, x, lambda, family)

if (nargin < 5),
	family = 'P';
end;

K = size(s, 1);
N = size(x, 1);

if (size(Dt, 1) ~= K),
	error('First dimension of transposed dictionary does not match sparse code dimension.');
elseif (size(Dt, 2) ~= N),
	error('Second dimension of transposed dictionary does not match signal dimension.');
end;

Ds = Dt' * s;
if (nargout >= 2),
	[aVal aPrime] = link_func(Ds, family);
else
	aVal = link_func(Ds, family);
end;
obj = aVal - x' * Ds + lambda * sum(abs(s));

if (nargout >= 2),
	deriv = Dt * (x - aPrime) + lambda * sign(s);
end;
