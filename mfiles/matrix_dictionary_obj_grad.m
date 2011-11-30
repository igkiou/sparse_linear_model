function [obj deriv] = matrix_dictionary_obj_grad(D, X, A, mu)

numSamples = size(X, 2);

res = D * A - X;
obj = norm(res, 'fro') ^ 2 / numSamples / 2.0 + norm(D, 'fro') ^ 2 * mu / 2.0;

if (nargout >= 2),
	deriv = res * A' / numSamples + mu * D;
end;
