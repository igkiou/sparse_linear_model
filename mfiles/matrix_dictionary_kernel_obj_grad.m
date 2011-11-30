function [obj deriv] = matrix_dictionary_kernel_obj_grad(D, X, A, Ksq, mu)

[MN numSamples] = size(X);
N = size(Ksq, 1);
M = MN / N;
K = size(A, 1);

Dt = zeros(size(D));
Dtemp = zeros(M, N);
for iterK = 1:K,
	Dtemp = reshape(D(:, iterK), [M N]) * Ksq;
	Dt(:, iterK) = Dtemp(:);
end;

res = Dt * A - X;
obj = norm(res, 'fro') ^ 2 / numSamples / 2.0 + norm(D, 'fro') ^ 2 * mu / 2.0;

if (nargout >= 2),
	deriv = res * A' / numSamples;
	for iterK = 1:K,
		Dtemp = reshape(deriv(:, iterK), [M N]) * Ksq';
		deriv(:, iterK) = Dtemp(:);
	end;
	deriv = deriv + mu * D;
end;
