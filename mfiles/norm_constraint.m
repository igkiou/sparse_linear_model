function [C Ceq Cgrad Ceqgrad] = norm_constraint(X, M, N)

Phi = reshape(X, [M N]);
C = sum(Phi.^2,2) - 1;
Ceq = 0;

if (nargout > 2),
	Cgrad = zeros(numel(X), M);
	for iter = 1:M,
		Cgrad((N * (iter - 1) + 1):(N * iter), iter) = Phi(iter, :)';
	end;
	Cgrad = 2* Cgrad;
	Ceqgrad = zeros(numel(X), 1);
end;
