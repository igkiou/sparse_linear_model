function [obj1 obj2 obj3] = sensing_objectives(Phi, D)

[m n] = size(Phi);
k = size(D, 2);
Ig = eye(k);
obj1 = norm(D' * (Phi' * Phi) * D - Ig, 'fro') ^ 2;

if (nargout >= 2),
	vec = sqrt(sum((Phi * D) .^ 2));
	Dtild = Phi * D * diag(1 ./ vec);
	Gtild = Dtild' * Dtild;
	obj2 = sum(sum((Gtild - Ig) .^ 2)) / k / (k - 1);
end;

if (nargout >= 3),
	[V L] = eig(D * D');
	Gamma = Phi * V;
	obj3 = norm(L - L * (Gamma' * Gamma) * L,'fro') ^ 2;
end;
