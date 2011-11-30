function [W, iter, fvalVec] = apg(Win, lambda, objgradfunc, proximalfunc, numIters, Linit, gamma, epsilon)

if (nargin < 5),
	numIters = 100;
end;

if (nargin < 6),
	Linit = 100;
end;

if (nargin < 7),
	gamma = 1.1;
end;

if (nargin < 8),
	epsilon = 10 ^ -9;
end;

alpha = 1;
L = Linit;
ZOld = Win;
W = ZOld;
[objTemp Zgrad] = objgradfunc(ZOld);
Zgrad = reshape(Zgrad, size(ZOld));
Zinterm = ZOld - 1 / L * Zgrad;
obj = objTemp;
fvalVec = zeros(numIters + 1, 1);
fvalVec(1) = obj;
iter = 1;

while (true),
	[Z normValue] = proximalfunc(Zinterm, lambda / L);
	[Fval Qval] = Qeval(Z, ZOld, Zgrad, objgradfunc, objTemp, normValue, L, lambda);
	while Fval > Qval,
		L = gamma * L;
		Zinterm = ZOld - 1 / L * Zgrad;
		[Z normValue] = proximalfunc(Zinterm, lambda / L);
		[Fval Qval] = Qeval(Z, ZOld, Zgrad, objgradfunc, objTemp, normValue, L, lambda);
	end;
	WOld = W;
	W = Z;
	objOld = obj;
	obj = Fval;
	fvalVec(iter + 1) = obj;
	if ((abs((objOld - obj) / objOld) < epsilon) || (iter > numIters)),
		break;
	end;
	alphaOld = alpha;
	alpha = (1 + sqrt(1 + 4 * alphaOld ^ 2)) / 2;
	ZOld = W + (alphaOld - 1) / alpha * (W - WOld);
	[objTemp Zgrad] = objgradfunc(ZOld);
	Zgrad = reshape(Zgrad, size(ZOld));
	Zinterm = ZOld - 1 / L * Zgrad;
	iter = iter + 1;
end;
fvalVec = fvalVec(1:(1+iter));
	
end

function [Fval Qval] = Qeval(X, Z, Zgrad, objgradfunc, obj, normValue, L, lambda)

Zdiff = X - Z;
Qval = obj + sum(Zdiff(:) .* Zgrad(:)) + L / 2 * sum(Zdiff(:) .^ 2) + lambda * normValue;
Fval = objgradfunc(X) + lambda * normValue;

end

% function [W, iter, fvalVec] = apg(Win, lambda, objgradfunc, proximalfunc, numIters, Linit, gamma, epsilon)
% 
% if (nargin < 5),
% 	numIters = 100;
% end;
% 
% if (nargin < 6),
% 	Linit = 100;
% end;
% 
% if (nargin < 7),
% 	gamma = 1.1;
% end;
% 
% if (nargin < 8),
% 	epsilon = 10 ^ -9;
% end;
% 
% alpha = 1;
% L = Linit;
% ZOld = Win;
% WOld = ZOld;
% [objTemp Zgrad] = objgradfunc(ZOld);
% Zgrad = reshape(Zgrad, size(ZOld));
% Zinterm = ZOld - 1 / L * Zgrad;
% obj = objTemp;
% fvalVec = zeros(numIters + 1, 1);
% fvalVec(1) = obj;
% iter = 1;
% 
% while (true),
% 	[Z normValue] = proximalfunc(Zinterm, lambda / L);
% 	[Fval Qval] = Qeval(Z, ZOld, Zgrad, objgradfunc, objTemp, normValue, L, lambda);
% 	while Fval > Qval,
% 		L = gamma * L;
% 		Zinterm = ZOld - 1 / L * Zgrad;
% 		[Z normValue] = proximalfunc(Zinterm, lambda / L);
% 		[Fval Qval] = Qeval(Z, ZOld, Zgrad, objgradfunc, objTemp, normValue, L, lambda);
% 	end;
% % 	WOld = W;
% % 	W = Z;
% 	objOld = obj;
% 	obj = Fval;
% 	fvalVec(iter + 1) = obj;
% 	if ((abs((objOld - obj) / objOld) < epsilon) || (iter > numIters)),
% 		break;
% 	end;
% 	alphaOld = alpha;
% 	alpha = (1 + sqrt(1 + 4 * alphaOld ^ 2)) / 2;
% 	ZOld = Z + (alphaOld - 1) / alpha * (Z - WOld);
% 	[objTemp Zgrad] = objgradfunc(ZOld);
% 	Zgrad = reshape(Zgrad, size(ZOld));
% 	Zinterm = ZOld - 1 / L * Zgrad;
% 	WOld = Z;
% 	iter = iter + 1;
% end;
% fvalVec = fvalVec(1:(1+iter));
% W = Z;
% end
% 
