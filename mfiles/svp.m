function [D, iter] = svp(Din, minrank, objgradfunc, hardthreshfunc,...
			numIters, etainit, gamma, epsilon, maxtol)

if (nargin < 5),
	numIters = 100;
end;

if (nargin < 6),
	etainit = 100;
end;

if (nargin < 7),
	gamma = 1.1;
end;

if (nargin < 8),
	epsilon = 10 ^ - 9;
end;

if (nargin < 9),
	maxtol = 10 ^ - 4;
end;

eta = etainit;
D = Din;
obj = objgradfunc(D);
iter = 1;

while(1),
	[foo Dgrad] = objgradfunc(D);	
	Dinterm = D;
	objOld = obj;
	D = D - eta * Dgrad;
	D = hardthreshfunc(D, minrank);
	obj = objgradfunc(D);	
	
	while((obj - objOld) / objOld > maxtol),
		D = Dinterm;
		eta = eta * gamma;
		D = D - eta * Dgrad;
		D = hardthreshfunc(D, minrank);
		obj = objgradfunc(D);	
	end;
	if (((abs(objOld - obj) / objOld) < epsilon) || (iter > numIters)),
		break;
	end;
	iter = iter + 1;
end;
