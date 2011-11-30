ai = 10;
a = ai;
z = randn(5,1);
y = randn(5,1);
X = randn(5,5); X = X' * X;
Xt = X;
p = z'*X*z;
q = y'*X*y;
r = z'*X*y;
A = z*z'-y*y';

X1 = inv(inv(X)+ai*A);
X2 = Xt...
	- ai/(1+ai*p)*Xt*z*z'*Xt...
	+ ai*(1+ai*p)/((1-ai*q)*(1+ai*p)-(ai*r)^2)*Xt*y*y'*Xt...
	- ai^2/((1-ai*q)*(1+ai*p)-(ai*r)^2)*Xt*z*y'*Xt*r...
	- ai^2/((1-ai*q)*(1+ai*p)-(ai*r)^2)*Xt*y*z'*Xt*r...
	+ ai^3/((1-ai*q)*(1+ai*p)-(ai*r)^2)/(1+ai*p)*Xt*z*z'*Xt*r^2;
X3 = Xt...
	+ ai * (1 + ai * p) / ((1 + a * p) * (1 - a * q) + a ^ 2 * r ^ 2) * X * y * y' * X...
	- ai * (1 - ai * q) / ((1 + a * p) * (1 - a * q) + a ^ 2 * r ^ 2) * X * z * z' * X...
	- ai ^ 2 * r / ((1 + a * p) * (1 - a * q) + a ^ 2 * r ^ 2) * (X * y * z' * X + X * z * y' * X);

f = ai / ((1 + a * p) * (1 - a * q) + a ^ 2 * r ^ 2);

Tr1 = trace(X1 * A);
Tr2 = trace(X2 * A);
Tr3 = trace(X3 * A);
Tr4 = p - q + f * (1 + a * p) * (r ^ 2 - q ^ 2)...
			+ f * (1 - a * q) * (r ^ 2 - p ^ 2)...
			- 2 * f * a * r ^ 2 * (p - q);
Tr5 = (2 * a * r  ^ 2 - q - p * ( 2 * a * q - 1))...
	/ ((1 + a * p) * (1 - a * q) + a ^ 2 * r ^ 2);
		
b = 1;
Delta = b ^ 2 * ((p + q) ^ 2 - 4 * r ^ 2) + 4 * (r ^ 2 - p * q) ^ 2;
a1 = (b * p - (b + 2 * p) * q + 2 * r  ^ 2 + sqrt(Delta))/ (2 * b * (p * q - r ^ 2));
a2 = (b * p - (b + 2 * p) * q + 2 * r  ^ 2 - sqrt(Delta))/ (2 * b * (p * q - r ^ 2));
