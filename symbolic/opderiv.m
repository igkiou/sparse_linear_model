M = 2;
N = 3;
F = 2;
K = 2;
S = 2;
syms a11 a21 a12 a22
A = [a11 a12;a21 a22];
syms b11 b21 b31 b41 b12 b22 b32 b42
B = [b11 b12;b21 b22;b31 b32;b41 b42];
B1 = reshape(B(:,1),[M F]);
B2 = reshape(B(:,2),[M F]);
syms x11 x21 x31 x41 x51 x61 x12 x22 x32 x42 x52 x62
X = [x11 x12;x21 x22;x31 x32;x41 x42;x51 x52;x61 x62];
X1 = reshape(X(:,1),[M N]);
X2 = reshape(X(:,2),[M N]);
syms w1 w2 w3 w4 w5 w6
W = [w1 w4 w6; w4 w2 w5; w6 w5 w3];
Wsq = W * W;

syms y11 y21 y31 y12 y22 y32
Y = [y11 y12; y21 y22; y31 y32];

numSamples = S;
L = 1 / 2 * trace(transpose((X1 - B1 * transpose(Y) * a11 - B2 * transpose(Y) * a21) * W) * ((X1 - B1 * transpose(Y) * a11 - B2 * transpose(Y) * a21) * W))...
	+ 1 / 2 * trace(transpose((X2 - B1 * transpose(Y) * a12 - B2 * transpose(Y) * a22)* W) * ((X2 - B1 * transpose(Y) * a12 - B2 * transpose(Y) * a22)* W));
L = L / numSamples;

for iterK = 1:K,
	BYt(:, iterK) = vec(reshape(B(:, iterK), [M F]) * Y.');
end;
Ds = 1 / numSamples * (BYt * A - X);
for iterS = 1:numSamples,
	D(:, iterS) = vec(reshape(Ds(:, iterS), [M N]) * Wsq * Y);
end;

for iterK = 1:K,
	G(:, iterK) = vec(reshape(D * A(iterK, 1:numSamples).', [M F]));
end;

error = [simplify(diff(L,B(1,1))-G(1,1)) simplify(diff(L,B(1,2))-G(1,2));
	simplify(diff(L,B(2,1))-G(2,1)) simplify(diff(L,B(2,2))-G(2,2));
	simplify(diff(L,B(3,1))-G(3,1)) simplify(diff(L,B(3,2))-G(3,2));
	simplify(diff(L,B(4,1))-G(4,1)) simplify(diff(L,B(4,2))-G(4,2));];
