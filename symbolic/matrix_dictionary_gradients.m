syms d11 d12 d13 d14 d21 d22 d23 d24
syms k1 k2 k3 k4
K = [k1 k3; k2 k4];
d1 = [d11 d13; d12 d14];
d2 = [d21 d23; d22 d24];
D = [d1(:) d2(:)];
syms m11 m12 m13 m14 m21 m22 m23 m24
m1 = [m11 m13; m12 m14];
m2 = [m21 m23; m22 m24];
M = [m1(:) m2(:)];
syms a11 a12 a21 a22
a1 = [a11; a12];
a2 = [a21; a22];
A = [a1 a2];
L1 = trace((M - D * A).' * (M - D * A));
L2 = trace((m1 - d1 * a11 - d2 * a12).'*(m1 - d1 * a11 - d2 * a12)) +...
	trace((m2 - d1 * a21 - d2 * a22).'*(m2 - d1 * a21 - d2 * a22));
dL1 = -2*(M - D * A) * A.';

simplify(diff(L1,d11)-dL1(1,1))
simplify(diff(L2,d12)-dL1(2,1))
simplify(diff(L2,d23)-dL1(3,2))

d1t = d1 * K;
d2t = d2 * K;
Dt = [d1t(:) d2t(:)];
dL1t = -2 * (M - Dt * A) * A.';
dL1tf = [vec(reshape(dL1t(:, 1), [2 2]) * K.') vec(reshape(dL1t(:, 2), [2 2]) * K.')] + 2 * [vec(d1) vec(d2)];
L1t = trace((M - Dt * A).' * (M - Dt * A)) + sum(d1(:) .^ 2) + sum(d2(:) .^ 2);

simplify(diff(L1t, d23)-dL1tf(3, 2))
simplify(diff(L1t, d12)-dL1tf(2, 1))
