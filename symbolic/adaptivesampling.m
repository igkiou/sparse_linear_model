syms m11 m12 m13 m21 m22 m23 m31 m32 m33
syms m
M = [m11 m12 m13; m21 m22 m23; m31 m32 m33];
dab = M(:,1).'*M(:,1)+M(:,2).'*M(:,2)-2*M(:,1).'*M(:,2);
dac = M(:,1).'*M(:,1)+M(:,3).'*M(:,3)-2*M(:,1).'*M(:,3);
p = (m + dac) / (2 * m + dab + dac);
l = log(1/p);
d = 2 * m + dab + dac;
ddac = - 1 / p * (m + dab) / d ^ 2;
ddab = 1 / p * (m + dac) / d ^ 2;

M = [2 * ddac * (M(:,1) - M(:,3)) + 2 * ddab * (M(:,1) - M(:,2)),...
	2 * ddab * (M(:,2) - M(:,1)), 2 * ddac * (M(:,3) - M(:,1))];

D = [diff(l,m11) diff(l,m12) diff(l,m13);...
	diff(l,m21) diff(l,m22) diff(l,m23);...
	diff(l,m31) diff(l,m32) diff(l,m33)];
