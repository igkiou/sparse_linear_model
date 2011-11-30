function [obj deriv] = mahalanobis_regression_obj_grad(X, DDt, DDt2, VL, L, AtA, S, mu)
% objective function value and gradient of sensing matrix learning routine,
% for objective function ||L - L * Gamma' * Gamma * L||_fro ^ 2.
%
%	Note: Inputs are chosen to optimize performance. Optional arguments
%	should be precalculated before the optimization algorithm begins
%	running and provided, to significantly reduce running speed.
%
%	Inputs
%	X: vectorized form of sensing matrix.
%	DDt: D * D', where D is the dictionary.
%	DDt2: DDt ^ 2 (optional).
%	VL: V * L, see below (optional).
%	L: matrix L, see below (optional). L should be stored as sparse for
%	optimal speed.
%	In the above, [V, L] are the outputs of eig(DDt). 
%
%	Outputs
%	obj: value of objective function.
%	deriv: vectorized form of gradient.
%

% original signal size
n = size(DDt, 1);
M = reshape(X, [n n]);

if (nargin < 3),
	DDt2 = DDt ^ 2;
end;

if (nargin < 5),
	[V L] = eig(DDt);
	VL = V * L;
end;

% obj = trace(DDt^2)-2*trace(Phi*DDt^3*Phi')+trace(Phi*DDt^2*(Phi'*Phi)*DDt^2*Phi');
% DDt2PtP = DDt^2*PtP;
% obj2 = norm(DDt, 'fro')^2+trace((-2*DDt+DDt2PtP)*DDt2PtP);
% GtG = V' * PtP * V;

GtG = VL' * M * VL;
StLtLS = S' * M * S;
obj = norm(L - GtG, 'fro') ^ 2 + mu * norm(AtA - StLtLS, 'fro') ^ 2;

% Gamma = Phi * V * L;
% obj = norm(L - Gamma' * Gamma, 'fro') ^ 2;

if (nargout >= 2),
% 	deriv = 4 * Phi * DDt2 * (Phi' * Phi * DDt - Ig) * DDt;
%	deriv = 4 * Phi * (PhiDDt2' * PhiDDt2 - DDt3);
%	deriv = 4 * Phi * ((PhiDDt2' * PhiDDt2) - DDt3);
	deriv = 2 * DDt2 * (M * DDt2 - DDt);
	deriv = deriv(:);
end;
