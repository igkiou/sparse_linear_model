function [obj deriv] = eig_regression_obj_grad(X, DDt, DDt2, VL, L, AtA, S, mu)
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
m = numel(X) / n;
Phi = reshape(X, [m n]);

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

Gamma = Phi * VL;
PhiS = Phi * S;
obj = norm(L - Gamma' * Gamma, 'fro') ^ 2 + mu * norm(AtA - PhiS' * PhiS, 'fro') ^ 2;

% Gamma = Phi * V * L;
% obj = norm(L - Gamma' * Gamma, 'fro') ^ 2;

if (nargout >= 2),
% 	deriv = 4 * Phi * DDt2 * (Phi' * Phi * DDt - Ig) * DDt;
%	deriv = 4 * Phi * (PhiDDt2' * PhiDDt2 - DDt3);
%	deriv = 4 * Phi * ((PhiDDt2' * PhiDDt2) - DDt3);
	PhiDDt2 = Phi * DDt2;
	deriv = 4 * PhiDDt2 * (Phi' * PhiDDt2 - DDt) + mu * 4 * PhiS * (PhiS' * PhiS - AtA) * S';
	deriv = deriv(:);
end;
