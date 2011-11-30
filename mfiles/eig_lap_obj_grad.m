function [obj deriv objlap objorth derivlap derivorth] = eig_lap_obj_grad(Phi, M, XLXt, numSamples, DDt, DDt2, VL, L, alphaReg, betaReg)
% objective function value and gradient of sensing matrix learning routine,
% for objective function ||L - L * Gamma' * Gamma * L||_fro ^ 2.
%
%	Note: Inputs are chosen to optimize performance. Optional arguments
%	should be precalculated before the optimization algorithm begins
%	running and provided, to significantly reduce running speed.
%
%	Inputs
%	Phi: vectorized form of sensing matrix.
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
N = size(XLXt, 1);
Phi = reshape(Phi, [M N]);

if ((nargin < 5) && (betaReg > 0)),
	DDt2 = DDt ^ 2;
end;

if ((nargin < 7) && (betaReg > 0)),
	[V L] = eig(DDt);
	VL = V * L;
end;

Gamma = Phi * VL;
PhiXLXt = Phi * XLXt;
objlap = alphaReg * trace(PhiXLXt * Phi') / numSamples ^ 2;
objorth = betaReg * norm(L - Gamma' * Gamma, 'fro') ^ 2;
obj = objlap + objorth;

if (nargout >= 2),
	PhiDDt2 = Phi * DDt2;
	derivlap = alphaReg * 2 * PhiXLXt / numSamples ^ 2;
	derivorth = betaReg * 4 * PhiDDt2 * (Phi' * PhiDDt2 - DDt);
	deriv = derivlap + derivorth;
	
	deriv = deriv(:);
end;
