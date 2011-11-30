function sc_coeffs = sc_yang(feaSet, Bor, mat, gamma)
%================================================
% 
% Usage:
% Sparse codes for data vectors using given dictionary. 
%
% Inputss:
% feaSet        -local feature array extracted from the
%                image, column-wise
% Bor           -sparse dictionary, column-wise
% mat           -projection matrix (optional)
% gamma         -regularization parameter 
% 
% Output:
% sc_coeffs     -vector of sparse code coefficients
%
%===============================================

dSize = size(Bor, 2);
nSmp = size(feaSet, 2);

sc_coeffs = zeros(dSize, nSmp);

% compute the local feature for each local feature
beta = 1e-4;
if(~isempty(mat)),
	B = mat * Bor;
else
	B = Bor;
end;

A = B'*B + 2*beta*eye(dSize);
Q = -B'*feaSet;

for iter1 = 1:nSmp,
    sc_coeffs(:, iter1) = L1QP_FeatureSign_yang(gamma, A, Q(:, iter1));
end;
