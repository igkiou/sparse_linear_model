function Phi = learn_sensing(D, m, initPhi)
% learn sensing matrix based on given dictionary
%
%
%	Inputs
%	D: dictionary, n x k matrix, where n is the dimension of the original
%	   signal and k is the number of atoms in the dictionary.
%	m: number of dimensions in compressed signal.
%	initPhi: matrix to be used for the initialization of Phi (optional). If
%	it is not provided, then Phi is initialized as a random Gaussian
%	matrix.
%
%	Outputs
%	Phi: learnt sensing matrix, of dimension m x n.

% original signal size
n = size(D, 1);

% initialize sensing matrix
if ((nargin >= 3) && ~isempty(initPhi)),
	if (size(initPhi, 1) ~= m) || (size(initPhi, 2) ~= n),
		error('Invalid initialization matrix.');
	end;
	Phi = initPhi;
else
	Phi = random_sensing(n, m);
end;

% not sure if it is required, tests show that probably not
% vec=sqrt(sum((Phi *D).^2));
% D = D * diag(1 ./ vec);

% precalculate eigendecomposition and rank of D*D'
DDt = D * transpose(D);
[V L] = eig(DDt);

% order eigenvalues and eigenvectors
[lvec indexes] = sort(diag(L), 1, 'descend');
V = V(:,indexes);
L = sparse(diag(lvec));

% find rank of D*D'
tol = n * eps(lvec(1));
r = sum(lvec > tol);

Gamma = Phi * V;

% pre-allocate everything
% Vmat = zeros(m, n);
% sumv = zeros(n, n);
% E = zeros(n, n);
% Ej = zeros(n, n);
% U = zeros(n, n);
% Xi = zeros(n, n);
% xi = 0;
% index = 0;
% u = zeros(n, 1);

% run iteration
for j = 1:m,
% 	Vmat = bsxfun(@times, lvec, Gamma');
	Vmat = L * Gamma';
	sumv = Vmat * Vmat';
	E = L - sumv;
	Ej = E + Vmat(:, j) * Vmat(:, j)';
	[U Xi] = eig(Ej);
	Xi = diag(Xi);
	[xi index] = max(Xi);
	u = U(:, index);
	Gamma(j, 1:r) = sqrt(xi) * u(1:r) ./ lvec(1:r);
end;

% final sensing matrix estimate
Phi = Gamma * transpose(V);
