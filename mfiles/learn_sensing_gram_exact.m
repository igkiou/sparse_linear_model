function Phi = learn_sensing_gram_exact(D, m, initPhi)
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

% precalculate eigendecomposition and rank of D*D'
DDt = D * transpose(D);
[V L] = eig(DDt);

% precalculate eigendecomposition and rank of D*D'
DDt = D * transpose(D);
[V L] = eig(DDt);

% order eigenvalues and eigenvectors
[lvec indexes] = sort(diag(L), 1, 'descend');
V = V(:,indexes);

% find rank of D*D'
tol = n * eps(lvec(1));
r = sum(lvec > tol);
lvec(r+1:end) = 0;
% L = sparse(diag(lvec));

Gamma = [eye(m) zeros(m, n - m)];
Phi = Gamma * diag(sqrt(safeReciprocal(lvec, 0))) * V';

if (r < m)
	if ((nargin >= 3) && ~isempty(initPhi)),
		if (size(initPhi, 1) ~= m) || (size(initPhi, 2) ~= n),
			error('Invalid initialization matrix.');
		end;
	else
		initPhi = random_sensing(n, m);
	end;
	Phi((r + 1):end, :) = initPhi((r + 1):end, :);
end;
