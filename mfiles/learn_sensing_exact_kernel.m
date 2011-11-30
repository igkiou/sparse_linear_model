function Phi = learn_sensing_exact_kernel(D, m, initPhi, gramMatrix, varargin)
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
[n k] = size(D);

% precalculate eigendecomposition and rank of D*D'

if (nargin < 4),
	gramMatrix = [];
end;

kernelType = 'g';
kernelParam1 = 1;
kernelParam2 = 1;
if (nargin > 4),
	kernelType = varargin{1};
	if ((length(varargin) > 1) && strcmp(class(varargin{2}), 'double')), 
		kernelParam1 = varargin{2}; 
	end;
	if ((length(varargin) > 2) && strcmp(class(varargin{3}), 'double')), 
		kernelParam2 = varargin{3}; 
	end;
end;

if (isempty(gramMatrix)),
	gramMatrix = kernel_gram_mex(D, [], kernelType, kernelParam1, kernelParam2);
end;
	
DDt = gramMatrix ^ 2;
[V L] = eig(DDt);

% order eigenvalues and eigenvectors
[lvec indexes] = sort(diag(L), 1, 'descend');
V = V(:,indexes);

% find rank of D*D'
tol = n * eps(lvec(1));
r = sum(lvec > tol);
lvec(r+1:end) = 0;
% L = sparse(diag(lvec));

Gamma = [diag(sqrt(lvec(1:m))) zeros(m, k - m)];
Phi = Gamma * diag(safeReciprocal(lvec, 0)) * V';

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
