function [labelbest energybest] = knkmeans(K,m,numiters,numreplicates)
%% Written by Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.
% K: kernel matrix
% m: k (1 x 1) or label (1 x n, 1<=label(i)<=k)
% reference: [1] Kernel Methods for Pattern Analysis
% by John Shawe-Taylor, Nello Cristianini
if (nargin < 3),
	numiters = 1000;
end;
if (nargin < 4),
	numreplicates = 1000;
end;
if (numreplicates <= 0),
	numreplicates = 1;
end;
n = size(K,1);
energybest = Inf;

for iterR = 1:numreplicates,
	if max(size(m)) == 1
		k = m;
	%     label = randi(k,1,n);
		label = ceil(k*rand(1,n));
	elseif size(m,1) == 1 && size(m,2) == n
		k = max(m);
		label = m;
	else
		error('ERROR: m is not valid.');
	end

	last = 0;
	S = repmat((1:k)',1,n);
	iter = 0;
	while (any(label ~= last) && (iter < numiters)),
		iter = iter + 1;
		[foo1,foo2,label] = unique(label); % remove empty clusters
		E = double(bsxfun(@eq,S,label));
		E = bsxfun(@rdivide,E,sum(E,2));
		T = E*K;
		Z = repmat(diag(T*E'),1,n)-2*T;

		last = label;
		[f, label] = min(Z);
		energy = sum(f);
	end
	if energy < energybest,
		energybest = energy;
		labelbest = label;
	end;
end;
	
