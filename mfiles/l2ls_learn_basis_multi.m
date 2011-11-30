function D = l2ls_learn_basis_multi(X, L, A, D, incoherencelambda, l2norm, numiters)
% Learning basis using Lagrange dual (with basis normalization)
%
% This code solves the following problem:
% 
%    minimize_B   0.5*||X - B*S||^2
%    subject to   ||B(:,j)||_2 <= l2norm, forall j=1...size(S,1)
% 
% The detail of the algorithm is described in the following paper:
% 'Efficient Sparse Codig Algorithms', Honglak Lee, Alexis Battle, Rajat Raina, Andrew Y. Ng, 
% Advances in Neural Information Processing Systems (NIPS) 19, 2007
%
% Written by Honglak Lee <hllee@cs.stanford.edu>
% Copyright 2007 by Honglak Lee, Alexis Battle, Rajat Raina, and Andrew Y. Ng

signalsize = size(X, 1);
dictsize = size(A, 1);
numclasses = length(D) / dictsize;
DDt = zeros(signalsize, signalsize);

for iterclass = 1:numclasses,
	DDt = DDt + ...
		D(:, ((iterclass - 1) * dictsize + 1):(iterclass * dictsize)) * ...
		D(:, ((iterclass - 1) * dictsize + 1):(iterclass * dictsize))';
end;

for iter = 1:numiters,
	disp(sprintf('Inner iteration %d / %d begins\n', iter, numiters)); %#ok
	classinds = randperm(numclasses);
	for iterclass = 1:numclasses,
		classSamples = L == classinds(iterclass);
		Xclass = X(:, classSamples);
		Aclass = A(:, classSamples);
		DDt = DDt - ...
			D(:, ((classinds(iterclass) - 1) * dictsize + 1):(classinds(iterclass) * dictsize)) * ...
			D(:, ((classinds(iterclass) - 1) * dictsize + 1):(classinds(iterclass) * dictsize))';
		if (incoherencelambda ~= 0),
			Dclass = lyap(incoherencelambda * DDt, Aclass * Aclass', - Xclass * Aclass');
		else
			Dclass = Xclass / Aclass;
		end;
		Dclass = normcols(Dclass) * l2norm;
		D(:, ((classinds(iterclass) - 1) * dictsize + 1):(classinds(iterclass) * dictsize)) = Dclass;
		DDt = DDt + ...
			D(:, ((classinds(iterclass) - 1) * dictsize + 1):(classinds(iterclass) * dictsize)) * ...
			D(:, ((classinds(iterclass) - 1) * dictsize + 1):(classinds(iterclass) * dictsize))';
	end;
end;
