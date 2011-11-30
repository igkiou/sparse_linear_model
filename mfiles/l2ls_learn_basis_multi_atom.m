function D = l2ls_learn_basis_multi_atom(X, L, A, D, incoherencelambda, l2norm, numiters)
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

reguB = 10 ^ -4;
Ig = eye(signalsize);
for iter = 1:numiters,
	disp(sprintf('Inner iteration %d / %d begins\n', iter, numiters)); %#ok
	classinds = randperm(numclasses);
	for iterclass = 1:numclasses,
		classSamples = L == classinds(iterclass);
		Xclass = X(:, classSamples);
		Aclass = A(:, classSamples);
		XAt = Xclass * Aclass';
		AAt = Aclass * Aclass';
		Dclass = D(:, ((classinds(iterclass) - 1) * dictsize + 1):(classinds(iterclass) * dictsize));
		DDt = DDt - Dclass * Dclass';
		atominds = randperm(dictsize);
		if (incoherencelambda ~= 0),
			for iteratom = 1:dictsize,
				diter = (DDt + (AAt(atominds(iteratom), atominds(iteratom)) + reguB) * Ig) \ ...
					(XAt(:, atominds(iteratom)) - Dclass * AAt(:, atominds(iteratom)) + ...
					AAt(atominds(iteratom), atominds(iteratom)) * Dclass(:, atominds(iteratom)));
				diter = diter * safeReciprocal(norm(diter), 1) * l2norm;
				Dclass(:, atominds(iteratom)) = diter;
			end;
		else
			for iteratom = 1:dictsize,
				diter = (XAt(:, atominds(iteratom)) - Dclass * AAt(:, atominds(iteratom))) / ...
					AAt(atominds(iteratom), atominds(iteratom)) + Dclass(:, atominds(iteratom));
				diter = diter / norm(diter) * l2norm;
				Dclass(:, atominds(iteratom)) = diter;
			end;
		end;
		D(:, ((classinds(iterclass) - 1) * dictsize + 1):(classinds(iterclass) * dictsize)) = Dclass;
		DDt = DDt + ...
			D(:, ((classinds(iterclass) - 1) * dictsize + 1):(classinds(iterclass) * dictsize)) * ...
			D(:, ((classinds(iterclass) - 1) * dictsize + 1):(classinds(iterclass) * dictsize))';
	end;
end;
