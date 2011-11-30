function A = LegoAlg(C, X, A_0, params)
% A = LegoAlg(C, X, A_0, params)
%
% Core LEGO learning algorithm
% 
% C: 3 column matrix
%      column 1, 2: index of constrained points.  Indexes between 1 and n
%      column 3: target distance
%
% X: (n x m) data matrix - each row corresponds to a single instance
%
% A_0: (m x m) regularization matrix
%
% params: algorithm parameters - see see SetDefaultParams for defaults
%           params.thresh: algorithm convergence threshold
%           params.gamma: gamma value for slack variables
%           params.max_iters: maximum number of iterations
%
% returns A: learned Mahalanobis matrix

tol = params.thresh;
eta = params.eta;
max_iters = params.max_iters;

% check to make sure that no 2 constrained vectors are identical
valid = ones(size(C,1),1);
for (i=1:size(C,1)),
   i1 = C(i,1);
   i2 = C(i,2);
   v = X(i1,:)' - X(i2,:)'; 
   if (norm(v) < 10e-10),
      valid(i) = 0;
   end
end
C = C(valid>0,:);

i = 1;
il = 0;
iter = 0;
c = size(C, 1);
conv = Inf;
A = A_0;
AOld = A;
while (true),
	i1 = C(i,1);
	i2 = C(i,2);
	yt = C(i,3);
	v = X(i1,:)' - X(i2,:)';
	Av = A * v;
	ythat = v' * Av;
	ybar = (eta * yt * ythat - 1 ...
	+ sqrt((eta * yt * ythat - 1) ^ 2 + 4 * eta * ythat ^ 2))...
	/ (2 * eta * ythat);
	beta = - eta * (ybar - yt) / (1 + eta * (ybar - yt) * ythat);
	A = A  + beta * (Av * Av');

	il = il + 1;
	if i == c
		normOld = norm(AOld, 'fro');
		normDiff = norm(A - AOld, 'fro');
		conv = normDiff / normOld;
		iter = iter + 1;
		if (mod(iter, 5000) == 0),       
% 			disp(sprintf('lego iter: %d, conv = %f', iter, conv));
		end;
		if (conv < tol || iter > max_iters),
			break;
		end;
		AOld = A;
	end
	i = mod(i,c) + 1;
end
% disp(sprintf('lego converged to tol: %f, iter: %d loop iters: %d', conv, iter, il));



