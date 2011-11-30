function [Phi, weights, bias, stdvec] = backpropagation_orth_lbfgs(initPhi, initweights, initbias, origData, Y, stdvec)

%
% Input:
%        origData --- d x N matrix, n data points in a d-dim space
%        Y --- 1 x N matrix, labels of n data points -1 or 1

usebias = 1;
lossfun = 'huber';
[M N] = size(initPhi); 
numSamples = size(origData, 2);
lambda = 0.1;
orthlambda = 0.01;
gradsvmfun = 'huberhinge_obj_grad_mex';
gradphifun = 'orth_huber_obj_grad';

% w0 = zeros(D,1);
% b0 = 0;

wolfe = struct('a1',0.5,'a0',0.01,'c1',0.0001,'c2',0.9,'maxiter',10,'amax',1.1);
lbfgs_options = struct('maxiter', 60, ...
                       'termination', 1.0000e-004, ...
                       'xtermination', 1.0000e-004, ...
                       'm', 10, ...
                       'wolfe', wolfe, ...
                       'echo', false);

ooclabels = oneofc(Y');
% ooclabels = sign(ooclabels-0.5);
cnum = size(ooclabels, 2);
if (cnum == 2),
	ooclabels = Y';
	cnum = 1;
end;
				   
weights = initweights;
bias = initbias;
Phi = initPhi;

% Xt = xrepmat((origData(:))', [M, 1]);
% Xtt = reshape(Xt, [M * N numSamples]);

for iter = 1:20,
	
	disp(sprintf('Entered iter %d.\n', iter));
	weightsstd = weights ./ repmat(stdvec, [1 cnum]);
% 	Wt = xrepmat(weightsstd, [N 1]);
% 	wXtensor = bsxfun(@times, Wt, Xtt);
	
	disp(sprintf('Started optimizing for Phi.\n'));
	[retval, xstarbest, xstarfinal, history] = ...
		lbfgs2(initPhi(:), lbfgs_options, gradphifun, [], origData, ooclabels, weightsstd, bias, orthlambda, M, N);

	Phi = reshape(xstarbest, [M N]);
	initPhi = Phi;

	
	X = (Phi * origData)';
	[X, stdvec] = l2norm(X);
	stdvec = stdvec';

	disp(sprintf('Started optimizing for SVM.\n'));
	for classiter = 1 : cnum,
		[retval, xstarbest] = lbfgs2([weights(:, classiter); bias(classiter)], ...
			lbfgs_options, gradsvmfun, [], X', ooclabels(:, classiter), lambda, usebias);
		weights(:, classiter) = xstarbest(1:M);
		bias(classiter) = xstarbest(M+1);
	end;

	initweights = weights;
	initbias = bias;
	
	disp(sprintf('Saving results.\n'));
	save(sprintf('sanjeev/mat_files/interimorth%d', iter), 'Phi', 'weights', 'bias', 'stdvec');
end;
