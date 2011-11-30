function [Phi, weights, bias, stdvec] = backpropagation_mask_bounds(initPhi, initweights, initbias, origData, Y, stdvec, usenorm, mask)

%
% Input:
%        origData --- d x N matrix, n data points in a d-dim space
%        Y --- 1 x N matrix, labels of n data points -1 or 1

usebias = 1;
lossfun = 'huber';
[M N] = size(initPhi); 
numSamples = size(origData, 2);
lambda = 0.1;
gradsvmfun = 'huberhinge_obj_grad_mex';
gradphifun = 'huber_obj_grad_mex';
% nonlcon = 'face_constraint';

if (nargin < 7),
	usenorm = 1;
end;

if (nargin < 8),
	mask = [];
end;

% w0 = zeros(D,1);
% b0 = 0;

wolfe = struct('a1',0.5,'a0',0.01,'c1',0.0001,'c2',0.9,'maxiter',10,'amax',1.1);
lbfgs_options = struct('maxiter', 30, ...
                       'termination', 1.0000e-004, ...
                       'xtermination', 1.0000e-004, ...
                       'm', 10, ...
                       'wolfe', wolfe, ...
                       'echo', false);
options = optimset('GradObj','on', 'GradConstr','on');
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
if (usenorm == 0),
	stdvec = ones(size(stdvec));
end;

if (~isempty(mask)),
	regions = unique(mask);
	maskedPhi = zeros(size(initPhi));
	for iter = 1:length(regions),
		maskedPhi(iter, mask ~= regions(iter)) = 1;
	end;
	Cgrad = zeros(numel(initPhi), 1);
	nonz = find(maskedPhi);
	Ceqgrad = zeros(numel(initPhi), length(nonz));
	subs = sub2ind([numel(maskedPhi) length(nonz)], nonz', 1:length(nonz));
	Ceqgrad(subs) = 1;
end;

for iter = 1:5,
	
	disp(sprintf('Entered iter %d.\n', iter));
	weightsstd = weights ./ repmat(stdvec, [1 cnum]);
% 	Wt = xrepmat(weightsstd, [N 1]);
% 	wXtensor = bsxfun(@times, Wt, Xtt);
	
	disp(sprintf('Started optimizing for Phi.\n'));
% 	xstarbest = minimize(initPhi(:), gradphifun, 300, 0, origData, ooclabels, weightsstd, bias);
% 	xstarbest = minimize_huber(initPhi(:), 600, origData, ooclabels, weightsstd, bias);
	
	xstarbest = fmincon(@(x) huber_obj_grad_mex(x, origData, ooclabels, weightsstd, bias), initPhi(:),...
				[], [], [], [], zeros(size(initPhi(:))), ones(size(initPhi(:))),...
				@(x) face_constraint(x, M, N, mask, Cgrad, Ceqgrad, maskedPhi), options);

	Phi = reshape(xstarbest, [M N]);
	initPhi = Phi;

	
	X = (Phi * origData)';
	if (usenorm == 1),
		[X, stdvec] = l2norm(X);
		stdvec = stdvec';
	else
		stdvec = ones(size(stdvec));
	end;

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
	save(sprintf('sanjeev/mat_files/mask_boxes_bounds_conj_iter%d', iter), 'Phi', 'weights', 'bias', 'stdvec');
end;
