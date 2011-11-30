load ~/MATLAB/datasets_all/hyperspectral/HPdictionary.mat D sv
load ~/MATLAB/datasets_all/hyperspectral/data_split.mat trainData
X = trainData; clear trainData;
l1 = mean(sv(1, :));

dictparams.mu = 0.1 * l1;
% dictparams.kappa = 0;
dictparams.tolerance = 10 ^ - 6;
dictparams.delta = 10 ^ - 2;
dictparams.numIters = 200;
dictparams.eta = 0.8;
dictparams.K = size(D, 2);

kernelparams.tau = 1000;

dataparams.numRows = 64;
dataparams.numColumns = 31;
dataparams.numSamples = size(X, 2);

codingparams.lambda = 0.01;
codingparams.mode = 2;
codingparams.lambda2 = 0;

params.numGlobalIters = 100;

wavelengths = 420:10:720;
kernelMat = kernel_gram(wavelengths, [], 'h', kernelparams.tau);
[U L] = eig(kernelMat);
Y = U * sqrt(L);

YtY = Y'*Y;
XY = zeros(dataparams.numRows * dataparams.numColumns, dataparams.numSamples);
for iter = 1:dataparams.numSamples,
	XY(:,iter) = vec(reshape(X(:,iter), [dataparams.numRows dataparams.numColumns]) * Y);
end;

XYAt = zeros(dataparams.numSamples, dictparams.K);
AAt = zeros(dataparams.numSamples, dictparams.K);
DYt = zeros(dataparams.numRows * dataparams.numColumns, dictparams.K);

fprintf(...
	'Parameters: tau %g mu %g kappa %g delta %g numIters %d eta %g numSamples %d lambda %g numAtoms %d numGlobalIters %d.\n',...
	kernelparams.tau, dictparams.mu, dictparams.kappa, dictparams.delta,...
	dictparams.numIters, dictparams.eta, dataparams.numSamples,...
	codingparams.lambda, dictparams.K, params.numGlobalIters);
for iter = 1:params.numGlobalIters,
	fprintf('Now running iter %d of %d.\n', iter, params.numGlobalIters);
	
	fprintf('Starting preparing dictionary.\n');
	for iterAtom = 1:dictparams.K,
		DYt(:,iterAtom) = vec(reshape(D(:,iterAtom), [dataparams.numRows dataparams.numColumns]) * Y');
	end;
	fprintf('Finished preparing dictionary.\n');
	
	fprintf('Starting sparse coding.\n');
	A = mexLasso(X, DYt, codingparams);
	fprintf('Finished sparse coding.\n');
	
	fprintf('Starting intermediate computations.\n');
	XYAt = mexCalcXAt(XY, A);
	AAt = mexCalcAAt(A);
	fprintf('Finished ntermediate computations.\n'); 
	
	fprintf('Starting dictionary iteration.\n');
	D = operator_dictionary_learning_lowrank_weighted_apg_mex(D, XYAt, AAt,...
			YtY, dataparams.numRows, dataparams.numSamples, dictparams.mu,...
			dictparams.kappa, dictparams.tolerance, dictparams.delta,...
			dictparams.numIters, dictparams.eta);
	fprintf('Finished dictionary iteration.\n');
end;
