load ~/MATLAB/datasets_all/hyperspectral/HPdictionary.mat D sv
load ~/MATLAB/datasets_all/hyperspectral/data_split.mat trainData
X = trainData; clear trainData;
l1 = mean(sv(1, :));

dictparams.mu = 0.000001 * l1;
% dictparams.kappa = 0;
dictparams.tolerance = 10 ^ - 6;
dictparams.delta = 10 ^ - 2;
dictparams.numIters = 200;
dictparams.eta = 0.8;
dictparams.K = size(D, 2);

dataparams.numRows = 64;
dataparams.numSamples = size(X, 2);

codingparams.lambda = 0.01;
codingparams.mode = 2;
codingparams.lambda2 = 0;

params.numGlobalIters = 2;

XAt = zeros(dataparams.numSamples, dictparams.K);
AAt = zeros(dataparams.numSamples, dictparams.K);

fprintf(...
	'Parameters: mu %g kappa %g delta %g numIters %d eta %g numSamples %d lambda %g numAtoms %d numGlobalIters %d.\n',...
	dictparams.mu, dictparams.kappa, dictparams.delta, dictparams.numIters, dictparams.eta, dataparams.numSamples,...
	codingparams.lambda, dictparams.K, params.numGlobalIters);
for iter = 1:params.numGlobalIters,
	fprintf('Now running iter %d of %d.\n', iter, params.numGlobalIters);
	
	fprintf('Starting sparse coding.\n');
	A = mexLasso(X, D, codingparams);
	fprintf('Finished sparse coding.\n');
	
	if(nnz(A) == 0),
		fprintf('All sparse codes are zero.');
		break;
	end;
	
	fprintf('Starting intermediate computations.\n');
	XAt = mexCalcXAt(X, A);
	AAt = mexCalcAAt(A);
	fprintf('Finished ntermediate computations.\n'); 
	
	fprintf('Starting dictionary iteration.\n');
	D = matrix_dictionary_learning_lowrank_apg_mex(D, XAt, AAt,...
			dataparams.numRows, dataparams.numSamples, dictparams.mu,...
			dictparams.kappa, dictparams.tolerance, dictparams.delta,...
			dictparams.numIters, dictparams.eta);
	fprintf('Finished dictionary iteration.\n');
		
	if(nnz(D) == 0),
		fprintf('All dictionary elements are zero.');
		break;
	end;
end;
