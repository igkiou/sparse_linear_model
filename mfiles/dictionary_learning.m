function [dictionary projectionMatrix] = dictionary_learning(data_matrix, params, results_file)
% TODO: add initialization of dictionary using k-means.

%% parse input

if (nargin < 2) || (~isstruct(params)),
	params = sparseClassificationParams;
end;

if (nargin < 3) || (~isstruct(params)),
	results_file = [];
end;

verboseMode = params.allVerboseMode;

dictionaryMethod = params.dictionaryMethod;
dictSize = params.dictionarySize;
kappa = params.dictionaryKappa;
numIters = params.dictionaryNumIters;
numIters2 = params.dictionaryNumIters2;
initDict = params.dictionaryInitDict;
alpha = params.dictionaryAlpha;
noisestd = params.dictionaryNoiseSTD;
initProjMat = params.dictionaryInitProjMat;
lambda = params.dictionaryLambda;
mode = params.dictionaryMode;
sensmethod = params.dictionarySensMethod;
blockRatio = params.dictionaryBlockRatio;
dictclear = params.dictionaryClearing;

noiseModel = params.dictionaryNoiseModel;
expFamily = params.expFamily;
kernelType = params.kernelType;	
kernelParam1 = params.kernelParam1;
kernelParam2 = params.kernelParam2;

senssize = params.compressNumSamples;

%% check init dict

if(verboseMode == 1),
	fprintf('Started checking initial dictionary.\n');
end;

dataSamples = size(data_matrix, 2);
signalSize = size(data_matrix, 1);
if ~any(isnan(initDict(:))) && ((size(initDict, 1) ~= signalSize) || ...
		(size(initDict, 2) ~= dictSize)),
	error('Incompatible initial dictionary dimensions');
end;

if(verboseMode == 1),
	fprintf('Finished checking initial dictionary.\n');
end;

%% run dictionary training

if(verboseMode == 1),
	fprintf('Started running dictionary learning.\n');
end;

if (noiseModel == 1),
	expgradientparams.initdict = initDict;
	expgradientparams.dictsize = dictSize;
	expgradientparams.iternum = numIters;
	expgradientparams.blockratio = blockRatio;
	expgradientparams.codinglambda = lambda;
	expgradientparams.dictclear = dictclear;
	expgradientparams.family = expFamily;
	expgradientparams.savepath = results_file;
	if (verboseMode == 1),
		expgradientparams.printinfo = 1;
		expgradientparams.errorinfo = 0;
	end;
	dictionary = expdictgradient(data_matrix, expgradientparams);
	projectionMatrix = NaN;
elseif (noiseModel == 2),
	kernelgradientparams.initdict = initDict;
	kernelgradientparams.dictsize = dictSize;
	kernelgradientparams.iternum = numIters;
	kernelgradientparams.iternum2 = numIters2;
	kernelgradientparams.blockratio = blockRatio;
	kernelgradientparams.codinglambda = lambda;
	kernelgradientparams.dictclear = dictclear;
	kernelgradientparams.kerneltype = kernelType;
	kernelgradientparams.kernelparam1 = kernelParam1;
	kernelgradientparams.kernelparam2 = kernelParam2;
	kernelgradientparams.savepath = results_file;
	if (verboseMode == 1),
		kernelgradientparams.printinfo = 1;
		kernelgradientparams.errorinfo = 0;
	end;
	dictionary = kerneldictgradient(data_matrix, kernelgradientparams);
	projectionMatrix = NaN;
elseif(strcmp(dictionaryMethod, 'ksvd')),
	ksvdparams.Tdata = kappa;
	ksvdparams.initdict = initDict;
	ksvdparams.dictsize = dictSize;
	ksvdparams.iternum = numIters;
	ksvdparams.memusage = 'high';
	if (verboseMode == 1),
		ksvdparams.printinfo = 1;
		ksvdparams.errorinfo = 1;
	end;
	dictionary = ksvd(data_matrix, ksvdparams);
	projectionMatrix = NaN;
	
elseif(strcmp(dictionaryMethod, 'coupledksvd')),	
	coupledksvdparams.Tdata = kappa;
	coupledksvdparams.initdict = initDict;
	coupledksvdparams.dictsize = dictSize;
	coupledksvdparams.iternum = numIters;
	coupledksvdparams.memusage = 'high';
	coupledksvdparams.senssize = senssize;
	coupledksvdparams.noisestd = noisestd;
	coupledksvdparams.sensmethod = sensmethod;
	coupledksvdparams.alpha = alpha;
	if (verboseMode == 1),
		coupledksvdparams.printinfo = 1;
		coupledksvdparams.errorinfo = 1;
	end;
	if ~any(isnan(initProjMat(:))) 
		if ((size(initProjMat, 2) ~= signalSize) || (size(initProjMat, 1) ~= senssize)),
			error('Incompatible initial dictionary dimensions');
		else
			coupledksvdparams.initsens = initProjMat;
		end;
	end;
	[dictionary, projectionMatrix] = coupledksvd(data_matrix, coupledksvdparams);
	
elseif(strcmp(dictionaryMethod, 'memory')),	
	memoryparams.mode = mode;
	if ~any(isnan(initDict(:))),
		memoryparams.D = initDict;
	end;
	memoryparams.K = dictSize;
	memoryparams.batchsize = ceil(blockRatio * size(data_matrix, 2));
	if (mode == 1),
		memoryparams.lambda = kappa;
	elseif (mode == 2)
		memoryparams.lambda = lambda;
	end;
	memoryparams.iter = numIters; 
	dictionary = mexTrainDL_Memory(data_matrix, memoryparams);
	projectionMatrix = NaN;
	
elseif(strcmp(dictionaryMethod, 'coupledmemory')),	
	coupledmemoryparams.dictsize = dictSize;
	coupledmemoryparams.iternum = numIters;
	coupledmemoryparams.iternum2 = numIters2;
	coupledmemoryparams.memusage = 'low';
	coupledmemoryparams.senssize = senssize;
	coupledmemoryparams.noisestd = noisestd;
	coupledmemoryparams.lambda = lambda;
	coupledmemoryparams.kappa = alpha;
	[dictionary, projectionMatrix] = coupledmemory(data_matrix, coupledmemoryparams);

elseif(strcmp(dictionaryMethod, 'gradient')),	
	gradientparams.initdict = initDict;
	gradientparams.dictsize = dictSize;
	gradientparams.iternum = numIters;
	gradientparams.blockratio = blockRatio;
	gradientparams.codinglambda = lambda;
	gradientparams.dictclear = dictclear;
	gradientparams.savepath = results_file;
	if (verboseMode == 1),
		gradientparams.printinfo = 1;
		gradientparams.errorinfo = 0;
	end;
	dictionary = dictgradient(data_matrix, gradientparams);
	projectionMatrix = NaN;
	
elseif(strcmp(dictionaryMethod, 'coupledgradient')),
	coupledgradientparams.initdict = initDict;	
	coupledgradientparams.dictsize = dictSize;
	coupledgradientparams.iternum = numIters;
	coupledgradientparams.blockratio = blockRatio;
	coupledgradientparams.senssize = senssize;
	coupledgradientparams.noisestd = noisestd;
	coupledgradientparams.codinglambda = lambda;
	coupledgradientparams.sensmethod = sensmethod;
	coupledgradientparams.alpha = alpha;
	coupledgradientparams.savepath = results_file;
	coupledgradientparams.dictclear = dictclear;
	if (verboseMode == 1),
		coupledgradientparams.printinfo = 1;
		coupledgradientparams.errorinfo = 0;
	end;
	[dictionary, projectionMatrix] = coupledgradient(data_matrix, coupledgradientparams);
	
else
	error('Unknown dictionary learning method: %s.', dictionaryMethod);
	
end;	

if(verboseMode == 1),
	fprintf('Finished running dictionary learning.\n');
end;
