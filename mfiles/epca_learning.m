function dictionary = epca_learning(data_matrix, results_file, save_option, params)
% TODO: add initialization of dictionary using k-means.

%% parse input

if (nargin <= 3) || (~isstruct(params)),
	params = sparseClassificationParams;
end;

verboseMode = params.allVerboseMode;

dictSize = params.dictionarySize;
numIters = params.dictionaryNumIters;
initDict = params.dictionaryInitDict;
blockRatio = params.dictionaryBlockRatio;

expFamily = params.expFamily;

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

expgradientparams.initdict = initDict;
expgradientparams.dictsize = dictSize;
expgradientparams.iternum = numIters;
expgradientparams.blockratio = blockRatio;
expgradientparams.family = expFamily;
expgradientparams.savepath = results_file;
if (verboseMode == 1),
	expgradientparams.printinfo = 1;
	expgradientparams.errorinfo = 0;
end;
dictionary = exppca(data_matrix, expgradientparams);

if(verboseMode == 1),
	fprintf('Finished running dictionary learning.\n');
end;

%% save results

if (save_option == 1),
	if(verboseMode == 1),
		fprintf('Started saving results.\n');
	end;

	eval(sprintf('save %s dictionary params dataSamples', results_file));

	if(verboseMode == 1),
		fprintf('Finished saving results.\n');
	end;
end;
