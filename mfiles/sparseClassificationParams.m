% sparseClassificationParams - creates parameter structure for sparseClassification algorithms.
%
% params = sparseClassificationParams
%    Returns a structure with default values for parameter fields.
%
% params = sparseClassificationParams('field',value,...)
%    Returns a structure with the parameter structure 'field' set to value.
%	 All other fields are set to their default values. Arguments must come 
%	 in couples, otherwise an error is returned.
%

function params = sparseClassificationParams(varargin)

% set up parameters to default values

% general
params.allVerboseMode = -1;			% verbosity setting:
									% -1=show no messages at all,
									% 1=show only results,
									% 2=show all messages.

% data related params
params.dataWinSize = 12;			% patch window size.
params.dataWhitening = 0;			% how to preprocess the data patches: 
									% 0=nothing, 
									% 1=whitening, 
									% 2=l1-norm normalization,
									% 3=l2-norm normalization.
params.dataResizeRatio = 0.2;		% resize patches to scale.
params.dataBaseSize = 100;			% base dimension of patches before 
									% scaling.

% dictionary related params
params.dictionaryMethod = 'gradient';	% dictionary learning algorithm to 
									% be used:
									% ksvd, memory, gradient, coupledksvd,
									% coupledmemory, coupledgradient.
params.dictionarySize = 128;		% number of atoms.
params.dictionaryMode = 2;			% learning mode (1 or 2).
params.dictionaryKappa = 4;			% maximun number of nonzero atoms.
params.dictionaryNumIters = 100;	% number of iterations of main loop.
params.dictionaryNumIters2 = 10;	% number of iterations of inner loop.
params.dictionaryBlockRatio = 0;	% ratio of block size over total 
									% training samples (default 0, i.e. use
									% all samples at each iteration).
params.dictionaryInitDict = NaN;	% dictionary used for initialization
									% (NaN if none provided).
params.dictionaryLambda = 0.15;		% regularization parameter.
params.dictionaryLambda2 = 0.1;		% secondary regularization parameter.
params.dictionarySensMethod = 'orig'; % method for learning projection 
									% matrix to be used in coupled
									% algorithms:
									% orig, conj.
params.dictionaryAlpha = 1 / 32;	% relative weight of reconstruction 
									% costs.
params.dictionaryNoiseSTD = 0.15;	% noise std for projected data.
params.dictionaryInitProjMat = NaN;	% projection matrix used for 
									% initialization (NaN if none 
									% provided).
params.dictionaryClearing = 1;		% perform dictionary clearing.
params.dictionaryNoiseModel = 0;	% noise model to be used:
									% 0=Gaussian (lasso),
									% 1=exponential family,
									% 2=kernel.

% coding related params
params.codingMethod = 'lasso';		% sparse coding algorithm to be used:
									% omp, lasso, feature.
params.codingMode = 2;				% learning mode (1 or 2).
params.codingKappa = 4;				% maximun number of nonzero atoms.
params.codingLambda = 0.15;			% regularization parameter.
params.codingLambda2 = 0.1;			% secondary regularization parameter.
									% path to save codes.
params.codingEpsilon = 0.001;		% maximum reconstruction error.

% compression related params
params.compressNumSamples = 8;		% number of compressive measurements.

% svm related params
params.svmUseBias = 0;				% use bias term.
% params.svmLambda = [.0000001 .000001 .00001, .0001, .001, .01, .1, 1, 10, 100, 1000];
params.svmLambda = 0.1;				% regularization parameter.
params.svmLossFunction = 'huber';	% loss function to be used:
									% hinge=hinge loss (uses libsvm)
									% square=squared hinge loss,
									% huber=huberized hinge loss.
params.svmPreprocessMode = 3;		% how to preprocess the features used: 
									% 0=nothing, 
									% 1=whitening, 
									% 2=linear scaling,
									% 3=scaling l2 norm.
params.svmProbabilityEstimates = 0;	% train SVM for probability estimates 
									% (0 or 1).
params.svmCramerSingerApprox = 1;	% parameter for approximation of 
									% Cramer-Singer loss function.
params.svmNuclearApprox = 0.01;		% parameter for approximation of
									% nuclear norm.
params.svmNumIters = 300;			% number of line-searches.
params.svmRegularization = 'frobenius'; % regularization to use (only 
									% multi-class and multi-task):
									% frobenius, nuclear

% exponential family and kernel params
% TODO: Revert changes I had done that removed kernel/expfamily selections.
params.expFamily = 'P';				% exponential family to be used:
									% P or p=Poisson,
									% B or b=Bernoulli.
params.kernelType = 'G';			% kernel function to be used:
									% L or l: linear,
									% G or g: Gaussian,
									% P or p: polynomial,
									% S or s: sigmoid.
params.kernelParam1 = 1;			% first parameter for kernel methods.
params.kernelParam2 = 1;			% second parameter for kernel methods.

% text datasets params
params.textDataSet = 'tng';			% data to be used:
									% tng=20_newsgroups,
									% tng+reu=20_newsgroups and
									% Reuters-21578 union,
									% reu+tng=20_newsgroups and
									% Reuters-21578 union,
params.textVocabulary = 'tng';		% vocabulary to be used:
									% tng=20_newsgroups,
									% reu+tng=20_newsgroups and
									% Reuters-21578 intersection (in that
									% order).
params.textSelectionMethod = 'counts';	% vocabulary selection method to be
									% used:
									% mutinfo=largest mutual information
									% with class labels,
									% counts=largest counts over all
									% documents,
									% occurences=largest number of unique
									% occurrences over all documents,
									% random=random selection.
params.textNumWords = 100;			% number of words in vocabulary.

% check for incorrect inputs
if (mod(length(varargin),2) ~= 0)
	error('Invalid input arguments, type help sparseClassificationParams.');
end;

fieldsList = fieldnames(params);
fieldsSet = zeros(length(fieldsList));

% check for invalid inputs
for argCount = 1:2:length(varargin)
	if(~isfield(params, varargin{argCount}))
		error('Invalid field "%s".', varargin{argCount});
	else
		fieldIndex = find(strcmp(fieldsList, varargin{argCount}));
		if (fieldsSet(fieldIndex) == 1),
			warning('Field "%s" already set by previous argument.', fieldsList{fieldIndex});
		else
			fieldsSet(fieldIndex) = 1;
		end;
		eval(sprintf('params.%s = varargin{argCount + 1};',varargin{argCount}));
	end;
end;
