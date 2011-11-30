function [bestaccuracy bestmodel bestlabels stdVec trainOutputs testOutputs] = ...
	multiclass_svm(trainData, trainLabels, testData, testLabels, params)

if (nargin < 5) || (~isstruct(params)),
	params = sparseClassificationParams;
end;

verboseMode = params.allVerboseMode;

lambda = params.svmLambda;
preprocessMode = params.svmPreprocessMode;
gamma = params.svmCramerSingerApprox;
rho = params.svmNuclearApprox;
numIters = params.svmNumIters;
regularization = params.svmRegularization;

%% preprocess data

trainSamples = size(trainData, 2);
testSamples = size(testData, 2);
dataSamples = trainSamples + testSamples;
signalDim = size(trainData, 1);
numTasks = length(unique(trainLabels));

trainData = trainData';
trainLabels = trainLabels';
testData = testData';
testLabels = testLabels';

if(verboseMode == 1),
	fprintf('Started whitening the data.\n');
end;

if (preprocessMode == 1),
	[trainData, meanVec, stdVec] = whiten(trainData);
	testData = testData - repmat(meanVec, [testSamples, 1]);
	testData = testData ./ repmat(stdVec, [testSamples, 1]);
elseif (preprocessMode == 3),
	[trainData, stdVec] = l2norm(trainData);
	testData = testData ./ repmat(stdVec, [testSamples, 1]);
else
	stdVec = ones(1, size(testData, 2));
end;

if(verboseMode == 1),
	fprintf('Finished whitening the data.\n');
end;

trainData = trainData';
trainLabels = trainLabels';
testData = testData';
testLabels = testLabels';

%% Train SVM

numLambdas = length(lambda);
accuracy = zeros(1, numLambdas);
accuracyAlt = zeros(1, numLambdas);
predictedLabels = zeros(testSamples, numLambdas);

for iter = 1:numLambdas,
	if(verboseMode == 1),
		fprintf('Now running iter %d of %d, lambda: %g. ', iter, numLambdas, lambda(iter));
	end;
	
	% TODO: Find better initialization, perhaps from one-vs-all?
	initW = zeros(signalDim, numTasks);
	if (strcmp(regularization, 'nuclear')),
		[weightVector foo i] = minimize(initW(:), @cramersinger_nuclear_obj_grad_mex, numIters, 0, trainData, trainLabels, gamma, rho, lambda(iter));
	% 	options = optimset('GradObj','on');
	% 	[weightVector, fval, exitflag, output] = fminunc(@(x) cramersinger_nuclear_obj_grad(x, trainData, trainLabels, gamma, rho, lambda(iter)), initW, options);
	elseif (strcmp(regularization, 'frobenius')),
		[weightVector foo i] = minimize(initW(:), @cramersinger_frobenius_obj_grad_mex, numIters, 0, trainData, trainLabels, gamma, lambda(iter));
	% 	options = optimset('GradObj','on');
	% 	[weightVector, fval, exitflag, output] = fminunc(@(x)
	% 	cramersinger_nuclear_obj_grad(x, trainData, trainLabels, gamma, rho, lambda(iter)), initW, options);
	end;
	weightVector = reshape(weightVector, [signalDim numTasks]);
	svmModel(iter).w = weightVector;
	
	[foo predictedLabels(:, iter)] = max(weightVector'*testData, [], 1);
% 		[predictedLabels(:, iter) tempacc] = li2nsvm_fwd(testData, testLabels, weightVector, biasTerm);
	accuracy(iter) = sum(testLabels == predictedLabels(:, iter)') / testSamples * 100;
% 		className = [-1 1];
% 	accuracyTemp = zeros(1, length(className));
% 	for classIter = 1 : length(className),
% 		c = className(classIter);
% 		idx = find(testLabels == c);
% 		currPredictedLabels = predictedLabels(:, iter);
% 		currPredictedLabels = currPredictedLabels(idx);
% 		currTestLabels = testLabels(idx);
% 		accuracyTemp(classIter) = length(find(currPredictedLabels == currTestLabels)) / length(idx);
% 	end; 
	accuracyAlt(iter) = accuracy(iter);
	if(verboseMode == 1),
		fprintf('Accuracy: %g, accuracyAlt: %g.\n', accuracy(iter), accuracyAlt(iter));
	end;
end;

if(verboseMode == 1),
	fprintf('Finished running SVM. ');
end;
%%

[bestaccuracy index] = max(accuracy, [], 2);
% bestaccuracyalt = max(accuracyAlt, [], 2);
bestmodel = svmModel(index);
bestlabels = predictedLabels(:, index);

if (nargout >= 5),
	trainOutputs = svmModel(index).w' * trainData; %#ok
	if (nargout >= 6),
		testOutputs = svmModel(index).w' * testData; %#ok
	end;
end;
			
if(verboseMode >= 0),
	fprintf('Max accuracy: %g%%, lambda: %g\n', bestaccuracy, lambda(index));
% 	fprintf('Max accuracy alt: %g%%\n', bestaccuracyalt);
end;
