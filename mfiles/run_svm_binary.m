function [bestaccuracy bestmodel bestlabels stdVec trainOutputs testOutputs] = run_svm_binary(trainData, trainLabels, testData, testLabels, params)
%% parse inputs

if (nargin <= 4) || (~isstruct(params)),
	params = sparseClassificationParams;
end;

verboseMode = params.allVerboseMode;

lambda = params.svmLambda;
useBias = params.svmUseBias;
preprocessMode = params.svmPreprocessMode;
lossFunction = params.svmLossFunction;
probabilityEstimates = params.svmProbabilityEstimates;

%% convert labels to -1/1

uniqueLabels = unique(trainLabels);
if (length(uniqueLabels) > 2),
	error('There are more than two classes. Use run_svm.m instead.');
end;

if ~(((uniqueLabels(1) == 1) && (uniqueLabels(2) == - 1)) || ...
		((uniqueLabels(1) == - 1) && (uniqueLabels(2) == 1))),
	trainInds1 = trainLabels == uniqueLabels(1);
	trainInds2 = trainLabels == uniqueLabels(2);
	trainLabels(trainInds1) = -1;
	trainLabels(trainInds2) = 1;

	testInds1 = testLabels == uniqueLabels(1);
	testInds2 = testLabels == uniqueLabels(2);
	testLabels(testInds1) = -1;
	testLabels(testInds2) = 1;
end;

%% preprocess data

signalDim = size(trainData, 1);
trainSamples = size(trainData, 2);
testSamples = size(testData, 2);
dataSamples = trainSamples + testSamples;

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

%% train svm

if(verboseMode == 1),
	fprintf('Started running SVM.\n');
end;

numLambdas = length(lambda);
accuracy = zeros(1, numLambdas);
% accuracyAlt = zeros(1, numLambdas);
predictedLabels = zeros(testSamples, numLambdas);
for iter = 1:numLambdas,
	
	if(verboseMode == 1),
		fprintf('Now running iter %d of %d, lambda: %g. ', iter, numLambdas, lambda(iter));
	end;
	
	if (probabilityEstimates == 1),
		if(strcmp(lossFunction, 'huber') || strcmp(lossFunction, 'square')),
			warning('Loss function %s specified, but using libsvm in order to get probability estimates.', lossFunction); %#ok
		end;
		svmparams = ['-s', blanks(1), num2str(0), blanks(1), '-t', blanks(1), num2str(0), blanks(1),...
					'-c', blanks(1), num2str(1 / lambda(iter)), blanks(1), '-b', blanks(1), num2str(1)];
		svmModel(iter) = libsvm_train(trainLabels, sparse(trainData), svmparams);
		[predictedLabels(:, iter), accuracy(iter)] = libsvm_predict(testLabels, sparse(testData), svmModel(iter), '-b 1');
	elseif(strcmp(lossFunction, 'hinge')),
		if (trainSamples / signalDim < 1.2),
			svmparams = ['-s', blanks(1), num2str(1), blanks(1), '-c', blanks(1),...
						num2str(1 / lambda(iter)), blanks(1), '-B', blanks(1), num2str(useBias)];
		else
			svmparams = ['-s', blanks(1), num2str(2), blanks(1), '-c', blanks(1),...
						num2str(1 / lambda(iter)), blanks(1), '-B', blanks(1), num2str(useBias)];
		end;
		svmModel(iter) = liblinear_train(trainLabels, sparse(trainData), svmparams);
		[predictedLabels(:, iter), accuracy(iter)] = liblinear_predict(testLabels, sparse(testData), svmModel(iter));
	else
		[weightVector biasTerm] = li2nsvm_lbfgs(trainData', trainLabels', lambda(iter),...
 			[], [], lossFunction, useBias);
		svmModel(iter).w = weightVector(:, 1);
		svmModel(iter).b = biasTerm(1);
		[predictedLabels(:, iter) accuracy(iter)] = li2nsvm_fwd(testData', testLabels', weightVector, biasTerm);		
	end;
	if(verboseMode == 1),
		fprintf('Accuracy: %g.\n', accuracy(iter));
	end;
end;

if(verboseMode == 1),
	fprintf('Finished running SVM. ');
end;

[bestaccuracy index] = max(accuracy, [], 2);
% bestaccuracyalt = max(accuracyAlt, [], 2);
bestmodel = svmModel(index);
bestlabels = predictedLabels(:, index);
if (preprocessMode == 1),
	bestmodel.meanVec = meanVec;
	bestmodel.stdVec = stdVec;
elseif (preprocessMode == 3),
	bestmodel.stdVec = stdVec;
end;

if (nargout >= 5),
	if (probabilityEstimates == 1),
		if(strcmp(lossFunction, 'huber') || strcmp(lossFunction, 'square')),
			warning('Loss function %s specified, but using libsvm in order to get probability estimates.', lossFunction); %#ok
		end;
		if(length(unique(trainLabels)) > 2),
			warning('Using libsvm (one-vs-one multiclass SVM) in order to get probability estimates.'); %#ok
		end;
		[dump1 dump2 trainOutputs] = libsvm_predict(trainLabels, sparse(trainData), svmModel(index), '-b 1'); %#ok
		trainOutputs = trainOutputs';
		if (nargout >= 6),
			[dump1 dump2 testOutputs] = libsvm_predict(testLabels, sparse(testData), svmModel(index), '-b 1'); %#ok
			testOutputs = testOutputs';
		end;
	elseif (strcmp(lossFunction, 'hinge')),
		[dump1 dump2 trainOutputs] = liblinear_predict(trainLabels, sparse(trainData), svmModel(index)); %#ok
		trainOutputs = trainOutputs';
		if (nargout >= 6),
			[dump1 dump2 testOutputs] = liblinear_predict(testLabels, sparse(testData), svmModel(index)); %#ok
			testOutputs = testOutputs';
		end;
	else
		[dump1 dump2 trainOutputs] = li2nsvm_fwd(trainData', trainLabels', svmModel(index).w, svmModel(index).b);	%#ok
		if (nargout >= 6),
			[dump1 dump2 testOutputs] = li2nsvm_fwd(testData', testLabels', svmModel(index).w, svmModel(index).b); %#ok	
		end;
	end;
end;

if(verboseMode >= 0),
	fprintf('MaxIndex: %d, lambda: %g, max accuracy: %g%%\n', index, lambda(index), bestaccuracy);
% 	fprintf('Max accuracy alt: %g%%\n', bestaccuracyalt);
end;
