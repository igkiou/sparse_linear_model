function [bestaccuracy bestmodel bestlabels stdVec trainOutputs testOutputs] = run_svm(trainData, trainLabels, testData, testLabels, params)
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

%% preprocess data

trainSamples = size(trainData, 2);
testSamples = size(testData, 2);
dataSamples = trainSamples + testSamples;
signalDim = size(trainData, 1);

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
accuracyAlt = zeros(1, numLambdas);
predictedLabels = zeros(testSamples, numLambdas);
for iter = 1:numLambdas,
	
	if(verboseMode == 1),
		fprintf('Now running iter %d of %d, lambda: %g. ', iter, numLambdas, lambda(iter));
	end;
	
	if (probabilityEstimates == 1),
		if(strcmp(lossFunction, 'huber') || strcmp(lossFunction, 'square')),
			warning('Loss function %s specified, but using libsvm in order to get probability estimates.', lossFunction); %#ok
		end;
		if(length(unique(trainLabels)) > 2),
			warning('Using libsvm (one-vs-one multiclass SVM) in order to get probability estimates.'); %#ok
		end;
		svmparams = ['-s', blanks(1), num2str(0), blanks(1), '-t', blanks(1), num2str(0), blanks(1),...
					'-c', blanks(1), num2str(1 / lambda(iter)), blanks(1), '-b', blanks(1), num2str(1)];
		svmModel(iter) = libsvm_train(trainLabels, sparse(trainData), svmparams);
		[predictedLabels(:, iter), accuracy(iter)] = libsvm_predict(testLabels, sparse(testData), svmModel(iter), '-b 1');
		className = svmModel(iter).Label;
		accuracyTemp = zeros(1, length(className));
		for classIter = 1 : length(className),
			c = className(classIter);
			idx = find(testLabels == c);
			currPredictedLabels = predictedLabels(:, iter);
			currPredictedLabels = currPredictedLabels(idx);
			currTestLabels = testLabels(idx);
			accuracyTemp(classIter) = length(find(currPredictedLabels == currTestLabels)) / length(idx);
		end; 
		accuracyAlt(iter) = mean(accuracyTemp) * 100;
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
		className = svmModel(iter).Label;
		accuracyTemp = zeros(1, length(className));
		for classIter = 1 : length(className),
			c = className(classIter);
			idx = find(testLabels == c);
			currPredictedLabels = predictedLabels(:, iter);
			currPredictedLabels = currPredictedLabels(idx);
			currTestLabels = testLabels(idx);
			accuracyTemp(classIter) = length(find(currPredictedLabels == currTestLabels)) / length(idx);
		end; 
		accuracyAlt(iter) = mean(accuracyTemp) * 100;
	
	else
		[weightVector biasTerm className] = li2nsvm_multiclass_lbfgs(trainData', trainLabels', lambda(iter),...
			[], lossFunction, useBias);
		svmModel(iter).w = weightVector;
		svmModel(iter).b = biasTerm;
% 		[weightVector biasTerm] = li2nsvm_lbfgs(trainData, trainLabels, lambda(iter),...
%  			[], [], lossFunction, useBias);
		predictedLabels(:, iter) = li2nsvm_multiclass_fwd(testData', weightVector, biasTerm, className);
% 		[predictedLabels(:, iter) tempacc] = li2nsvm_fwd(testData, testLabels, weightVector, biasTerm);
		accuracy(iter) = sum(testLabels == predictedLabels(:, iter)) / testSamples * 100;
% 		className = [-1 1];
		accuracyTemp = zeros(1, length(className));
		for classIter = 1 : length(className),
			c = className(classIter);
			idx = find(testLabels == c);
			currPredictedLabels = predictedLabels(:, iter);
			currPredictedLabels = currPredictedLabels(idx);
			currTestLabels = testLabels(idx);
			accuracyTemp(classIter) = length(find(currPredictedLabels == currTestLabels)) / length(idx);
		end; 
		accuracyAlt(iter) = mean(accuracyTemp) * 100;
	end;
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
		[dump1 trainOutputs] = li2nsvm_multiclass_fwd(trainData', svmModel(index).w, svmModel(index).b, className); %#ok
		if (nargout >= 6),
			[dump1 testOutputs] = li2nsvm_multiclass_fwd(testData', svmModel(index).w, svmModel(index).b, className); %#ok
		end;
	end;
end;
			
if(verboseMode >= 0),
	fprintf('Max accuracy: %g%%, lambda: %g\n', bestaccuracy, lambda(index));
% 	fprintf('Max accuracy alt: %g%%\n', bestaccuracyalt);
end;
