function [weights, bias, class_labels] = pegasos_multiclass_svm(X, Y, lambda, biasFlag,...
									numIters, batchSize, returnAverageFlag)
								
[N numSamples] = size(X);
if (length(Y) ~= numSamples),
	error('Length of label vector is different from the number of samples.');
end;

[codedY class_labels] = oneofc(Y);
numTasks = length(class_labels);
weights = zeros(N, numTasks);
bias = zeros(numTasks, 1);

for iterT = 1:numTasks,
	[weights(:, iterT) bias(iterT)] = pegasos_binary_svm(X, codedY(:, iterT)',...
				lambda, biasFlag, numIters, batchSize, returnAverageFlag);
end;
