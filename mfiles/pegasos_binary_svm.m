function [weights, bias] = pegasos_binary_svm(X, Y, lambda, biasFlag,...
									numIters, batchSize, returnAverageFlag)
								
[N numSamples] = size(X);
if (length(Y) ~= numSamples),
	error('Length of label vector is different from the number of samples.');
end;

weights = zeros(N, 1);
bias = 0;
invSqrtLambda = 1 / sqrt(lambda);

if (returnAverageFlag == 1),
	weightsAverage = zeros(N, 1);
	biasAverage = 0;
end;

for iter = 1:numIters,
 	batch = randsample(numSamples, batchSize);
	Ypred = weights' * X(:, batch) + bias;
	YYpred = Y(batch) .* Ypred < 1;
	eta = 1 / lambda / (iter + 1);
	weights = (1 - eta * lambda) * weights +...
		eta / batchSize * sum(bsxfun(@times, Y(batch(YYpred)), X(:, batch(YYpred))), 2);
	multiplier = invSqrtLambda / norm(weights); 
	if (multiplier < 1),
		weights = weights * multiplier; 
	end;
	if (biasFlag == 1) 
		bias = bias + eta / batchSize * sum(Y(batch(YYpred)));
	end;
	if (returnAverageFlag == 1),
		weightsAverage = weightsAverage + 1 / numIters * weights;
		biasAverage = biasAverage + 1 / numIters * bias;
	end;
end;
		
if (returnAverageFlag == 1),
	weights = weightsAverage;
	bias = biasAverage;
end;
