function [result accuracy] = knn_classify(testFeatures, trainFeatures, numNeighbors, testLabels, trainLabels)

if (numNeighbors > 1)
	distancePairs = l2_distance_mex(testFeatures', trainFeatures');
	% [foo neighbors] = sort(distancePairs, 2, 'ascend');
	[foo neighbors] = mink(distancePairs', numNeighbors);
	clear foo distancePairs
	neighbors = neighbors';
	neighborLabels = trainLabels(neighbors);
	clear neighbors

	baseLabels = unique(trainLabels);

	neighborHist = histc(neighborLabels(:, 1:numNeighbors), baseLabels, 2);
	clear neighborLabels
	[bar maxind] = max(neighborHist, [], 2);
	result = baseLabels(maxind);
	clear bar
	
% 	result = knnclassify(testFeatures, trainFeatures, trainLabels, numNeighbors, 'euclidean', 'nearest');
% 	
elseif (numNeighbors == 1)
	blockSize = 1000;
	testSamples = length(testLabels);
	if (testSamples < blockSize),
		blockSize = testSamples;
	end;
	result = zeros(size(testLabels));
	for iter = 1:(testSamples / blockSize)
		distancePairs = l2_distance_mex(testFeatures(((iter - 1) * blockSize + 1):(iter * blockSize), :)', trainFeatures');
		[foo nearestNeighbor] = min(distancePairs, [], 2);
		result(((iter - 1) * blockSize + 1):(iter * blockSize)) = trainLabels(nearestNeighbor);
	end;
	if (iter * blockSize ~= testSamples),
		distancePairs = l2_distance_mex(testFeatures((iter * blockSize + 1):end, :)', trainFeatures');
		[foo nearestNeighbor] = min(distancePairs, [], 2);
		result((iter * blockSize + 1):end) = trainLabels(nearestNeighbor);
	end;
end;

if (nargout > 1),
	accuracy = sum(testLabels == result) / length(testLabels) * 100;
end;
