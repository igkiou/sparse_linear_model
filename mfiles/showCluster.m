function showCluster(samples, assignment)

numClusters = length(unique(assignment));
sampleDim = sqrt(size(samples, 1));
numSamples = size(samples, 2);
avgClusterDim = ceil(sqrt(numSamples / numClusters));


for iterCluster = 1:numClusters,
	figure;showdict(samples(:,assignment == iterCluster), [sampleDim sampleDim], avgClusterDim, avgClusterDim, 'lines');
end;
