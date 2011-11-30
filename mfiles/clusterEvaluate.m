function [totalAccuracy, clusterIDs, clusterAccuracies] = clusterEvaluate(labels, assignments)

labelNames = unique(labels);
numClusters = length(unique(assignments));
classifiedSamples = 0;
for iterCluster = 1:numClusters,
	clusterMembers = assignments == iterCluster;
	clusterLabels = labels(clusterMembers);
	clusterHist = hist(clusterLabels, labelNames);
	[correctLabels, maxLabelInd] = max(clusterHist);
	clusterIDs(iterCluster) = labelNames(maxLabelInd);
	clusterAccuracies(iterCluster) = correctLabels / length(clusterLabels) * 100;
	classifiedSamples = classifiedSamples + correctLabels;
end;
totalAccuracy = classifiedSamples / length(labels) * 100;
