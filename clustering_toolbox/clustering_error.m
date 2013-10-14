function [clusterError labelAssignment] = clustering_error(labels, assignments)
%CLUSTERING_ERROR Use hungarian algorithm to find best label assignment 
% and classification error for a clustering assignment

labelNames = unique(labels);
numLabels = length(labelNames);
clusterNames = unique(assignments);
numClusters = length(clusterNames);

costMat = zeros(numLabels, numClusters);

for iterLabel = 1:numLabels,
	for iterCluster = 1:numClusters,
		costMat(iterLabel, iterCluster) = sum(labels(assignments == clusterNames(iterCluster)) ~= labelNames(iterLabel));
	end;
end;

[assignment misClassified] = hungarian(costMat);
labelAssignment = labelNames(assignment);
clusterError = misClassified / length(labels) * 100;
