function plotClusters(trainFeatures, trainLabels, Vec, idVector)

% ColorOrder = get(gca,'ColorOrder');
% numColors = size(ColorOrder, 1);
colors = pmkmp(length(idVector));
if (~isempty(Vec)),
	trainFeaturesReduced = reduceDimension(Vec, trainFeatures, trainFeatures);
else
	trainFeaturesReduced = trainFeatures;
end;

for iter = 1:length(idVector),
% 	plot(trainFeaturesReduced(trainLabels==idVector(iter), 1), trainFeaturesReduced(trainLabels == idVector(iter), 2),...
% 		'.','Color',[ColorOrder(mod(iter, numColors) + 1, :)], 'MarkerSize', 20);
	plot(trainFeaturesReduced(trainLabels==idVector(iter), 1), trainFeaturesReduced(trainLabels == idVector(iter), 2),...
		'.', 'Color', colors(iter, :), 'MarkerSize', 20);
	hold on;
end
axis off
