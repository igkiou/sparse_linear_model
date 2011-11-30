function plotClusters3(trainFeatures, trainLabels, Vec, idVector)

ColorOrder = get(gca,'ColorOrder');
numColors = size(ColorOrder, 1);
if (~isempty(Vec)),
	trainFeaturesReduced = reduceDimension(Vec, trainFeatures, trainFeatures);
else
	trainFeaturesReduced = trainFeatures;
end;

for iter = 1:length(idVector),
	plot3(trainFeaturesReduced(trainLabels==idVector(iter), 1), trainFeaturesReduced(trainLabels == idVector(iter), 2),...
		trainFeaturesReduced(trainLabels==idVector(iter), 3),'.','Color',[ColorOrder(mod(iter, numColors) + 1, :)]);
	hold on;
end
axis off
