reducedDim = 10:10:100;
numDims = length(reducedDim);
accuracy_PCA = zeros(1, numDims);
accuracy_D_Graz = zeros(1, numDims);
accuracy_D_Graz_custom = zeros(1, numDims);

for iterDim = 1:numDims,
	fprintf('Dimension %d our of %d. ', iterDim, numDims);
	
	options.ReducedDim = reducedDim(iterDim);
	[Vec eigVal sampleMean] = PCA(fea_Train, options);
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train, fea_Test, sampleMean); 
	fprintf('PCA. ');
	[result accuracy_PCA(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);
	
	Vec = learn_sensing_exact(D_Graz, reducedDim(iterDim))';
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train, fea_Test); 
	fprintf('D Graz. ');
	[result accuracy_D_Graz(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);

	Vec = learn_sensing_exact(D_Graz_custom, reducedDim(iterDim))';
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train, fea_Test); 
	fprintf('D Graz custom. ');
	[result accuracy_D_Graz_custom(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, gnd_Test, gnd_Train);

	fprintf('\n');
end;
