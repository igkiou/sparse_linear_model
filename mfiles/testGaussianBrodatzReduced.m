load brodatz_experiments/brodatzDataAllUnnorm.mat
svmparams = setParameters;
svmparams.allVerboseMode = -1;
reducedDim = [2 5 10:10:60 64];
numDims = length(reducedDim);

folder_contents = ls('brodatz_experiments/brodatzGaussianDictionaryAll*');
tempNames = regexp(folder_contents, '\s\s*', 'split');
numDicts = length(tempNames);
fid = fopen('brodatz_runs_reduced_gaussian.txt', 'wt');
for iterDict = 1:(numDicts - 1),
	load(sprintf('%s', tempNames{iterDict}));
	fprintf(fid, '%s\n', tempNames{iterDict});
	D_gaussian = D;
	accuracy_D_hinge = zeros(1, numDims);
	accuracy_D_huber = zeros(1, numDims);
	accuracy_D_knn = zeros(1, numDims);
	Vec = learn_sensing_exact(D_gaussian, reducedDim(end))';
	[fea_Train_Reduced_Large fea_Test_Reduced_Large] = reduceDimension(Vec, trainFeatures', testFeatures'); 

	for iterDim = 1:numDims,
		fprintf('Dimension %d our of %d. ', iterDim, numDims);
		fea_Train_Reduced = fea_Train_Reduced_Large(:, 1:reducedDim(iterDim));
		fea_Test_Reduced = fea_Test_Reduced_Large(:, 1:reducedDim(iterDim));
% 		svmparams.svmLossFunction = 'hinge';
% 		fprintf('D hinge. ');
% 		accuracy_D_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
% 		svmparams.svmLossFunction = 'huber';
% 		fprintf('D huber. ');
% 		accuracy_D_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
		fprintf('D KNN. ');
		[foo accuracy_D_knn(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testLabels', trainLabels');
		fprintf('\n');
	end;
	fprintf(fid, '%g ', accuracy_D_hinge);
	fprintf(fid, '\n');
	fprintf(fid, '%g ', accuracy_D_huber);
	fprintf(fid, '\n');
	fprintf(fid, '%g ', accuracy_D_knn);
	fprintf(fid, '\n');
	clear D params memoryparams gradientparams
end;
fclose(fid);
