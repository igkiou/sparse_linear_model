reducedDim = 10:10:100;
numDims = length(reducedDim);
accuracy_PCA_hinge = zeros(1, numDims);
accuracy_PCA_huber = zeros(1, numDims);
accuracy_D_Graz_hinge = zeros(1, numDims);
accuracy_D_Graz_huber = zeros(1, numDims);
accuracy_D_Graz_custom_hinge = zeros(1, numDims);
accuracy_D_Graz_custom_huber = zeros(1, numDims);
svmparams = setParameters;
svmparams.allVerboseMode = -1;
% load Graz_test
fea_Train = trainX';
fea_Test = testX';
gnd_Train = trainY';
gnd_Test = testY';
% load graz_bikes_dict;
% D1 = D;
% load graz_bikes_dict_new
% D2 = D;
for iterDim = 1:numDims,
	fprintf('Dimension %d our of %d. ', iterDim, numDims);
	
	options.ReducedDim = reducedDim(iterDim);
	[Vec eigVal sampleMean] = PCA(fea_Train, options);
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train, fea_Test, sampleMean); 
	svmparams.svmLossFunction = 'hinge';
	fprintf('PCA hinge. ');
	accuracy_PCA_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('PCA huber. ');
	accuracy_PCA_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', svmparams);
	
	Vec = learn_sensing_exact(D1, reducedDim(iterDim))';
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train, fea_Test); 
	svmparams.svmLossFunction = 'hinge';
	fprintf('D Graz hinge. ');
	accuracy_D_Graz_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('D Graz huber. ');
	accuracy_D_Graz_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', svmparams);

	Vec = learn_sensing_exact(D2, reducedDim(iterDim))';
	[fea_Train_Reduced fea_Test_Reduced] = reduceDimension(Vec, fea_Train, fea_Test); 
	svmparams.svmLossFunction = 'hinge';
	fprintf('D Graz custom hinge. ');
	accuracy_D_Graz_custom_hinge(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', svmparams);
	svmparams.svmLossFunction = 'huber';
	fprintf('D Graz custom huber. ');
	accuracy_D_Graz_custom_huber(iterDim) = run_svm(fea_Train_Reduced', gnd_Train', fea_Test_Reduced', gnd_Test', svmparams);

	fprintf('\n');
end;

% load ../results/Graz_experiments/sift_features/carsTestSIFTFeatures testLabels testSIFTFeatures
% testX = testSIFTFeatures(:, testLabels == 1);
% load ../results/Graz_experiments/sift_features/bikeTestSIFTFeatures testLabels testSIFTFeatures
% testX = [testX testSIFTFeatures(:, testLabels == 1)];
% load ../results/Graz_experiments/sift_features/personTestSIFTFeatures testLabels testSIFTFeatures
% testX = [testX testSIFTFeatures(:, testLabels == 1)];
% params.K = 512;
% params.iter = 1000;
% params.batchsize = 45000;
% params.lambda = 0.1;
% D = mexTrainDL_Memory(trainX, params);
