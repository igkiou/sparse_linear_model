datasetName = 'tiny';
datasetNameSecondary = 'tinyPCA';
load tiny_experiments/cifar_32x32_PCA.mat fea_Train_PCA fea_Test_PCA
load /home/igkiou/MATLAB/datasets_all/cifar-10-batches-mat/cifar_32x32.mat gnd_Train gnd_Test
trainFeatures = fea_Train_PCA';
testFeatures = fea_Test_PCA';
trainLabels = gnd_Train';
testLabels = gnd_Test';
svmparams = setParameters;
svmparams.allVerboseMode = -1;

folder_contents = ls(sprintf('%s_experiments/%sGaussianDictionary*', datasetName, datasetNameSecondary));
tempNames = regexp(folder_contents, '\s\s*', 'split');
numDicts = length(tempNames);
fid = fopen(sprintf('%s_runs_gaussian.txt', datasetNameSecondary), 'wt');
for iterDict = 1:(numDicts - 1),
	load(sprintf('%s', tempNames{iterDict}));
	fprintf(fid, '%s\n', tempNames{iterDict});
	fprintf('Dictionary %s. ', tempNames{iterDict});
	D_gaussian = D;
	if (exist('params', 'var')),
		lassoparams.lambda = params.dictionaryLambda;
	elseif (exist('memoryparams', 'var')),
		lassoparams.lambda = memoryparams.lambda;
	elseif (exist('gradientparams', 'var')),
		lassoparams.lambda = gradientparams.codinglambda;
	else
		error('Cannot find dictionary learning parameter file.');
	end;
	
	svmparams = setParameters;
	svmparams.allVerboseMode = -1;
	trainA = full(mexLasso(trainFeatures, D_gaussian, lassoparams));
	testA = full(mexLasso(testFeatures, D_gaussian, lassoparams));
	
% 	svmparams.svmLossFunction = 'hinge';
% 	fprintf('Sparse hinge. ');
% 	accuracy_hinge = run_svm(trainA, trainLabels, testA, testLabels, [], [], svmparams);
	accuracy_hinge = 0;
% 	svmparams.svmLossFunction = 'huber';
% 	fprintf('Sparse huber. ');
% 	accuracy_huber = run_svm(trainA, trainLabels, testA, testLabels, [], [], svmparams);
	accuracy_huber = 0;
	fprintf('Sparse KNN. ');
	[foo accuracy_knn] = knn_classify(testA', trainA', 1, testLabels', trainLabels');
	
	fprintf(fid, '%g ', accuracy_hinge);
	fprintf(fid, '\n');
	fprintf(fid, '%g ', accuracy_huber);
	fprintf(fid, '\n');
	fprintf(fid, '%g ', accuracy_knn);
	fprintf(fid, '\n');
	clear D params memoryparams gradientparams
end;
fclose(fid);

folder_contents = ls(sprintf('%s_experiments/%sKernelDictionary*', datasetName, datasetNameSecondary));
tempNames = regexp(folder_contents, '\s\s*', 'split');
numDicts = length(tempNames);
fid = fopen(sprintf('%s_runs_kernel.txt', datasetNameSecondary), 'wt');
for iterDict = 1:(numDicts - 1),
	load(sprintf('%s', tempNames{iterDict}));
	fprintf(fid, '%s\n', tempNames{iterDict});
	fprintf('Dictionary %s. ', tempNames{iterDict});
	D_kernel = D;
	if (exist('params', 'var')),
		lambda = params.dictionaryLambda;
		kernelparam1 = params.kernelParam1;
	elseif (exist('kernelgradientparams', 'var')),
		lambda = kernelgradientparams.codinglambda;
		kernelparam1 = kernelgradientparams.kernelparam1;
	else
		error('Cannot find dictionary learning parameter file.');
	end;
	
	svmparams = setParameters;
	svmparams.allVerboseMode = -1;
	trainA = l1kernel_featuresign_mex(trainFeatures, D_kernel, lambda, 'g', [], kernelparam1);
	testA = l1kernel_featuresign_mex(testFeatures, D_kernel, lambda, 'g', [], kernelparam1);
	
% 	svmparams.svmLossFunction = 'hinge';
% 	fprintf('Sparse hinge. ');
% 	accuracy_hinge = run_svm(trainA, trainLabels, testA, testLabels, [], [], svmparams);
	accuracy_hinge = 0;
% 	svmparams.svmLossFunction = 'huber';
% 	fprintf('Sparse huber. ');
% 	accuracy_huber = run_svm(trainA, trainLabels, testA, testLabels, [], [], svmparams);
	accuracy_huber = 0;
	fprintf('Sparse KNN. ');
	[foo accuracy_knn] = knn_classify(testA', trainA', 1, testLabels', trainLabels');
	
	fprintf(fid, '%g ', accuracy_hinge);
	fprintf(fid, '\n');
	fprintf(fid, '%g ', accuracy_huber);
	fprintf(fid, '\n');
	fprintf(fid, '%g ', accuracy_knn);
	fprintf(fid, '\n');
	clear D params kernelgradientparams
end;
fclose(fid);
