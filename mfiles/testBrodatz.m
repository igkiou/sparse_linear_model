% load brodatz_experiments/brodatzDataAllUnnorm.mat
svmparams = setParameters;
svmparams.allVerboseMode = -1;

slideType = 'sliding';
largeExperiment = 0;
useNorm = 1;
patchSize = 8;
class1 = 12;
class2 = 17;
load(sprintf('brodatz_experiments/brodatz_%d%d_norm%d_patch%d_large%d_%s',...
	class1, class2, useNorm, patchSize, largeExperiment, slideType));

disp('Started training memory Gaussian dictionary.');
folder_contents = ls(sprintf('brodatz_experiments/brodatzGaussianDictionary%d%d_memory_*', class1, class2));
tempNames = regexp(folder_contents, '\s\s*', 'split');
numDicts = length(tempNames);
fid = fopen(sprintf('brodatz_%d%d_runs_gaussian.txt', class1, class2), 'wt');
for iterDict = 1:(numDicts - 1),
	load(sprintf('%s', tempNames{iterDict}));
	fprintf(fid, '%s\n', tempNames{iterDict});
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
	
	svmparams.svmLossFunction = 'hinge';
	fprintf('Sparse hinge. ');
	accuracy_hinge = run_svm(trainA, trainLabels, testA, testLabels, [], [], svmparams);
% 	accuracy_hinge = 0;
	svmparams.svmLossFunction = 'huber';
	fprintf('Sparse huber. ');
	accuracy_huber = run_svm(trainA, trainLabels, testA, testLabels, [], [], svmparams);
% 	accuracy_huber = 0;
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

folder_contents = ls(sprintf('brodatz_experiments/brodatzKernelDictionary%d%d_iter500_*', class1, class2));
tempNames = regexp(folder_contents, '\s\s*', 'split');
numDicts = length(tempNames);
fid = fopen(sprintf('brodatz_%d%d_runs_kernel.txt', class1, class2), 'wt');
for iterDict = 1:(numDicts - 1),
	load(sprintf('%s', tempNames{iterDict}));
	fprintf(fid, '%s\n', tempNames{iterDict});
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
	
	svmparams.svmLossFunction = 'hinge';
	fprintf('Sparse hinge. ');
	accuracy_hinge = run_svm(trainA, trainLabels, testA, testLabels, [], [], svmparams);
% 	accuracy_hinge = 0;
	svmparams.svmLossFunction = 'huber';
	fprintf('Sparse huber. ');
	accuracy_huber = run_svm(trainA, trainLabels, testA, testLabels, [], [], svmparams);
% 	accuracy_huber = 0;
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
