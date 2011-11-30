% load brodatz_experiments/brodatzDataAllUnnorm.mat
svmparams = setParameters;
svmparams.allVerboseMode = -1;

slideType = 'distinct';
largeExperiment = 1;
useNorm = [0 1];
patchSize = [8 10 12];
% class1Vec = [12 4 5 8];
% class2Vec = [17 84 92 84];
class1Vec = -1;
class2Vec = -1;
numClassPairs = length(class1Vec);

for iterClass = 1:numClassPairs,
	class1 = class1Vec(iterClass);
	class2 = class2Vec(iterClass);
	if ((class1 < 0) || (class2 < 0)),
		classString = 'All';
	else
		classString = sprintf('%d%d', class1, class2);
	end;
	fprintf('Class pair %s.\n', classString);
	fid = fopen(sprintf('brodatz_%s_runs.txt', classString), 'wt');
	for iterNorm = useNorm,
		for iterPatchSize = patchSize,
			%load data
			load(sprintf('brodatz_experiments/brodatz_%s_norm%d_patch%d_large%d_%s',...
				classString, iterNorm, iterPatchSize, largeExperiment, slideType));

			% run Gaussian experiments
			dictName = sprintf('brodatz_experiments/brodatzGaussianDictionary%s_memory_norm%d_patch%d_large%d_%s',...
				classString, iterNorm, iterPatchSize, largeExperiment, slideType);
			load(sprintf('%s', dictName));
			fprintf(fid, '%s\n', dictName);
			fprintf('Dictionary %s. ', dictName);
			D_gaussian = D;
			lassoparams.lambda = memoryparams.lambda;

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
			clear D D_gaussian memoryparams 

			% run Kernel experiments
			dictName = sprintf('brodatz_experiments/brodatzKernelDictionary%s_iter500_norm%d_patch%d_large%d_%s',...
				classString, iterNorm, iterPatchSize, largeExperiment, slideType);
			load(sprintf('%s', dictName));
			fprintf(fid, '%s\n', dictName);
			fprintf('Dictionary %s. ', dictName);
			D_kernel = D;
			lambda = kernelgradientparams.codinglambda;
			kernelparam1 = kernelgradientparams.kernelparam1;

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
			clear D D_kernel kernelgradientparams
		end;
	end;
	fclose(fid);
end;
