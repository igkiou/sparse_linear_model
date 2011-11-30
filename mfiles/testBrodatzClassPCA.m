% load brodatz_experiments/brodatzDataAllUnnorm.mat
svmparams = setParameters;
svmparams.allVerboseMode = -1;
reducedDim = [2 5 10:10:60 64];
maxDimension = 100;
numDims = length(reducedDim);

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
	fid = fopen(sprintf('brodatz_%s_runs_PCA.txt', classString), 'wt');
	for iterNorm = useNorm,
		for iterPatchSize = patchSize,
			%load data
			load(sprintf('brodatz_experiments/brodatz_%s_norm%d_patch%d_large%d_%s',...
				classString, iterNorm, iterPatchSize, largeExperiment, slideType));
				
			% run Linear PCA experiments
			pcaName = sprintf('LinearPCA_%s_norm%d_patch%d_large%d_%s',...
				classString, iterNorm, iterPatchSize, largeExperiment, slideType);
			fprintf(fid, '%s\n', pcaName);
			fprintf('Case %s. ', pcaName);
			svmparams = setParameters;
			svmparams.allVerboseMode = -1;
			accuracy_PCA_hinge = zeros(1, numDims);
			accuracy_PCA_huber = zeros(1, numDims);
			accuracy_PCA_knn = zeros(1, numDims);
			options.ReducedDim = maxDimension;
			[Vec eigVal sampleMean] = PCA(trainFeatures', options);
			clear eigVal
			[fea_Train_Reduced_Large fea_Test_Reduced_Large] = reduceDimension(Vec, trainFeatures', testFeatures', sampleMean);
			
			for iterDim = 1:numDims,
				fprintf('Dimension %d our of %d. ', iterDim, numDims);
				fea_Train_Reduced = fea_Train_Reduced_Large(:, 1:reducedDim(iterDim));
				fea_Test_Reduced = fea_Test_Reduced_Large(:, 1:reducedDim(iterDim));
				svmparams.svmLossFunction = 'hinge';
				fprintf('PCA hinge. ');
				accuracy_PCA_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
				svmparams.svmLossFunction = 'huber';
				fprintf('PCA huber. ');
				accuracy_PCA_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
				fprintf('PCA kernel KNN. ');
				[foo accuracy_PCA_knn(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testLabels', trainLabels');
				fprintf('\n');
			end;
			fprintf(fid, '%g ', accuracy_PCA_hinge);
			fprintf(fid, '\n');
			fprintf(fid, '%g ', accuracy_PCA_huber);
			fprintf(fid, '\n');
			fprintf(fid, '%g ', accuracy_PCA_knn);
			fprintf(fid, '\n');
			clear fea_Train_Reduced_Large fea_Test_Reduced_Large Vec

			% run Kernel PCA experiments
			pcaName = sprintf('KernelPCA_%s_norm%d_patch%d_large%d_%s',...
				classString, iterNorm, iterPatchSize, largeExperiment, slideType);
			fprintf(fid, '%s\n', pcaName);
			fprintf('Case %s. ', pcaName);
			accuracy_PCA_kernel_hinge = zeros(1, numDims);
			accuracy_PCA_kernel_huber = zeros(1, numDims);
			accuracy_PCA_kernel_knn = zeros(1, numDims);

			load(sprintf('brodatz_experiments/brodatz_%s_PCA%d_norm%d_patch%d_large%d_%s',...
				classString, maxDimension, iterNorm, iterPatchSize, largeExperiment, slideType),...
				'fea_Train_Reduced_Large', 'fea_Test_Reduced_Large');

			for iterDim = 1:numDims,
				fprintf('Dimension %d our of %d. ', iterDim, numDims);
				fea_Train_Reduced = fea_Train_Reduced_Large(:, 1:reducedDim(iterDim));
				fea_Test_Reduced = fea_Test_Reduced_Large(:, 1:reducedDim(iterDim));
				svmparams.svmLossFunction = 'hinge';
				fprintf('PCA kernel hinge. ');
				accuracy_PCA_kernel_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
				svmparams.svmLossFunction = 'huber';
				fprintf('PCA kernel huber. ');
				accuracy_PCA_kernel_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
				fprintf('PCA kernel KNN. ');
				[foo accuracy_PCA_kernel_knn(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testLabels', trainLabels');
				fprintf('\n');
			end;
			fprintf(fid, '%g ', accuracy_PCA_kernel_hinge);
			fprintf(fid, '\n');
			fprintf(fid, '%g ', accuracy_PCA_kernel_huber);
			fprintf(fid, '\n');
			fprintf(fid, '%g ', accuracy_PCA_kernel_knn);
			fprintf(fid, '\n');
			clear fea_Train_Reduced_Large fea_Test_Reduced_Large Vec
		end;
	end;
	fclose(fid);
end;
