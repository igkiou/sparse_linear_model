% load brodatz_experiments/brodatzDataAllUnnorm.mat
svmparams = setParameters;
svmparams.allVerboseMode = -1;
reducedDim = [2 5 10:10:60 64];
numDims = length(reducedDim);

slideType = 'distinct';
largeExperiment = 1;
useNorm = [0];
patchSize = [12];
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
	fid = fopen(sprintf('brodatz_%s_runs_reduced_new.txt', classString), 'wt');
	for iterNorm = useNorm,
		for iterPatchSize = patchSize,
			%load data
			load(sprintf('brodatz_experiments/brodatz_%s_norm%d_patch%d_large%d_%s',...
				classString, iterNorm, iterPatchSize, largeExperiment, slideType));

% 			% run Gaussian experiments
% 			dictName = sprintf('brodatz_experiments/brodatzGaussianDictionary%s_memory_norm%d_patch%d_large%d_%s',...
% 				classString, iterNorm, iterPatchSize, largeExperiment, slideType);
% 			load(sprintf('%s', dictName));
% 			fprintf(fid, '%s\n', dictName);
% 			fprintf('Dictionary %s. ', dictName);
% 			D_gaussian = D;
% 			svmparams = setParameters;
% 			svmparams.allVerboseMode = -1;
% 			accuracy_D_hinge = zeros(1, numDims);
% 			accuracy_D_huber = zeros(1, numDims);
% 			accuracy_D_knn = zeros(1, numDims);
% 			Vec = learn_sensing_exact(D_gaussian, reducedDim(end))';
% 			[fea_Train_Reduced_Large fea_Test_Reduced_Large] = reduceDimension(Vec, trainFeatures', testFeatures'); 
% 
% 			for iterDim = 1:numDims,
% 				fprintf('Dimension %d our of %d. ', iterDim, numDims);
% 				fea_Train_Reduced = fea_Train_Reduced_Large(:, 1:reducedDim(iterDim));
% 				fea_Test_Reduced = fea_Test_Reduced_Large(:, 1:reducedDim(iterDim));
% 				svmparams.svmLossFunction = 'hinge';
% 				fprintf('D hinge. ');
% 				accuracy_D_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
% 				svmparams.svmLossFunction = 'huber';
% 				fprintf('D huber. ');
% 				accuracy_D_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
% 				fprintf('D KNN. ');
% 				[foo accuracy_D_knn(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testLabels', trainLabels');
% 				fprintf('\n');
% 			end;
% 			fprintf(fid, '%g ', accuracy_D_hinge);
% 			fprintf(fid, '\n');
% 			fprintf(fid, '%g ', accuracy_D_huber);
% 			fprintf(fid, '\n');
% 			fprintf(fid, '%g ', accuracy_D_knn);
% 			fprintf(fid, '\n');
% 			clear D D_gaussian memoryparams 

			% run Kernel experiments
% 			dictName = sprintf('brodatz_experiments/brodatzKernelDictionary%s_iter500_norm%d_patch%d_large%d_%s',...
% 				classString, iterNorm, iterPatchSize, largeExperiment, slideType);
			load('brodatz_experiments/_iter111.mat', 'kernelgradientparams');
			dictName = 'brodatz_experiments/_iter195.mat';
			load(sprintf('%s', dictName));
			fprintf(fid, '%s\n', dictName);
			fprintf('Dictionary %s. ', dictName);
			D_kernel = D;
			kernelparam1 = kernelgradientparams.kernelparam1;
			accuracy_D_kernel_hinge = zeros(1, numDims);
			accuracy_D_kernel_huber = zeros(1, numDims);
			accuracy_D_kernel_knn = zeros(1, numDims);

			disp('gramDD.');
			gramDD = kernel_gram_mex(D_kernel, [], 'g', kernelparam1);
			disp('gramTrainD.');
			gramTrainD = kernel_gram_mex(trainFeatures, D_kernel, 'g', kernelparam1);
			disp('gramTestD.');
			gramTestD = kernel_gram_mex(testFeatures, D_kernel, 'g', kernelparam1);
			Vec = learn_sensing_exact_kernel(D_kernel, reducedDim(end), [], gramDD, 'g', kernelparam1)';
			[fea_Train_Reduced_Large fea_Test_Reduced_Large] = reduceDimensionKernel(Vec, trainFeatures', testFeatures',...
					D_kernel, gramTrainD, gramTestD, 'g', kernelparam1);

			for iterDim = 1:numDims,
				fprintf('Dimension %d our of %d. ', iterDim, numDims);
				fea_Train_Reduced = fea_Train_Reduced_Large(:, 1:reducedDim(iterDim));
				fea_Test_Reduced = fea_Test_Reduced_Large(:, 1:reducedDim(iterDim));
				svmparams.svmLossFunction = 'hinge';
				fprintf('D kernel hinge. ');
				accuracy_D_kernel_hinge(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
				svmparams.svmLossFunction = 'huber';
				fprintf('D kernel huber. ');
				accuracy_D_kernel_huber(iterDim) = run_svm(fea_Train_Reduced', trainLabels, fea_Test_Reduced', testLabels, [], [], svmparams);
% 				fprintf('D kernel KNN. ');
% 				[foo accuracy_D_kernel_knn(iterDim)] = knn_classify(fea_Test_Reduced, fea_Train_Reduced, 1, testLabels', trainLabels');
% 				fprintf('\n');
			end;
			fprintf(fid, '%g ', accuracy_D_kernel_hinge);
			fprintf(fid, '\n');
			fprintf(fid, '%g ', accuracy_D_kernel_huber);
			fprintf(fid, '\n');
			fprintf(fid, '%g ', accuracy_D_kernel_knn);
			fprintf(fid, '\n');
			clear D D_kernel kernelgradientparams
		end;
	end;
	fclose(fid);
end;
