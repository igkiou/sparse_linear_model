% % cd ~/MATLAB/sparse_linear_model/MNIST_experiments/
% % loadMNISTdata
% % trainData = normcols(trainData);
% % inds = (trainLabels == 1) | (trainLabels == 3) | (trainLabels == 0);
% % params = setParameters;
% % params.dictionarySize = 28 * 28;
% % params.dictionaryMethod = 'memory';
% % trainX = trainData(:, inds);
% % [D Phi] = dictionary_learning(trainX, [], [], params);
% % figure;showdict(D,[28 28], 28, 28,'whitelines','highcontrast');
% % trainA = l1qp_featuresign_mex(trainX, D, 0.15);
% % save MNISTclusteringTrainSmallMemory013 D trainA trainX
% 
% loadMNISTdata
% trainData = normcols(trainData);
% load MNISTclusteringTrainSmallMemory013 % D trainA
% global_options
% 
% ndim = 10;
% inds = (trainLabels == 1) | (trainLabels == 3);
% X = trainData(:, inds);
% X = trainX;
% A = trainA;
% labels = trainLabels(inds);
% 
% L = learn_sensing_exact(D, ndim);
% LPPCos = trainLPP(X', labels', ndim, 'Cosine', 'KNN')';
% LPPEu = trainLPP(X', labels', ndim, 'Euclidean', 'KNN')';
% LPCA = trainPCA(X', labels', ndim)';
% 
% reducedL = L * X;
% reducedLPPCos = LPPCos * X;
% reducedLPPEu = LPPEu * X;
% reducedLPCA = LPCA * X;
% 
% % WOrig = createW(X', labels', 'Euclidean', 'KNN');
% % WSparseProd = abs(A'*A);
% % WSparseEu = createW(A', labels', 'Euclidean', 'KNN');
% % WSparseCos = createW(A', labels', 'Cosine', 'KNN');
% % WL = createW(reducedL', labels', 'Euclidean', 'KNN');
% % WLPPCos = createW(reducedLPPCos', labels', 'Cosine', 'KNN');
% % WLPPEu = createW(reducedLPPEu', labels', 'Euclidean', 'KNN');
% % WLPCA = createW(reducedLPCA', labels', 'Euclidean', 'KNN');
% % 
% % assignmentOrig = mcut_kmeans(WOrig, 2);
% % assignmentSparseProd = mcut_kmeans(WSparseProd, 2);
% % assignmentSparseEu = mcut_kmeans(WSparseEu, 2);
% % assignmentSparseCos = mcut_kmeans(WSparseCos, 2);
% % assignmentL = mcut_kmeans(WL, 2);
% % assignmentLPPCos = mcut_kmeans(WLPPCos, 2);
% % assignmentLPPEu = mcut_kmeans(WLPPEu, 2);
% % assignmentLPCA = mcut_kmeans(WLPCA, 2);
% 
% assignmentOrig = kmeans(X', 3);
% assignmentSparseProd = assignmentOrig;
% assignmentSparseEu = kmeans(A', 3);
% assignmentSparseCos = assignmentSparseEu;
% assignmentL = kmeans(reducedL', 3);
% assignmentLPPCos = kmeans(reducedLPPCos', 3);
% assignmentLPPEu = kmeans(reducedLPPEu', 3);
% assignmentLPCA = kmeans(reducedLPCA', 3);
% 
% [id(1, :) acc(1, :) tacc(1, :)] = clusterEvaluate(labels', assignmentOrig);
% [id(2, :) acc(2, :) tacc(2, :)] = clusterEvaluate(labels', assignmentSparseProd);
% [id(3, :) acc(3, :) tacc(3, :)] = clusterEvaluate(labels', assignmentSparseEu);
% [id(4, :) acc(4, :) tacc(4, :)] = clusterEvaluate(labels', assignmentSparseCos);
% [id(5, :) acc(5, :) tacc(5, :)] = clusterEvaluate(labels', assignmentL);
% [id(6, :) acc(6, :) tacc(6, :)] = clusterEvaluate(labels', assignmentLPPCos);
% [id(7, :) acc(7, :) tacc(7, :)] = clusterEvaluate(labels', assignmentLPPEu);
% [id(8, :) acc(8, :) tacc(8, :)] = clusterEvaluate(labels', assignmentLPCA);

%%
useNorm = [0 1];
patchSize = [8 10 12];
% fiveClasses = {'5c', '5m', '5v', '5v2', '5v3'};
% class1 = fiveClasses{1};
% class2 = fiveClasses{1};
% class1Vec = [12 4 5 8];
% class2Vec = [17 84 92 84];
class1Vec = -1;
class2Vec = -1;
numClassPairs = length(class1Vec);
reducedDim = [2 5 10:10:60 64];
numDimensions = reducedDim(4);

fid = fopen(sprintf('brodatz_all_cluster_runs.txt'), 'wt');
for iterPairs = 1:numClassPairs,
	class1 = class1Vec(iterPairs);
	class2 = class2Vec(iterPairs);
	for iterNorm = useNorm,
		for iterPatch = patchSize,
			if (ischar(class1) || ischar(class2)),
				if (~strcmp(class1, class2)),
					warning('Strings class1=%s and class2=%s are not the same.', class1, class2);
				end;
				classString = class1;
				numClusters = 5;
				slideType = 'sliding';
				largeExperiment = 0;
			elseif ((class1 < 0) || (class2 < 0)),
				if (class1 ~= class2),
					warning('Integers class1=%d and class2=%d are not the same.', class1, class2);
				end;
				classString = 'All';
				numClusters = 7;
				slideType = 'distinct';
				largeExperiment = 1;
			else
				classString = sprintf('%d%d', class1, class2);
				numClusters = 2;
				slideType = 'sliding';
				largeExperiment = 0;
			end;
			caseName = sprintf('brodatz_experiments/brodatz_%s_norm%d_patch%d_large%d_%s',...
					classString, iterNorm, iterPatch, largeExperiment, slideType);
			fprintf(fid, '%s\n', caseName);
			fprintf('%s\n', caseName);
			load(caseName);

			load(sprintf('brodatz_experiments/brodatzGaussianDictionary%s_memory_norm%d_patch%d_large%d_%s',...
					classString, iterNorm, iterPatch, largeExperiment, slideType));
			D_gaussian = D;
			testAGaussian = full(mexLasso(testFeatures, D_gaussian, memoryparams));

			load(sprintf('brodatz_experiments/brodatzKernelDictionary%s_iter500_norm%d_patch%d_large%d_%s',...
						classString, iterNorm, iterPatch, largeExperiment, slideType));
			D_kernel = D;
			testAKernel = l1kernel_featuresign_mex(testFeatures, D_kernel, kernelgradientparams.codinglambda, 'g', [], kernelgradientparams.kernelparam1);

			
			Vec = learn_sensing_exact(D_gaussian, numDimensions)';
			testReducedGaussian = reduceDimension(Vec, testFeatures', [])'; 
			
			disp('gramDD.');
			gramDD = kernel_gram_mex(D_kernel, [], 'g', kernelgradientparams.kernelparam1);
			disp('gramTestD.');
			gramTestD = kernel_gram_mex(testFeatures, D_kernel, 'g', kernelgradientparams.kernelparam1);
			Vec = learn_sensing_exact_kernel(D_kernel, numDimensions, [], gramDD, 'g', kernelgradientparams.kernelparam1)';
			testReducedKernel = reduceDimensionKernel(Vec, testFeatures', [], D_kernel, gramTestD, [], 'g', kernelgradientparams.kernelparam1);
				
			options.ReducedDim = numDimensions;
			[Vec eigVal sampleMean] = PCA(testFeatures', options);
			clear eigVal
			testReducedPCA = reduceDimension(Vec, testFeatures', [], sampleMean)';
			
			
%%
% WOrig = createW(testFeatures', testLabels', 'Euclidean', 'KNN');
% WSparseGauss = createW(testAGaussian', testLabels', 'Euclidean', 'KNN');
% WSparseKernel = createW(testAKernel', testLabels', 'Euclidean', 'KNN');
% 
% global_options;
% indsNcutsOrig = mcut_kmeans(WOrig, numClusters);
% indsNcutsGauss = mcut_kmeans(WSparseGauss, numClusters);
% indsNcutsKernel = mcut_kmeans(WSparseKernel, numClusters);

%% 
			[foo bar indsKmeansOrig] = mpi_kmeans(testFeatures, numClusters); indsKmeansOrig = double(indsKmeansOrig);
			[foo bar indsKmeansGauss] = mpi_kmeans(testAGaussian, numClusters); indsKmeansGauss = double(indsKmeansGauss);
			[foo bar indsKmeansKernel] = mpi_kmeans(testAKernel, numClusters); indsKmeansKernel = double(indsKmeansKernel);
			[foo bar indsKmeansReducedGauss] = mpi_kmeans(testReducedGaussian, numClusters); indsKmeansReducedGauss = double(indsKmeansReducedGauss);
			[foo bar indsKmeansReducedKernel] = mpi_kmeans(testReducedKernel, numClusters); indsKmeansReducedKernel = double(indsKmeansReducedKernel);
			[foo bar indsKmeansReducedPCA] = mpi_kmeans(testReducedPCA, numClusters); indsKmeansReducedPCA = double(indsKmeansReducedPCA);

			[a b c1] = clusterEvaluate(testLabels, indsKmeansOrig);
			[a b c2] = clusterEvaluate(testLabels, indsKmeansGauss);
			[a b c3] = clusterEvaluate(testLabels, indsKmeansKernel);
			[a b c4] = clusterEvaluate(testLabels, indsKmeansReducedGauss);
			[a b c5] = clusterEvaluate(testLabels, indsKmeansReducedKernel);
			[a b c6] = clusterEvaluate(testLabels, indsKmeansReducedPCA);
			acc = [c1 c2 c3 c4 c5 c6];
			fprintf(fid, '%g ', acc);
			fprintf(fid, '\n');
		end;
	end;
end;
