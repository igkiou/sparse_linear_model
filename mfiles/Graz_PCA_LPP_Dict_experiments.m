%%
load ../results/Graz_experiments/Graz_test.mat trainX trainY %testX testY
trainX = trainX(:,1:100:end);
trainY = trainY(:,1:100:end);
gnd = trainY + 1;

% %%
% params.codinglambda = 0.1;
% params.iternum = 100;
% params.blockratio = 0.1;
% params.printinfo = 1;
% params.errorinfo = 0;
% params.distclear = 0;
% params.dictsize = 1000;
% params.initdict = randn(3060,1000);
% D = kerneldictgradient_representer(KXX, params);
% 
%%
% [VecLPP W] = trainLPP(trainX', trainY', 100, 'Cosine', 'KNN');
% [fea_Train_Reduced_Large_LPP, fea_Test_Reduced_Large_LPP] = reduceDimension(VecLPP, trainX', []);
% fea_Train_Reduced_Large_LPP = fea_Train_Reduced_Large_LPP';
% fea_Test_Reduced_Large_LPP = fea_Test_Reduced_Large_LPP';

%%
[VecPCA meanSample] = trainPCA(trainX', trainY', 100);
[fea_Train_Reduced_Large_PCA, mapping] = reduceDimension(VecPCA, trainX', [], meanSample);
fea_Train_Reduced_Large_PCA = fea_Train_Reduced_Large_PCA';
% fea_Test_Reduced_Large_PCA = fea_Test_Reduced_Large_PCA';

%%
load ../results/Graz_experiments/graz_dict D
VecDict = learn_sensing_exact(D, 100)';
[fea_Train_Reduced_Large_Dict fea_Test_Reduced_Large_Dict] = reduceDimension(VecDict, trainX', []);
fea_Train_Reduced_Large_Dict = fea_Train_Reduced_Large_Dict';
fea_Test_Reduced_Large_Dict = fea_Test_Reduced_Large_Dict';

%%
load ../results/Graz_experiments/graz_dict_new D
VecDict1 = learn_sensing_exact(D, 100)';
[fea_Train_Reduced_Large_Dict1 fea_Test_Reduced_Large_Dict1] = reduceDimension(VecDict1, trainX', []);
fea_Train_Reduced_Large_Dict1 = fea_Train_Reduced_Large_Dict1';
% fea_Test_Reduced_Large_Dict1 = fea_Test_Reduced_Large_Dict1';

dimensions = [5 10:10:100];
numDimensions = length(dimensions);

MAXiter = 1000; % Maximum iteration for KMeans Algorithm
REPlic = 50; % Replication for KMeans Algorithm
opts = statset('Display','off');

methods = {'PCA', 'Dict', 'Dict1'}; %'LPP', 
numMethods = length(methods);
kernelNMI = zeros(numDimensions, 6, numMethods);
kernelRand = zeros(numDimensions, 6, numMethods);
kernelACC = zeros(numDimensions, 6, numMethods);
kernelRate = zeros(numDimensions, 6, numMethods);

%%
for iterMeth = 1:numMethods,
	method = methods{iterMeth};
	fprintf('Now running method %s.\n', method);
	for iterDim = 1:numDimensions,
		fprintf('Now running dimension %d/%d.\n', iterDim, numDimensions);
		eval(sprintf('fea_Train = fea_Train_Reduced_Large_%s(1:dimensions(iterDim), :);', method));
		KXXLPP = l2_distance(fea_Train);
		KXXLPP = - KXXLPP;
		KXXLPPSq = l2_distance(fea_Train, [], 1);
		KXXLPPSq = - KXXLPPSq;
		KIP = fea_Train' * fea_Train;

% 		Grps1 = apclusterK(KXXLPP, 3, 0);
% 		Grps2 = apclusterK(KXXLPPSq, 3, 0);
		Grps3 = kmeans(fea_Train', 3, 'start', 'sample', 'maxiter', MAXiter,...
			'replicates', REPlic, 'EmptyAction', 'singleton', 'Options', opts);
		Grps4 = SpectralClusteringAlt(KIP, 3);
		Grps1 = Grps3;
		Grps2 = Grps3;
		
		exemplar_labels = unique(Grps1);
		Grps1_mapped = zeros(size(Grps1));
		for iter = 1:length(exemplar_labels),
			Grps1_mapped(Grps1 == exemplar_labels(iter)) = iter;
		end;
		Grps1 = Grps1_mapped;
		exemplar_labels = unique(Grps2);
		Grps2_mapped = zeros(size(Grps2));
		for iter = 1:length(exemplar_labels),
			Grps2_mapped(Grps2 == exemplar_labels(iter)) = iter;
		end;
		Grps2 = Grps2_mapped;

		if (length(unique(Grps1)) == 3),
			kernelNMI(iterDim, 1, iterMeth) = nmi(gnd, Grps1);
			kernelRand(iterDim, 1, iterMeth) = RandIndex(gnd, Grps1);
			kernelACC(iterDim, 1, iterMeth) = (100 - clustering_error(gnd, Grps1)) / 100;
			kernelRate(iterDim, 1, iterMeth) = clusterEvaluate(gnd, Grps1) / 100;
		end;

		if (length(unique(Grps2)) == 3),
			kernelNMI(iterDim, 2, iterMeth) = nmi(gnd, Grps2);
			kernelRand(iterDim, 2, iterMeth) = RandIndex(gnd, Grps2);
			kernelACC(iterDim, 2, iterMeth) = (100 - clustering_error(gnd, Grps2)) / 100;
			kernelRate(iterDim, 2, iterMeth) = clusterEvaluate(gnd, Grps2) / 100;
		end;

		if (length(unique(Grps3)) == 3),
			kernelNMI(iterDim, 3, iterMeth) = nmi(gnd, Grps3);
			kernelRand(iterDim, 3, iterMeth) = RandIndex(gnd, Grps3);
			kernelACC(iterDim, 3, iterMeth) = (100 - clustering_error(gnd, Grps3)) / 100;
			kernelRate(iterDim, 3, iterMeth) = clusterEvaluate(gnd, Grps3) / 100;
		end;

		if (length(unique(Grps4(:, 1))) == 3),
			kernelNMI(iterDim, 4, iterMeth) = nmi(gnd, Grps4(:, 1));
			kernelRand(iterDim, 4, iterMeth) = RandIndex(gnd, Grps4(:, 1));
			kernelACC(iterDim, 4, iterMeth) = (100 - clustering_error(gnd, Grps4(:, 1))) / 100;
			kernelRate(iterDim, 4, iterMeth) = clusterEvaluate(gnd, Grps4(:, 1)) / 100;
		end;

		if (length(unique(Grps4(:, 2))) == 3),
			kernelNMI(iterDim, 5, iterMeth) = nmi(gnd, Grps4(:, 2));
			kernelRand(iterDim, 5, iterMeth) = RandIndex(gnd, Grps4(:, 2));
			kernelACC(iterDim, 5, iterMeth) = (100 - clustering_error(gnd, Grps4(:, 2))) / 100;
			kernelRate(iterDim, 5, iterMeth) = clusterEvaluate(gnd, Grps4(:, 2)) / 100;
		end;

		if (length(unique(Grps4(:, 3))) == 3),
			kernelNMI(iterDim, 6, iterMeth) = nmi(gnd, Grps4(:, 3));
			kernelRand(iterDim, 6, iterMeth) = RandIndex(gnd, Grps4(:, 3));
			kernelACC(iterDim, 6, iterMeth) = (100 - clustering_error(gnd, Grps4(:, 3))) / 100;
			kernelRate(iterDim, 6, iterMeth) = clusterEvaluate(gnd, Grps4(:, 3)) / 100;
		end;
	end;
end;
