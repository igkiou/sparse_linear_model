load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_feataverage39.mat
K_train_train = K;
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_test_feataverage39.mat
K_test_train = K;
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_labels
K_train_train = double(K_train_train);
K_test_train = double(K_test_train);
KXX = K_train_train;

%%
load caltech_names
KXX = KXX(inds_subset, inds_subset);
tr_label = tr_label(inds_subset);
legend = 1:length(labels_subset);
tr_label_mapped = zeros(size(tr_label));
for iter = 1:length(labels_subset),
	tr_label_mapped(tr_label == labels_subset(iter)) = legend(iter);
end;
gnd = tr_label_mapped;

load caltech_sc_A_l0_small_All As
distanceMatrix = As;
[Srt,Ind] = sort(distanceMatrix, 1, 'descend');
clear Srt

dimensions = 10:5:150;
neighbors = [1 2 3 4 5:5:30];
numNeighborSetting = length(neighbors);
numDimensions = length(dimensions);

MAXiter = 1000; % Maximum iteration for KMeans Algorithm
REPlic = 10; % Replication for KMeans Algorithm
opts = statset('Display','iter');

kernelNMI = zeros(numNeighborSetting, numDimensions, 6);
kernelRand = zeros(numNeighborSetting, numDimensions, 6);
kernelACC = zeros(numNeighborSetting, numDimensions, 6);
kernelRate = zeros(numNeighborSetting, numDimensions, 6);

%%
for iterNeigh = 1:numNeighborSetting,
	fprintf('Now running neighbor %d/%d.\n', iterNeigh, numNeighborSetting);
	numNeighbors = neighbors(iterNeigh);
	WEu = zeros(size(distanceMatrix));
	for iter = 1:size(distanceMatrix,1),
		WEu(Ind(1:(numNeighbors + 1),iter),iter) = 1;
	end;
	WEu = max(WEu, WEu');

	Vec = trainKernelSRLPP([], [], dimensions(end), 'Cosine', 'KNN', KXX, WEu);
	fea_Train_Reduced_Large = KXX * Vec;
	fea_Train_Reduced_Large = fea_Train_Reduced_Large';

	for iterDim = 1:numDimensions,
		fprintf('Now running dimension %d/%d.\n', iterDim, numDimensions);
		fea_Train = fea_Train_Reduced_Large(1:dimensions(iterDim), :);
% 		KXXLPP = l2_distance(fea_Train);
% 		KXXLPP = - KXXLPP;
% 		KXXLPPSq = l2_distance(fea_Train, [], 1);
% 		KXXLPPSq = - KXXLPPSq;
		KIP = fea_Train' * fea_Train;

% 		Grps1 = apclusterK(KXXLPP, 20, 0);
% 		Grps2 = apclusterK(KXXLPPSq, 20, 0);
% 		Grps3 = kmeans(fea_Train', 20, 'start', 'sample', 'maxiter', MAXiter,...
% 			'replicates', REPlic, 'EmptyAction', 'singleton', 'Options', opts);
		Grps4 = SpectralClusteringAlt(KIP, 20);
		
% 		exemplar_labels = unique(Grps1);
% 		Grps1_mapped = zeros(size(Grps1));
% 		for iter = 1:length(exemplar_labels),
% 			Grps1_mapped(Grps1 == exemplar_labels(iter)) = iter;
% 		end;
% 		Grps1 = Grps1_mapped;
% 		exemplar_labels = unique(Grps2);
% 		Grps2_mapped = zeros(size(Grps2));
% 		for iter = 1:length(exemplar_labels),
% 			Grps2_mapped(Grps2 == exemplar_labels(iter)) = iter;
% 		end;
% 		Grps2 = Grps2_mapped;
% 		
% 		if (length(unique(Grps1)) == 20),
% 			kernelNMI(iterNeigh, iterDim, 1) = nmi(gnd, Grps1);
% 			kernelRand(iterNeigh, iterDim, 1) = RandIndex(gnd, Grps1);
% 			kernelACC(iterNeigh, iterDim, 1) = (100 - clustering_error(gnd, Grps1)) / 100;
% 			kernelRate(iterNeigh, iterDim, 1) = clusterEvaluate(gnd, Grps1) / 100;
% 		end;
% 		
% 		if (length(unique(Grps2)) == 20),
% 			kernelNMI(iterNeigh, iterDim, 2) = nmi(gnd, Grps2);
% 			kernelRand(iterNeigh, iterDim, 2) = RandIndex(gnd, Grps2);
% 			kernelACC(iterNeigh, iterDim, 2) = (100 - clustering_error(gnd, Grps2)) / 100;
% 			kernelRate(iterNeigh, iterDim, 2) = clusterEvaluate(gnd, Grps2) / 100;
% 		end;
% 		
% 		if (length(unique(Grps3)) == 20),
% 			kernelNMI(iterNeigh, iterDim, 3) = nmi(gnd, Grps3);
% 			kernelRand(iterNeigh, iterDim, 3) = RandIndex(gnd, Grps3);
% 			kernelACC(iterNeigh, iterDim, 3) = (100 - clustering_error(gnd, Grps3)) / 100;
% 			kernelRate(iterNeigh, iterDim, 3) = clusterEvaluate(gnd, Grps3) / 100;
% 		end;
		
		if (length(unique(Grps4(:, 1))) == 20),
			kernelNMI(iterNeigh, iterDim, 4) = nmi(gnd, Grps4(:, 1));
			kernelRand(iterNeigh, iterDim, 4) = RandIndex(gnd, Grps4(:, 1));
			kernelACC(iterNeigh, iterDim, 4) = (100 - clustering_error(gnd, Grps4(:, 1))) / 100;
			kernelRate(iterNeigh, iterDim, 4) = clusterEvaluate(gnd, Grps4(:, 1)) / 100;
		end;
		
		if (length(unique(Grps4(:, 2))) == 20),
			kernelNMI(iterNeigh, iterDim, 5) = nmi(gnd, Grps4(:, 2));
			kernelRand(iterNeigh, iterDim, 5) = RandIndex(gnd, Grps4(:, 2));
			kernelACC(iterNeigh, iterDim, 5) = (100 - clustering_error(gnd, Grps4(:, 2))) / 100;
			kernelRate(iterNeigh, iterDim, 5) = clusterEvaluate(gnd, Grps4(:, 2)) / 100;
		end;
		
		if (length(unique(Grps4(:, 3))) == 20),
			kernelNMI(iterNeigh, iterDim, 6) = nmi(gnd, Grps4(:, 3));
			kernelRand(iterNeigh, iterDim, 6) = RandIndex(gnd, Grps4(:, 3));
			kernelACC(iterNeigh, iterDim, 6) = (100 - clustering_error(gnd, Grps4(:, 3))) / 100;
			kernelRate(iterNeigh, iterDim, 6) = clusterEvaluate(gnd, Grps4(:, 3)) / 100;
		end;
	end;
end;
