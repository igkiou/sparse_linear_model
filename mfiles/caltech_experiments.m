%%
% combination = [318:321,322:325,326:329,666:669,670:673,674:677,66:68,689:691,681,680,679,678,692,69,630,632,688];
% K_train_train = zeros(3060);
% K_test_train = zeros([2995 3060]);
% for iter = 1:length(combination),
% 	load(sprintf('/home/igkiou/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_feat%d', combination(iter)));
% 	K_train_train = K_train_train + K / length(combination);
% 	load(sprintf('/home/igkiou/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_test_feat%d', combination(iter)));
% 	K_test_train = K_test_train + K / length(combination);
% end;
% load /home/igkiou/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_labels
% K_train_train = double(K_train_train);
% K_test_train = double(K_test_train);

%%
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_feataverage39.mat
K_train_train = K;
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_test_feataverage39.mat
K_test_train = K;
load ~/MATLAB/datasets_all/caltech_kernels/caltech101_nTrain30_N1_labels
K_train_train = double(K_train_train);
K_test_train = double(K_test_train);
KXX = K_train_train;
KYX = K_test_train;

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

%%
KDist = kernel_distance(KXX, 0);
KDist = - KDist;
KDistSqrt = kernel_distance(KXX, 1);
KDistSqrt = - KDistSqrt;

%%
% numSamples = size(KXX, 2);
% lambda = 0.00001;
% base = 0;
% batchSize = 600;
% A = zeros(numSamples - 1, numSamples);
% 
% for iter = (base + 1):(base + batchSize),
% 	fprintf('Now sparse coding sample %d.\n', iter);
% 	x = KXX(:, iter);
% 	inds = true(numSamples, 1);
% 	inds(iter) = false;
% 	D = KXX(:, inds);
% 	KDx = KXX(inds, iter);
% 	KDD = KXX(inds, inds);
% 	A(:, iter) = l1kernel_featuresign_mex(x, D, lambda, [], [], [], [], KDD, KDx);
% % 	A(:, iter) = KDD \ KDx;
% end;

%%
% numSamples = size(KXX, 2);
% lassoparams.lambda = 0.001;
% lassoparams.lambda2 = 0.00001;
% base = 0;
% batchSize = 600;
% A = zeros(numSamples - 1, numSamples);
% 
% for iter = (base + 1):(base + batchSize),
% 	fprintf('Now sparse coding sample %d.\n', iter);
% 	x = KXX(:, iter);
% 	inds = true(numSamples, 1);
% 	inds(iter) = false;
% 	D = KXX(:, inds);
% 	KDx = KXX(inds, iter);
% 	KDD = KXX(inds, inds);
% 	[UD LD] = eig(KDD);
% 	LD = diag(LD);
% 	sqrtLD = diag(sqrt(LD));
% 	invsqrtLD = diag(1 ./ diag(sqrtLD));
% 	Dr = sqrtLD * UD';
% 	xr = invsqrtLD * UD' * KDx;
% 	A(:, iter) = mexLasso(xr, Dr, lassoparams);
% end;

%%
% numSamples = size(A, 2);
% At = zeros(numSamples);
% for iter = 1:numSamples,
% 	fprintf('Now sparse coding sample %d.\n', iter);
% 	inds = true(numSamples, 1);
% 	inds(iter) = false;
% 	At(inds, iter) = A(:, iter);
% end;
% Am = abs(At) + abs(At');
% As = BuildAdjacency(At, 0);
% AtA = At' * At;

%%
% load caltech_en_A_l001l00001_small_All A At Am As AtA
% load caltech_sc_A_l0_small_All A At Am As AtA
% load caltech_sc_A_l0_small_results kernelACC kernelNMI kernelRand kernelRate
% load caltech_sc_A_l00001_small_All A At Am As AtA
% load caltech_sc_A_l00001_small_results kernelACC kernelNMI kernelRand kernelRate
% load caltech_sc_A_l0001_small_All A At Am As AtA
% load caltech_sc_A_l0001_small_results kernelACC kernelNMI kernelRand kernelRate
% load caltech_sc_A_l001_small_All A At Am As AtA
% load caltech_sc_A_l001_small_results kernelACC kernelNMI kernelRand kernelRate
% load caltech_sc_A_l01_small_All A At Am As AtA
% load caltech_sc_A_l01_small_results kernelACC kernelNMI kernelRand kernelRate
load caltech_sc_A_l1_small_All A At Am As AtA
% load caltech_sc_A_l1_small_results kernelACC kernelNMI kernelRand kernelRate

%%
As = abs(sign(As));
Am = abs(sign(Am));

%%
Grps = SpectralClustering(KXX, 20);
Grps1 = SpectralClustering(As, 20);
Grps2 = SpectralClustering(Am, 20);

Grps3 = apclusterK(KXX, 20, 0);
Grps4 = apclusterK(As, 20, 0);
Grps5 = apclusterK(Am, 20, 0);
Grps6 = SpectralClustering(AtA, 20);
Grps7 = apclusterK(AtA, 20, 0);
Grps8 = apclusterK(KDist, 20, 0);
Grps9 = apclusterK(KDistSqrt, 20, 0);
Grps10 = kernel_kmeans(KXX, 20)';

%%
exemplar_labels = unique(Grps3);
Grps3_mapped = zeros(size(Grps3));
for iter = 1:length(exemplar_labels),
	Grps3_mapped(Grps3 == exemplar_labels(iter)) = iter;
end;
Grps3 = Grps3_mapped;

exemplar_labels = unique(Grps4);
Grps4_mapped = zeros(size(Grps4));
for iter = 1:length(exemplar_labels),
	Grps4_mapped(Grps4 == exemplar_labels(iter)) = iter;
end;
Grps4 = Grps4_mapped;

exemplar_labels = unique(Grps5);
Grps5_mapped = zeros(size(Grps5));
for iter = 1:length(exemplar_labels),
	Grps5_mapped(Grps5 == exemplar_labels(iter)) = iter;
end;
Grps5 = Grps5_mapped;

exemplar_labels = unique(Grps7);
Grps7_mapped = zeros(size(Grps7));
for iter = 1:length(exemplar_labels),
	Grps7_mapped(Grps7 == exemplar_labels(iter)) = iter;
end;
Grps7 = Grps7_mapped;

exemplar_labels = unique(Grps8);
Grps8_mapped = zeros(size(Grps8));
for iter = 1:length(exemplar_labels),
	Grps8_mapped(Grps8 == exemplar_labels(iter)) = iter;
end;
Grps8 = Grps8_mapped;

exemplar_labels = unique(Grps9);
Grps9_mapped = zeros(size(Grps9));
for iter = 1:length(exemplar_labels),
	Grps9_mapped(Grps9 == exemplar_labels(iter)) = iter;
end;
Grps9 = Grps9_mapped;

%%
if (length(unique(Grps4)) ~= 20),
	Grps4 = Grps3;
	warning('Grps4 not consistent labeling.');
end;
if (length(unique(Grps5)) ~= 20),
	Grps5 = Grps3;
	warning('Grps5 not consistent labeling.');
end;
if (length(unique(Grps7)) ~= 20),
	Grps7 = Grps3;
	warning('Grps7 not consistent labeling.');
end;

%%
kernelNMI = zeros(11, 3);
kernelNMI(1, 1) = nmi(gnd, Grps(:,1)); kernelNMI(1, 2) = nmi(gnd, Grps(:,2)); kernelNMI(1, 3) = nmi(gnd, Grps(:,3));
kernelNMI(2, 1) = nmi(gnd, Grps1(:,1)); kernelNMI(2, 2) = nmi(gnd, Grps1(:,2)); kernelNMI(2, 3) = nmi(gnd, Grps1(:,3));
kernelNMI(3, 1) = nmi(gnd, Grps2(:,1)); kernelNMI(3, 2) = nmi(gnd, Grps2(:,2)); kernelNMI(3, 3) = nmi(gnd, Grps2(:,3));
kernelNMI(4, 3) = nmi(gnd, Grps3);
kernelNMI(5, 3) = nmi(gnd, Grps4);
kernelNMI(6, 3) = nmi(gnd, Grps5);
kernelNMI(7, 1) = nmi(gnd, Grps6(:,1)); kernelNMI(7, 2) = nmi(gnd, Grps6(:,2)); kernelNMI(7, 3) = nmi(gnd, Grps6(:,3));
kernelNMI(8, 3) = nmi(gnd, Grps7);
kernelNMI(9, 3) = nmi(gnd, Grps8);
kernelNMI(10, 3) = nmi(gnd, Grps9);
kernelNMI(11, 3) = nmi(gnd, Grps10);

%%
kernelRand = zeros(11, 3);
kernelRand(1, 1) = RandIndex(gnd, Grps(:,1)); kernelRand(1, 2) = RandIndex(gnd, Grps(:,2)); kernelRand(1, 3) = RandIndex(gnd, Grps(:,3));
kernelRand(2, 1) = RandIndex(gnd, Grps1(:,1)); kernelRand(2, 2) = RandIndex(gnd, Grps1(:,2)); kernelRand(2, 3) = RandIndex(gnd, Grps1(:,3));
kernelRand(3, 1) = RandIndex(gnd, Grps2(:,1)); kernelRand(3, 2) = RandIndex(gnd, Grps2(:,2)); kernelRand(3, 3) = RandIndex(gnd, Grps2(:,3));
kernelRand(4, 3) = RandIndex(gnd, Grps3);
kernelRand(5, 3) = RandIndex(gnd, Grps4);
kernelRand(6, 3) = RandIndex(gnd, Grps5);
kernelRand(7, 1) = RandIndex(gnd, Grps6(:,1)); kernelRand(7, 2) = RandIndex(gnd, Grps6(:,2)); kernelRand(7, 3) = RandIndex(gnd, Grps6(:,3));
kernelRand(8, 3) = RandIndex(gnd, Grps7);
kernelRand(9, 3) = RandIndex(gnd, Grps8);
kernelRand(10, 3) = RandIndex(gnd, Grps9);
kernelRand(11, 3) = RandIndex(gnd, Grps10);

%%
kernelACC = zeros(11, 3);
kernelACC(1, 1) = clustering_error(gnd, Grps(:,1)); kernelACC(1, 2) = clustering_error(gnd, Grps(:,2));...
	kernelACC(1, 3) = clustering_error(gnd, Grps(:,3));
kernelACC(2, 1) = clustering_error(gnd, Grps1(:,1)); kernelACC(2, 2) = clustering_error(gnd, Grps1(:,2));...
	kernelACC(2, 3) = clustering_error(gnd, Grps1(:,3));
kernelACC(3, 1) = clustering_error(gnd, Grps2(:,1)); kernelACC(3, 2) = clustering_error(gnd, Grps2(:,2));...
	kernelACC(3, 3) = clustering_error(gnd, Grps2(:,3));
kernelACC(4, 3) = clustering_error(gnd, Grps3);
kernelACC(5, 3) = clustering_error(gnd, Grps4);
kernelACC(6, 3) = clustering_error(gnd, Grps5);
kernelACC(7, 1) = clustering_error(gnd, Grps6(:,1)); kernelACC(7, 2) = clustering_error(gnd, Grps6(:,2)); kernelACC(7, 3) = clustering_error(gnd, Grps6(:,3));
kernelACC(8, 3) = clustering_error(gnd, Grps7);
kernelACC(9, 3) = clustering_error(gnd, Grps8);
kernelACC(10, 3) = clustering_error(gnd, Grps9);
kernelACC(11, 3) = clustering_error(gnd, Grps10);
kernelACC = (100 - kernelACC) / 100;

%%
kernelRate = zeros(11, 3);
kernelRate(1, 1) = clusterEvaluate(gnd, Grps(:,1)); kernelRate(1, 2) = clusterEvaluate(gnd, Grps(:,2));...
	kernelRate(1, 3) = clusterEvaluate(gnd, Grps(:,3));
kernelRate(2, 1) = clusterEvaluate(gnd, Grps1(:,1)); kernelRate(2, 2) = clusterEvaluate(gnd, Grps1(:,2));...
	kernelRate(2, 3) = clusterEvaluate(gnd, Grps1(:,3));
kernelRate(3, 1) = clusterEvaluate(gnd, Grps2(:,1)); kernelRate(3, 2) = clusterEvaluate(gnd, Grps2(:,2));...
	kernelRate(3, 3) = clusterEvaluate(gnd, Grps2(:,3));
kernelRate(4, 3) = clusterEvaluate(gnd, Grps3);
kernelRate(5, 3) = clusterEvaluate(gnd, Grps4);
kernelRate(6, 3) = clusterEvaluate(gnd, Grps5);
kernelRate(7, 1) = clusterEvaluate(gnd, Grps6(:,1)); kernelRate(7, 2) = clusterEvaluate(gnd, Grps6(:,2)); kernelRate(7, 3) = clusterEvaluate(gnd, Grps6(:,3));
kernelRate(8, 3) = clusterEvaluate(gnd, Grps7);
kernelRate(9, 3) = clusterEvaluate(gnd, Grps8);
kernelRate(10, 3) = clusterEvaluate(gnd, Grps9);
kernelRate(11, 3) = clusterEvaluate(gnd, Grps10);
kernelRate = kernelRate / 100;

%%
save caltech_bi_A_l1_small_results kernelACC kernelNMI kernelRand kernelRate Grps Grps1 Grps2 Grps3 Grps4 Grps5 Grps6 Grps7 Grps8 Grps9 Grps10
