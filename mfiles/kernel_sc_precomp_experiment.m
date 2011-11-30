load ~/MATLAB/datasets_all/MNIST/MNIST_28x28.mat
Y = imnorm(fea_Test', [-1 1]);
Y = Y(:, (gnd_Test == 0) | (gnd_Test == 1) | (gnd_Test == 2) | (gnd_Test == 3) | (gnd_Test == 4) | (gnd_Test == 5));
gnd = gnd_Test((gnd_Test == 0) | (gnd_Test == 1) | (gnd_Test == 2) | (gnd_Test == 3) | (gnd_Test == 4) | (gnd_Test == 5));
gnd = gnd + 1;
% clear fea_Train fea_Test gnd_Train gnd_Test
% gamma = 0.00728932024638;
% param1 = sqrt(0.5/gamma);
% KYY = kernel_gram_mex(Y, [], 'g', param1);
% numSamples = size(Y, 2);
% lambda = 0.001;
% base = 3000;
% batchSize = 2;
% A = zeros(numSamples - 1, numSamples);
% 
% for iter = (base + 1):(base + batchSize),
% 	fprintf('Now sparse coding sample %d.\n', iter);
% 	x = Y(:,iter);
% 	inds = true(numSamples, 1);
% 	inds(iter) = false;
% 	D = Y(:, inds);
% 	KDx = KYY(inds, iter);
% 	KDD = KYY(inds, inds);
% 	A(:, iter) = l1kernel_featuresign_mex(x, D, lambda, 'G', [], param1, [], KDD, KDx);
% end;

% clear all
% load ~/MATLAB/sparse_linear_model/MNIST_experiments/MNISTGaussianDictionaryLarge_verylong_memory.mat
% KDD = D'*D;
% load ~/MATLAB/datasets_all/MNIST/MNIST_28x28.mat
% X = normcols(fea_Train');
% x = X(:, 1:10000);
% KDx = D'*x;
% lambda = 0.15;
% B = l1qp_featuresign_mex(x, D, lambda, [], KDD, KDx);
% B1 = l1qp_featuresign_mex(x, D, lambda);
% B2 = l1qp_featuresign_mex_old(x, D, lambda);
% B3 = l1qp_featuresign(x, D, lambda);

% numSamples = 6031;
% At = zeros(numSamples);
% for iter = 1:6031,
% 	fprintf('Now sparse coding sample %d.\n', iter);
% 	inds = true(numSamples, 1);
% 	inds(iter) = false;
% 	At(inds, iter) = A(:, iter);
% end;

X1sq = diag(KYY);
distanceMatrix = bsxfun(@plus, X1sq, bsxfun(@minus, X1sq', 2 * KYY));
clear X1tX1 X1sq
distanceMatrix = max(distanceMatrix, distanceMatrix');
distanceMatrix = distanceMatrix - diag(diag(distanceMatrix));
KYYd = - distanceMatrix;
KYYdsq = - sqrt(distanceMatrix);

% load MNIST_sc_A_l001_05_All Am As Ampsd Aspsd
% [Grps , SingVals, LapKernel] = SpectralClustering(KYY, 6);
% [Grps1 , SingVals, LapKernel] = SpectralClustering(As,6);
% [Grps2 , SingVals, LapKernel] = SpectralClustering(Am,6);
% Grps3 = apclusterK(KYY, 6, 0);
Grps4 = apclusterK(As, 6, 0);
Grps5 = apclusterK(Am, 6, 0);
Grps6 = apclusterK(KYYd, 6, 0);
Grps7 = apclusterK(KYYdsq, 6, 0);
%% 
% Y1 = imnorm(fea_Test', [-1 1]); 
% Y1 = Y1(:, (gnd_Test == 0) | (gnd_Test == 1) | (gnd_Test == 2) | (gnd_Test == 3) | (gnd_Test == 4) | (gnd_Test == 5));
% KYY1 = Y1' * Y1;
% Y2 = normcols(fea_Test'); 
% Y2 = Y2(:, (gnd_Test == 0) | (gnd_Test == 1) | (gnd_Test == 2) | (gnd_Test == 3) | (gnd_Test == 4) | (gnd_Test == 5));
% KYY2 = Y2' * Y2;
% Y3 = imnorm(fea_Test', [0 1]); 
% Y3 = Y3(:, (gnd_Test == 0) | (gnd_Test == 1) | (gnd_Test == 2) | (gnd_Test == 3) | (gnd_Test == 4) | (gnd_Test == 5));
% KYY3 = Y3' * Y3;
% MAXiter = 1000; % Maximum iteration for KMeans Algorithm
% REPlic = 10; % Replication for KMeans Algorithm
% opts = statset('Display','iter');
% group1 = kmeans(Y1',6,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton','Options',opts);
% group2 = kmeans(Y2',6,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton','Options',opts);
% group3 = kmeans(Y3',6,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton','Options',opts);
% GrpsL1 = apclusterK(KYY1, 6, 0);
% GrpsL2 = apclusterK(KYY2, 6, 0);
% GrpsL3 = apclusterK(KYY3, 6, 0);
%%
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

exemplar_labels = unique(Grps6);
Grps6_mapped = zeros(size(Grps6));
for iter = 1:length(exemplar_labels),
	Grps6_mapped(Grps6 == exemplar_labels(iter)) = iter;
end;
Grps6 = Grps6_mapped;

exemplar_labels = unique(Grps7);
Grps7_mapped = zeros(size(Grps7));
for iter = 1:length(exemplar_labels),
	Grps7_mapped(Grps7 == exemplar_labels(iter)) = iter;
end;
Grps7 = Grps7_mapped;

%%
exemplar_labels = unique(GrpsL1);
GrpsL1_mapped = zeros(size(GrpsL1));
for iter = 1:length(exemplar_labels),
	GrpsL1_mapped(GrpsL1 == exemplar_labels(iter)) = iter;
end;
GrpsL1 = GrpsL1_mapped;

exemplar_labels = unique(GrpsL2);
GrpsL2_mapped = zeros(size(GrpsL2));
for iter = 1:length(exemplar_labels),
	GrpsL2_mapped(GrpsL2 == exemplar_labels(iter)) = iter;
end;
GrpsL2 = GrpsL2_mapped;

exemplar_labels = unique(GrpsL3);
GrpsL3_mapped = zeros(size(GrpsL3));
for iter = 1:length(exemplar_labels),
	GrpsL3_mapped(GrpsL3 == exemplar_labels(iter)) = iter;
end;
GrpsL3 = GrpsL3_mapped;

%%
kernelNMI = zeros(6, 3);
kernelNMI(1, 1) = nmi(gnd, Grps(:,1)); kernelNMI(1, 2) = nmi(gnd, Grps(:,2)); kernelNMI(1, 3) = nmi(gnd, Grps(:,3));
kernelNMI(2, 1) = nmi(gnd, Grps1(:,1)); kernelNMI(2, 2) = nmi(gnd, Grps1(:,2)); kernelNMI(2, 3) = nmi(gnd, Grps1(:,3));
kernelNMI(3, 1) = nmi(gnd, Grps2(:,1)); kernelNMI(3, 2) = nmi(gnd, Grps2(:,2)); kernelNMI(3, 3) = nmi(gnd, Grps2(:,3));
kernelNMI(4, 3) = nmi(gnd, Grps3);
kernelNMI(5, 3) = nmi(gnd, Grps4);
kernelNMI(6, 3) = nmi(gnd, Grps5);
kernelNMI(7, 3) = nmi(gnd, Grps6);
kernelNMI(8, 3) = nmi(gnd, Grps7);

kernelRand = zeros(6, 3);
kernelRand(1, 1) = RandIndex(gnd, Grps(:,1)); kernelRand(1, 2) = RandIndex(gnd, Grps(:,2)); kernelRand(1, 3) = RandIndex(gnd, Grps(:,3));
kernelRand(2, 1) = RandIndex(gnd, Grps1(:,1)); kernelRand(2, 2) = RandIndex(gnd, Grps1(:,2)); kernelRand(2, 3) = RandIndex(gnd, Grps1(:,3));
kernelRand(3, 1) = RandIndex(gnd, Grps2(:,1)); kernelRand(3, 2) = RandIndex(gnd, Grps2(:,2)); kernelRand(3, 3) = RandIndex(gnd, Grps2(:,3));
kernelRand(4, 3) = RandIndex(gnd, Grps3);
kernelRand(5, 3) = RandIndex(gnd, Grps4);
kernelRand(6, 3) = RandIndex(gnd, Grps5);
kernelRand(7, 3) = RandIndex(gnd, Grps6);
kernelRand(8, 3) = RandIndex(gnd, Grps7);

kernelACC = zeros(6, 3);
kernelACC(1, 1) = clustering_error(gnd, Grps(:,1)); kernelACC(1, 2) = clustering_error(gnd, Grps(:,2));...
	kernelACC(1, 3) = clustering_error(gnd, Grps(:,3));
kernelACC(2, 1) = clustering_error(gnd, Grps1(:,1)); kernelACC(2, 2) = clustering_error(gnd, Grps1(:,2));...
	kernelACC(2, 3) = clustering_error(gnd, Grps1(:,3));
kernelACC(3, 1) = clustering_error(gnd, Grps2(:,1)); kernelACC(3, 2) = clustering_error(gnd, Grps2(:,2));...
	kernelACC(3, 3) = clustering_error(gnd, Grps2(:,3));
kernelACC(4, 3) = clustering_error(gnd, Grps3);
kernelACC(5, 3) = clustering_error(gnd, Grps4);
kernelACC(6, 3) = clustering_error(gnd, Grps5);
kernelACC(7, 3) = clustering_error(gnd, Grps6);
kernelACC(8, 3) = clustering_error(gnd, Grps7);
kernelACC = (100 - kernelACC) / 100;

kernelRate = zeros(6, 3);
kernelRate(1, 1) = clusterEvaluate(gnd, Grps(:,1)); kernelRate(1, 2) = clusterEvaluate(gnd, Grps(:,2));...
	kernelRate(1, 3) = clusterEvaluate(gnd, Grps(:,3));
kernelRate(2, 1) = clusterEvaluate(gnd, Grps1(:,1)); kernelRate(2, 2) = clusterEvaluate(gnd, Grps1(:,2));...
	kernelRate(2, 3) = clusterEvaluate(gnd, Grps1(:,3));
kernelRate(3, 1) = clusterEvaluate(gnd, Grps2(:,1)); kernelRate(3, 2) = clusterEvaluate(gnd, Grps2(:,2));...
	kernelRate(3, 3) = clusterEvaluate(gnd, Grps2(:,3));
kernelRate(4, 3) = clusterEvaluate(gnd, Grps3);
kernelRate(5, 3) = clusterEvaluate(gnd, Grps4);
kernelRate(6, 3) = clusterEvaluate(gnd, Grps5);
kernelRate(7, 3) = clusterEvaluate(gnd, Grps6);
kernelRate(8, 3) = clusterEvaluate(gnd, Grps7);
kernelRate = kernelRate / 100;

linearNMI(1) = nmi(gnd, GrpsL1);
linearNMI(2) = nmi(gnd, GrpsL2);
linearNMI(3) = nmi(gnd, GrpsL3);

linearRand(1) = RandIndex(gnd, GrpsL1);
linearRand(2) = RandIndex(gnd, GrpsL2);
linearRand(3) = RandIndex(gnd, GrpsL3);

linearACC(1) = clustering_error(gnd, GrpsL1);
linearACC(2) = clustering_error(gnd, GrpsL2);
linearACC(3) = clustering_error(gnd, GrpsL3);
linearACC = (100 - linearACC) / 100;

linearRate(1) = clusterEvaluate(gnd, GrpsL1);
linearRate(2) = clusterEvaluate(gnd, GrpsL2);
linearRate(3) = clusterEvaluate(gnd, GrpsL3);
linearRate = linearRate / 100;
% v1 = nmi(gnd, Grps(:,1)); v2 = nmi(gnd, Grps(:,2)); v3 = nmi(gnd, Grps(:,3));
% v11 = nmi(gnd, Grps1(:,1)); v12 = nmi(gnd, Grps1(:,2)); v13 = nmi(gnd, Grps1(:,3));
% v21 = nmi(gnd, Grps2(:,1)); v22 = nmi(gnd, Grps2(:,2)); v23 = nmi(gnd, Grps2(:,3));
% kernelNMI = [v1 v2 v3; v11 v12 v13; v21 v22 v23]
% 
% a1 = RandIndex(gnd, Grps(:,1)); a2 = RandIndex(gnd, Grps(:,2)); a3 = RandIndex(gnd, Grps(:,3));
% a11 = RandIndex(gnd, Grps1(:,1)); a12 = RandIndex(gnd, Grps1(:,2)); a13 = RandIndex(gnd, Grps1(:,3));
% a21 = RandIndex(gnd, Grps2(:,1)); a22 = RandIndex(gnd, Grps2(:,2)); a23 = RandIndex(gnd, Grps2(:,3));
% kernelRand = [a1 a2 a3; a11 a12 a13; a21 a22 a23]
% 
% v11 = nmi(gnd, GrpsL1(:,1)); v12 = nmi(gnd, GrpsL1(:,2)); v13 = nmi(gnd, GrpsL1(:,3));
% v21 = nmi(gnd, GrpsL2(:,1)); v22 = nmi(gnd, GrpsL2(:,2)); v23 = nmi(gnd, GrpsL2(:,3));
% v31 = nmi(gnd, GrpsL3(:,1)); v32 = nmi(gnd, GrpsL3(:,2)); v33 = nmi(gnd, GrpsL3(:,3));
% linearNMI = [v11 v12 v13; v21 v22 v23; v31 v32 v33]
% 
% a11 = RandIndex(gnd, GrpsL1(:,1)); a12 = RandIndex(gnd, GrpsL1(:,2)); a13 = RandIndex(gnd, GrpsL1(:,3));
% a21 = RandIndex(gnd, GrpsL2(:,1)); a22 = RandIndex(gnd, GrpsL2(:,2)); a23 = RandIndex(gnd, GrpsL2(:,3));
% a31 = RandIndex(gnd, GrpsL3(:,1)); a32 = RandIndex(gnd, GrpsL3(:,2)); a33 = RandIndex(gnd, GrpsL3(:,3));
% linearRand = [a11 a12 a13; a21 a22 a23; a31 a32 a33]
% 
% v1 = nmi(gnd, group1); v2 = nmi(gnd, group2); v3 = nmi(gnd, group3);
% linearKmeansNMI = [v1 v2 v3]
% a1 = RandIndex(gnd, group1); a2 = RandIndex(gnd, group2); a3 = RandIndex(gnd, group3);
% linearKmeansRand = [a1 a2 a3]
