%%
load ~/MATLAB/datasets_all/MNIST/MNIST_28x28.mat
Xorig = fea_Test';
Xorig = Xorig(:, (gnd_Test == 0) | (gnd_Test == 1) | (gnd_Test == 2) | (gnd_Test == 3) | (gnd_Test == 4) | (gnd_Test == 5));
gnd = gnd_Test((gnd_Test == 0) | (gnd_Test == 1) | (gnd_Test == 2) | (gnd_Test == 3) | (gnd_Test == 4) | (gnd_Test == 5));
gnd = gnd + 1;
clear fea_Train fea_Test gnd_Train gnd_Test

%%
X = mapcols(Xorig, [-1 1]);
gamma = 0.00728932024638;
param1 = sqrt(0.5/gamma);
KXX = kernel_gram_mex(X, [], 'g', param1);

%%
kernelgradientparams.initdict = NaN;
kernelgradientparams.dictsize = 784;
kernelgradientparams.iternum = 300;
kernelgradientparams.iternum2 = 10;
kernelgradientparams.blockratio = 1;
kernelgradientparams.codinglambda = 0.1500;
kernelgradientparams.dictclear = 0;
kernelgradientparams.kerneltype = 'G';
kernelgradientparams.kernelparam1 = param1;
kernelgradientparams.kernelparam2 = 1;
kernelgradientparams.printinfo = 1;
kernelgradientparams.errorinfo = 0;
D = kerneldictgradient(X, kernelgradientparams);

% %%
% numSamples = size(KXX, 2);
% lambda = 0.001;
% base = 2250;
% batchSize = 750;
% A = zeros(numSamples - 1, numSamples);
% B = zeros(numSamples - 1, numSamples);
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
%  	B(:, iter) = KDD \ KDx;
% end;
% 
% %%
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
% 
%%
% load MNIST_sc_A_l001_05_All
% load MNIST_sc_A_l001_05_results kernelACC kernelNMI kernelRand kernelRate

%%
% Grps = SpectralClustering(KXX, 6);	% Grps in kernel_sc_precomp_experiment notation
% Grps1 = SpectralClustering(As, 6);	% Grps1 in kernel_sc_precomp_experiment notation
% Grps2 = SpectralClustering(Am, 6);	% Grps2 in kernel_sc_precomp_experiment notation

% Grps3 = apclusterK(KXX, 6, 0);	% Grps3 in kernel_sc_precomp_experiment notation
% Grps4 = apclusterK(As, 6, 0);		% Grps4 in kernel_sc_precomp_experiment notation
% Grps5 = apclusterK(Am, 6, 0);		% Grps5 in kernel_sc_precomp_experiment notation
% Grps4 = Grps3;
% Grps5 = Grps3;
% Grps6 = Grps;						% N/A in kernel_sc_precomp_experiment notation
% Grps7 = Grps3;					% N/A in kernel_sc_precomp_experiment notation
% Grps8 = apclusterK(KDist, 6, 0);		% Grps6 in kernel_sc_precomp_experiment notation
% Grps9 = apclusterK(KDistSqrt, 6, 0);	% Grps7 in kernel_sc_precomp_experiment notation
% Grps10 = kernel_kmeans(KXX, 6)';		% N/A in kernel_sc_precomp_experiment notation

%%
% exemplar_labels = unique(Grps3);
% Grps3_mapped = zeros(size(Grps3));
% for iter = 1:length(exemplar_labels),
% 	Grps3_mapped(Grps3 == exemplar_labels(iter)) = iter;
% end;
% Grps3 = Grps3_mapped;
% 
% exemplar_labels = unique(Grps4);
% Grps4_mapped = zeros(size(Grps4));
% for iter = 1:length(exemplar_labels),
% 	Grps4_mapped(Grps4 == exemplar_labels(iter)) = iter;
% end;
% Grps4 = Grps4_mapped;
% 
% exemplar_labels = unique(Grps5);
% Grps5_mapped = zeros(size(Grps5));
% for iter = 1:length(exemplar_labels),
% 	Grps5_mapped(Grps5 == exemplar_labels(iter)) = iter;
% end;
% Grps5 = Grps5_mapped;
% 
% exemplar_labels = unique(Grps7);
% Grps7_mapped = zeros(size(Grps7));
% for iter = 1:length(exemplar_labels),
% 	Grps7_mapped(Grps7 == exemplar_labels(iter)) = iter;
% end;
% Grps7 = Grps7_mapped;
% 
% exemplar_labels = unique(Grps8);
% Grps8_mapped = zeros(size(Grps8));
% for iter = 1:length(exemplar_labels),
% 	Grps8_mapped(Grps8 == exemplar_labels(iter)) = iter;
% end;
% Grps8 = Grps8_mapped;
% 
% exemplar_labels = unique(Grps9);
% Grps9_mapped = zeros(size(Grps9));
% for iter = 1:length(exemplar_labels),
% 	Grps9_mapped(Grps9 == exemplar_labels(iter)) = iter;
% end;
% Grps9 = Grps9_mapped;

%%
% kernelNMI = zeros(11, 3);
% kernelNMI(1, 1) = nmi(gnd, Grps(:,1)); kernelNMI(1, 2) = nmi(gnd, Grps(:,2)); kernelNMI(1, 3) = nmi(gnd, Grps(:,3));
% kernelNMI(2, 1) = nmi(gnd, Grps1(:,1)); kernelNMI(2, 2) = nmi(gnd, Grps1(:,2)); kernelNMI(2, 3) = nmi(gnd, Grps1(:,3));
% kernelNMI(3, 1) = nmi(gnd, Grps2(:,1)); kernelNMI(3, 2) = nmi(gnd, Grps2(:,2)); kernelNMI(3, 3) = nmi(gnd, Grps2(:,3));
% kernelNMI(4, 3) = nmi(gnd, Grps3);
% kernelNMI(5, 3) = nmi(gnd, Grps4);
% kernelNMI(6, 3) = nmi(gnd, Grps5);
% kernelNMI(7, 1) = nmi(gnd, Grps6(:,1)); kernelNMI(7, 2) = nmi(gnd, Grps6(:,2)); kernelNMI(7, 3) = nmi(gnd, Grps6(:,3));
% kernelNMI(8, 3) = nmi(gnd, Grps7);
% kernelNMI(9, 3) = nmi(gnd, Grps8);
% kernelNMI(10, 3) = nmi(gnd, Grps9);
% kernelNMI(11, 3) = nmi(gnd, Grps10);

%%
% kernelRand = zeros(11, 3);
% kernelRand(1, 1) = RandIndex(gnd, Grps(:,1)); kernelRand(1, 2) = RandIndex(gnd, Grps(:,2)); kernelRand(1, 3) = RandIndex(gnd, Grps(:,3));
% kernelRand(2, 1) = RandIndex(gnd, Grps1(:,1)); kernelRand(2, 2) = RandIndex(gnd, Grps1(:,2)); kernelRand(2, 3) = RandIndex(gnd, Grps1(:,3));
% kernelRand(3, 1) = RandIndex(gnd, Grps2(:,1)); kernelRand(3, 2) = RandIndex(gnd, Grps2(:,2)); kernelRand(3, 3) = RandIndex(gnd, Grps2(:,3));
% kernelRand(4, 3) = RandIndex(gnd, Grps3);
% kernelRand(5, 3) = RandIndex(gnd, Grps4);
% kernelRand(6, 3) = RandIndex(gnd, Grps5);
% kernelRand(7, 1) = RandIndex(gnd, Grps6(:,1)); kernelRand(7, 2) = RandIndex(gnd, Grps6(:,2)); kernelRand(7, 3) = RandIndex(gnd, Grps6(:,3));
% kernelRand(8, 3) = RandIndex(gnd, Grps7);
% kernelRand(9, 3) = RandIndex(gnd, Grps8);
% kernelRand(10, 3) = RandIndex(gnd, Grps9);
% kernelRand(11, 3) = RandIndex(gnd, Grps10);

%%
% kernelACC = zeros(11, 3);
% kernelACC(1, 1) = clustering_error(gnd, Grps(:,1)); kernelACC(1, 2) = clustering_error(gnd, Grps(:,2));...
% 	kernelACC(1, 3) = clustering_error(gnd, Grps(:,3));
% kernelACC(2, 1) = clustering_error(gnd, Grps1(:,1)); kernelACC(2, 2) = clustering_error(gnd, Grps1(:,2));...
% 	kernelACC(2, 3) = clustering_error(gnd, Grps1(:,3));
% kernelACC(3, 1) = clustering_error(gnd, Grps2(:,1)); kernelACC(3, 2) = clustering_error(gnd, Grps2(:,2));...
% 	kernelACC(3, 3) = clustering_error(gnd, Grps2(:,3));
% kernelACC(4, 3) = clustering_error(gnd, Grps3);
% kernelACC(5, 3) = clustering_error(gnd, Grps4);
% kernelACC(6, 3) = clustering_error(gnd, Grps5);
% kernelACC(7, 1) = clustering_error(gnd, Grps6(:,1)); kernelACC(7, 2) = clustering_error(gnd, Grps6(:,2)); kernelACC(7, 3) = clustering_error(gnd, Grps6(:,3));
% kernelACC(8, 3) = clustering_error(gnd, Grps7);
% kernelACC(9, 3) = clustering_error(gnd, Grps8);
% kernelACC(10, 3) = clustering_error(gnd, Grps9);
% kernelACC(11, 3) = clustering_error(gnd, Grps10);
% kernelACC = (100 - kernelACC) / 100;

%%
% kernelRate = zeros(11, 3);
% kernelRate(1, 1) = clusterEvaluate(gnd, Grps(:,1)); kernelRate(1, 2) = clusterEvaluate(gnd, Grps(:,2));...
% 	kernelRate(1, 3) = clusterEvaluate(gnd, Grps(:,3));
% kernelRate(2, 1) = clusterEvaluate(gnd, Grps1(:,1)); kernelRate(2, 2) = clusterEvaluate(gnd, Grps1(:,2));...
% 	kernelRate(2, 3) = clusterEvaluate(gnd, Grps1(:,3));
% kernelRate(3, 1) = clusterEvaluate(gnd, Grps2(:,1)); kernelRate(3, 2) = clusterEvaluate(gnd, Grps2(:,2));...
% 	kernelRate(3, 3) = clusterEvaluate(gnd, Grps2(:,3));
% kernelRate(4, 3) = clusterEvaluate(gnd, Grps3);
% kernelRate(5, 3) = clusterEvaluate(gnd, Grps4);
% kernelRate(6, 3) = clusterEvaluate(gnd, Grps5);
% kernelRate(7, 1) = clusterEvaluate(gnd, Grps6(:,1)); kernelRate(7, 2) = clusterEvaluate(gnd, Grps6(:,2)); kernelRate(7, 3) = clusterEvaluate(gnd, Grps6(:,3));
% kernelRate(8, 3) = clusterEvaluate(gnd, Grps7);
% kernelRate(9, 3) = clusterEvaluate(gnd, Grps8);
% kernelRate(10, 3) = clusterEvaluate(gnd, Grps9);
% kernelRate(11, 3) = clusterEvaluate(gnd, Grps10);
% kernelRate = kernelRate / 100;

%%
% save MNIST_sc_A_l001_05_All A At Am As AtA
save MNIST_sc_A_l001_05_results kernelACC kernelNMI kernelRand kernelRate Grps Grps1 Grps2 Grps3 Grps4 Grps5 Grps6 Grps7 Grps8 Grps9 Grps10

% %%
% XL = normcols(Xorig); 
% KXXL = XL' * XL;

% %%
% KDistL = kernel_distance(KXXL, 0);
% KDistL = - KDistL;
% KDistSqrtL = kernel_distance(KXXL, 1);
% KDistSqrtL = - KDistSqrtL;

% %%
% numSamples = size(XL, 2);
% params.lambda = 0.001;
% base = 0;
% batchSize = 6031;
% AL = zeros(numSamples - 1, numSamples);
% 
% for iter = (base + 1):(base + batchSize),
% 	fprintf('Now sparse coding sample %d.\n', iter);
% 	xL = XL(:, iter);
% 	inds = true(numSamples, 1);
% 	inds(iter) = false;
% 	DL = XL(:, inds);
% 	AL(:, iter) = mexLasso(xL, DL, params);
% end;

% %%
% numSamples = size(AL, 2);
% AtL = zeros(numSamples);
% for iter = 1:numSamples,
% 	fprintf('Now sparse coding sample %d.\n', iter);
% 	inds = true(numSamples, 1);
% 	inds(iter) = false;
% 	AtL(inds, iter) = AL(:, iter);
% end;
% AmL = abs(AtL) + abs(AtL');
% AsL = BuildAdjacency(AtL, 0);
% AtAL = AtL' * AtL;

% %%
% % load MNIST_lin_A_l001_05_All
% % load MNIST_lin_A_l001_05_results kernelACC kernelNMI kernelRand kernelRate

% %%
% GrpsL = SpectralClustering(KXXL, 6);	% GrpsL in kernel_sc_precomp_experiment notation
% GrpsL1 = SpectralClustering(AsL, 6);	% GrpsL1 in kernel_sc_precomp_experiment notation
% GrpsL2 = SpectralClustering(AmL, 6);	% GrpsL2 in kernel_sc_precomp_experiment notation
% 
% GrpsL3 = apclusterK(KXXL, 6, 0);	% GrpsL3 in kernel_sc_precomp_experiment notation
% GrpsL4 = apclusterK(AsL, 6, 0);		% GrpsL4 in kernel_sc_precomp_experiment notation
% GrpsL5 = apclusterK(AmL, 6, 0);		% GrpsL5 in kernel_sc_precomp_experiment notation
% GrpsL6 = GrpsL;						% N/A in kernel_sc_precomp_experiment notation
% GrpsL7 = GrpsL3;					% N/A in kernel_sc_precomp_experiment notation
% GrpsL8 = apclusterK(KDistL, 6, 0);		% GrpsL6 in kernel_sc_precomp_experiment notation
% GrpsL9 = apclusterK(KDistSqrtL, 6, 0);	% GrpsL7 in kernel_sc_precomp_experiment notation
% opts = statset('Display','iter');
% MAXiter = 1000; % Maximum iteration for KMeans Algorithm
% REPlic = 20; % Replication for KMeans Algorithm
% GrpsL10 = kmeans(XL', 6, 'start', 'sample', 'maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton','Options',opts);

% %%
% exemplar_labels = unique(GrpsL3);
% GrpsL3_mapped = zeros(size(GrpsL3));
% for iter = 1:length(exemplar_labels),
% 	GrpsL3_mapped(GrpsL3 == exemplar_labels(iter)) = iter;
% end;
% GrpsL3 = GrpsL3_mapped;
% 
% exemplar_labels = unique(GrpsL4);
% GrpsL4_mapped = zeros(size(GrpsL4));
% for iter = 1:length(exemplar_labels),
% 	GrpsL4_mapped(GrpsL4 == exemplar_labels(iter)) = iter;
% end;
% GrpsL4 = GrpsL4_mapped;
% 
% exemplar_labels = unique(GrpsL5);
% GrpsL5_mapped = zeros(size(GrpsL5));
% for iter = 1:length(exemplar_labels),
% 	GrpsL5_mapped(GrpsL5 == exemplar_labels(iter)) = iter;
% end;
% GrpsL5 = GrpsL5_mapped;
% 
% exemplar_labels = unique(GrpsL7);
% GrpsL7_mapped = zeros(size(GrpsL7));
% for iter = 1:length(exemplar_labels),
% 	GrpsL7_mapped(GrpsL7 == exemplar_labels(iter)) = iter;
% end;
% GrpsL7 = GrpsL7_mapped;
% 
% exemplar_labels = unique(GrpsL8);
% GrpsL8_mapped = zeros(size(GrpsL8));
% for iter = 1:length(exemplar_labels),
% 	GrpsL8_mapped(GrpsL8 == exemplar_labels(iter)) = iter;
% end;
% GrpsL8 = GrpsL8_mapped;
% 
% exemplar_labels = unique(GrpsL9);
% GrpsL9_mapped = zeros(size(GrpsL9));
% for iter = 1:length(exemplar_labels),
% 	GrpsL9_mapped(GrpsL9 == exemplar_labels(iter)) = iter;
% end;
% GrpsL9 = GrpsL9_mapped;

% %%
% linearNMI = zeros(11, 3);
% linearNMI(1, 1) = nmi(gnd, GrpsL(:,1)); linearNMI(1, 2) = nmi(gnd, GrpsL(:,2)); linearNMI(1, 3) = nmi(gnd, GrpsL(:,3));
% linearNMI(2, 1) = nmi(gnd, GrpsL1(:,1)); linearNMI(2, 2) = nmi(gnd, GrpsL1(:,2)); linearNMI(2, 3) = nmi(gnd, GrpsL1(:,3));
% linearNMI(3, 1) = nmi(gnd, GrpsL2(:,1)); linearNMI(3, 2) = nmi(gnd, GrpsL2(:,2)); linearNMI(3, 3) = nmi(gnd, GrpsL2(:,3));
% linearNMI(4, 3) = nmi(gnd, GrpsL3);
% linearNMI(5, 3) = nmi(gnd, GrpsL4);
% linearNMI(6, 3) = nmi(gnd, GrpsL5);
% linearNMI(7, 1) = nmi(gnd, GrpsL6(:,1)); linearNMI(7, 2) = nmi(gnd, GrpsL6(:,2)); linearNMI(7, 3) = nmi(gnd, GrpsL6(:,3));
% linearNMI(8, 3) = nmi(gnd, GrpsL7);
% linearNMI(9, 3) = nmi(gnd, GrpsL8);
% linearNMI(10, 3) = nmi(gnd, GrpsL9);
% linearNMI(11, 3) = nmi(gnd, GrpsL10);

% %%
% linearRand = zeros(11, 3);
% linearRand(1, 1) = RandIndex(gnd, GrpsL(:,1)); linearRand(1, 2) = RandIndex(gnd, GrpsL(:,2)); linearRand(1, 3) = RandIndex(gnd, GrpsL(:,3));
% linearRand(2, 1) = RandIndex(gnd, GrpsL1(:,1)); linearRand(2, 2) = RandIndex(gnd, GrpsL1(:,2)); linearRand(2, 3) = RandIndex(gnd, GrpsL1(:,3));
% linearRand(3, 1) = RandIndex(gnd, GrpsL2(:,1)); linearRand(3, 2) = RandIndex(gnd, GrpsL2(:,2)); linearRand(3, 3) = RandIndex(gnd, GrpsL2(:,3));
% linearRand(4, 3) = RandIndex(gnd, GrpsL3);
% linearRand(5, 3) = RandIndex(gnd, GrpsL4);
% linearRand(6, 3) = RandIndex(gnd, GrpsL5);
% linearRand(7, 1) = RandIndex(gnd, GrpsL6(:,1)); linearRand(7, 2) = RandIndex(gnd, GrpsL6(:,2)); linearRand(7, 3) = RandIndex(gnd, GrpsL6(:,3));
% linearRand(8, 3) = RandIndex(gnd, GrpsL7);
% linearRand(9, 3) = RandIndex(gnd, GrpsL8);
% linearRand(10, 3) = RandIndex(gnd, GrpsL9);
% linearRand(11, 3) = RandIndex(gnd, GrpsL10);

% %%
% linearACC = zeros(11, 3);
% linearACC(1, 1) = clustering_error(gnd, GrpsL(:,1)); linearACC(1, 2) = clustering_error(gnd, GrpsL(:,2));...
% 	linearACC(1, 3) = clustering_error(gnd, GrpsL(:,3));
% linearACC(2, 1) = clustering_error(gnd, GrpsL1(:,1)); linearACC(2, 2) = clustering_error(gnd, GrpsL1(:,2));...
% 	linearACC(2, 3) = clustering_error(gnd, GrpsL1(:,3));
% linearACC(3, 1) = clustering_error(gnd, GrpsL2(:,1)); linearACC(3, 2) = clustering_error(gnd, GrpsL2(:,2));...
% 	linearACC(3, 3) = clustering_error(gnd, GrpsL2(:,3));
% linearACC(4, 3) = clustering_error(gnd, GrpsL3);
% linearACC(5, 3) = clustering_error(gnd, GrpsL4);
% linearACC(6, 3) = clustering_error(gnd, GrpsL5);
% linearACC(7, 1) = clustering_error(gnd, GrpsL6(:,1)); linearACC(7, 2) = clustering_error(gnd, GrpsL6(:,2)); linearACC(7, 3) = clustering_error(gnd, GrpsL6(:,3));
% linearACC(8, 3) = clustering_error(gnd, GrpsL7);
% linearACC(9, 3) = clustering_error(gnd, GrpsL8);
% linearACC(10, 3) = clustering_error(gnd, GrpsL9);
% linearACC(11, 3) = clustering_error(gnd, GrpsL10);
% linearACC = (100 - linearACC) / 100;

% %%
% linearRate = zeros(11, 3);
% linearRate(1, 1) = clusterEvaluate(gnd, GrpsL(:,1)); linearRate(1, 2) = clusterEvaluate(gnd, GrpsL(:,2));...
% 	linearRate(1, 3) = clusterEvaluate(gnd, GrpsL(:,3));
% linearRate(2, 1) = clusterEvaluate(gnd, GrpsL1(:,1)); linearRate(2, 2) = clusterEvaluate(gnd, GrpsL1(:,2));...
% 	linearRate(2, 3) = clusterEvaluate(gnd, GrpsL1(:,3));
% linearRate(3, 1) = clusterEvaluate(gnd, GrpsL2(:,1)); linearRate(3, 2) = clusterEvaluate(gnd, GrpsL2(:,2));...
% 	linearRate(3, 3) = clusterEvaluate(gnd, GrpsL2(:,3));
% linearRate(4, 3) = clusterEvaluate(gnd, GrpsL3);
% linearRate(5, 3) = clusterEvaluate(gnd, GrpsL4);
% linearRate(6, 3) = clusterEvaluate(gnd, GrpsL5);
% linearRate(7, 1) = clusterEvaluate(gnd, GrpsL6(:,1)); linearRate(7, 2) = clusterEvaluate(gnd, GrpsL6(:,2)); linearRate(7, 3) = clusterEvaluate(gnd, GrpsL6(:,3));
% linearRate(8, 3) = clusterEvaluate(gnd, GrpsL7);
% linearRate(9, 3) = clusterEvaluate(gnd, GrpsL8);
% linearRate(10, 3) = clusterEvaluate(gnd, GrpsL9);
% linearRate(11, 3) = clusterEvaluate(gnd, GrpsL10);
% linearRate = linearRate / 100;

% %%
% save MNIST_lin_A_l001_05_All AL AtL AmL AsL AtAL
% save MNIST_lin_A_l001_05_results linearACC linearNMI linearRand linearRate GrpsL GrpsL1 GrpsL2 GrpsL3 GrpsL4 GrpsL5 GrpsL6 GrpsL7 GrpsL8 GrpsL9 GrpsL10
