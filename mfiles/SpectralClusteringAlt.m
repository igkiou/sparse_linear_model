%--------------------------------------------------------------------------
% This function takes a NxN matrix CMat as adjacency of a graph and 
% computes the segmentation of data from spectral clustering.
% CMat: NxN adjacency matrix
% n: number of groups for segmentation
% K: number of largest coefficients to choose from each column of CMat
% Grps: [grp1,grp2,grp3] for three different forms of Spectral Clustering
% SingVals: [SV1,SV2,SV3] singular values for three different forms of SC
% LapKernel(:,:,i): n last columns of kernel of laplacian to apply KMeans
%--------------------------------------------------------------------------
% Copyright @ Ehsan Elhamifar, 2010
%--------------------------------------------------------------------------


function Grps = SpectralClustering(CKSym,n)

N = size(CKSym,1);
MAXiter = 1000; % Maximum iteration for KMeans Algorithm
REPlic = 20; % Replication for KMeans Algorithm

fprintf('Method 1: Unnormalized Method\n');
DKU = diag( sum(CKSym) );
LapKU = DKU - CKSym;
fprintf('Method 1: Started eigendecomposition\n');
% [uKU,sKU,vKU] = svd(LapKU);
% f = size(vKU,2);
% kerKU = vKU(:,f-n+1:f);
opts.disp = 0;
[kerKU sKU] = eigs(LapKU, n, 'sa', opts);
svalKU = diag(sKU);
fprintf('Method 1: Started kmeans\n');
opts = statset('Display','off');
group1 = kmeans(kerKU,n,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton','Options',opts);

% fprintf('Method 2: Random Walk Method\n');
% DKN = diag(sum(CKSym) .^ (-1));
% LapKN = speye(N) - DKN * CKSym;
% fprintf('Method 2: Started eigendecomposition\n');
% % [uKN,sKN,vKN] = svd(LapKN);
% % f = size(vKN,2);
% % kerKN = vKN(:,f-n+1:f);
% opts.disp = 0;
% [uKN,sKN,vKN] = svds(LapKN, n, 0);
% kerKN = vKN;
% % [kerKN sKN] = eigs(LapKN, n, 'sa', opts);
% svalKN = diag(sKN);
% fprintf('Method 2: Started kmeans\n');
% opts = statset('Display','off');
% group2 = kmeans(kerKN,n,'start','sample','maxiter',MAXiter,'replicates',REPlic,'EmptyAction','singleton','Options',opts);
group2 = group1;
group3 = group2;

%
Grps = [group1,group2,group3];
