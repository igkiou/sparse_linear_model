function [labels, singularValues, singularVectors] = spectralClustering(kernelGram, numClusters, varargin)

numDataPoints = size(kernelGram, 1);

%%
fprintf('Method 1: Unnormalized Method\n');
D = diag(sum(kernelGram));
L = D - kernelGram;

fprintf('Method 1: Started eigendecomposition\n');
% opts.disp = 0;
% [kerKU sKU] = eigs(LapKU, numClusters, 'sa', opts);
[kerKU, sKU] = laneig_modified_nocomplex(L, numClusters, 'AS');
singularValuesKU = diag(sKU);

fprintf('Method 1: Started kmeans\n');
labels1 = kmeans(kerKU, numClusters, varargin{:});

%%
fprintf('Method 2: Random Walk Method\n');
D = diag(sum(kernelGram) .^ (-1));
L = eye(numDataPoints) - D * kernelGram;

fprintf('Method 2: Started eigendecomposition\n');
% [uKN, sKN, vKN] = svds(LapKN, numClusters, 0);
[~, sKN, kerKN] = lansvd_modified_nocomplex(L, numClusters, 'S');
singularValuesKN = diag(sKN);

fprintf('Method 2: Started kmeans\n');
labels2 = kmeans(kerKN, numClusters, varargin{:});

%%
fprintf('Method 3: Normalized Symmetric\n');
D = diag(sum(kernelGram) .^ (-1/2));
L = speye(numDataPoints) - D * kernelGram * D;
L = (L + L') / 2;

fprintf('Method 3: Started eigendecomposition\n');
% opts.disp = 0;
% [kerKS, sKS] = eigs(LapKS, numClusters, 'sa', opts);
[kerKS, sKS] = laneig_modified_nocomplex(L, numClusters, 'AS');
for i = 1:numDataPoints
    kerKS(i,:) = kerKS(i,:) ./ norm(kerKS(i,:));
end
singularValuesKS = diag(sKS);

fprintf('Method 3: Started kmeans\n');
labels3 = kmeans(kerKS, numClusters, varargin{:});

%%
labels = [labels1, labels2, labels3];
singularValues = [singularValuesKU,singularValuesKN,singularValuesKS];
singularVectors(:,:,1) = kerKU;
singularVectors(:,:,2) = kerKN;
singularVectors(:,:,3) = kerKS;
