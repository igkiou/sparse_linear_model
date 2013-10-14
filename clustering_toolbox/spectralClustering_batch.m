function labels = spectralClustering_batch(kernelGram, numClusterSets, varargin)

numDataPoints = size(kernelGram, 1);
numClusterSettings = length(numClusterSets);
maxNumClusters = maxv(numClusterSets);

%%
fprintf('Method 1: Unnormalized Method\n');
D = diag(sum(kernelGram));
L = D - kernelGram;

fprintf('Method 1: Started eigendecomposition\n');
% opts.disp = 0;
% [kerKU sKU] = eigs(LapKU, numClusters, 'sa', opts);
[U, ~] = laneig_modified_nocomplex(L, maxNumClusters, 'AS');

fprintf('Method 1: Started kmeans\n');
labels1 = zeros(numDataPoints, 1, numClusterSettings);
for iter = 1:numClusterSettings,
	numClusters = numClusterSets(iter);
	labels1(:, 1, iter) = kmeans(U(:, 1:numClusters), numClusters, varargin{:});
end;

%%
fprintf('Method 2: Random Walk Method\n');
D = diag(sum(kernelGram) .^ (-1));
L = eye(numDataPoints) - D * kernelGram;

fprintf('Method 2: Started eigendecomposition\n');
% [uKN, sKN, vKN] = svds(LapKN, numClusters, 0);
[~, ~, U] = lansvd_modified_nocomplex(L, numClusters, 'S');

fprintf('Method 2: Started kmeans\n');
labels2 = zeros(numDataPoints, 1, numClusterSettings);
for iter = 1:numClusterSettings,
	numClusters = numClusterSets(iter);
	labels2(:, 1, iter) = kmeans(U(:, 1:numClusters), numClusters, varargin{:});
end;

%%
fprintf('Method 3: Normalized Symmetric\n');
D = diag(sum(kernelGram) .^ (-1/2));
L = speye(numDataPoints) - D * kernelGram * D;
L = (L + L') / 2;

fprintf('Method 3: Started eigendecomposition\n');
% opts.disp = 0;
% [kerKS, sKS] = eigs(LapKS, numClusters, 'sa', opts);
[U, ~] = laneig_modified_nocomplex(L, numClusters, 'AS');
for i = 1:numDataPoints
    U(i,:) = U(i,:) ./ norm(U(i,:));
end

fprintf('Method 3: Started kmeans\n');
labels3 = zeros(numDataPoints, 1, numClusterSettings);
for iter = 1:numClusterSettings,
	numClusters = numClusterSets(iter);
	labels3(:, 1, iter) = kmeans(U(:, 1:numClusters), numClusters, varargin{:});
end;

%%
labels = zeros(numDataPoints, 3, numClusterSettings);
for iter = 1:numClusterSettings,
	labels(:, :, iter) = [labels1(:, 1, iter), labels2(:, 1, iter), labels3(:, 1, iter)];
end;
