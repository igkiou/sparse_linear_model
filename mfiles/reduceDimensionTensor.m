function [trainFeatReduced testFeatReduced] = reduceDimensionTensor(trainFeat, testFeat, U, V, posIdx)

[nSmp, nFea] = size(trainFeat);
trainFeat = reshape(trainFeat', 32, 32, nSmp);
nRow = size(U, 2);
nCol = size(V, 2);
trainFeatReduced = zeros(nRow, nCol, nSmp);
for i=1:nSmp
	trainFeatReduced(:, :, i) = U' * trainFeat(:, :, i) * V;
end
trainFeatReduced = reshape(trainFeatReduced, nRow * nCol, nSmp)';
trainFeatReduced = trainFeatReduced(:, posIdx);

[nSmp, nFea] = size(testFeat);
testFeat = reshape(testFeat', 32, 32, nSmp);
nRow = size(U, 2);
nCol = size(V, 2);
testFeatReduced = zeros(nRow, nCol, nSmp);
for i=1:nSmp
	testFeatReduced(:, :, i) = U' * testFeat(:, :, i) * V;
end
testFeatReduced = reshape(testFeatReduced, nRow * nCol, nSmp)';
testFeatReduced = testFeatReduced(:, posIdx);
