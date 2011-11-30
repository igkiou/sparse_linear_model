function M = adaptive_kernel_learning(Minit, R, T, eta, numEpochs, numPairsPerQuery, Kgnd)

Dgnd = kernel_distance(Kgnd);
% Things to do:
% Write pair management;
% Write pair data structure.
% Write initializations for parameters.
% Write random querying, and local optimization calls.

numPoints = size(Minit, 2);
numPairs = nchoosek(numPoints - 1, 2);
pairMatrix = zeros(numPoints, numPairs);
pairCounter = 0;
dataMatrix = zeros(R * numPoints, 3);
dataMatrixTemp = zeros(R, 3);

for iter = 1:numPoints,
	a = iter;
	[b c] = random_query(numPoints, R, a);
	dab = Dgnd(a, b);
	dac = Dgnd(a, c);
	dataMatrixTemp(:, 1) = a;
	dataMatrixTemp(dab <= dac, 2) = b(dab <= dac);
	dataMatrixTemp(dab > dac, 2) = c(dab > dac);
	dataMatrixTemp(dab <= dac, 3) = c(dab <= dac);
	dataMatrixTemp(dab > dac, 3) = b(dab > dac);
	dataMatrix((iter - 1) * R + (1:R), :) = dataMatrixTemp;
end;
pairMatrix = pairMatrix | dataToPairMatrix(dataMatrix);

dataMatrix = dataMatrix(randperm(numPoints * R), :);
M = adkl_local_optimize(Minit, dataMatrix, mu, eta * ones(numPoints * R, 1), numEpochs);

pairsStatic = nchoosek(numPoints - 1, 2);
dataMatrixTemp = zeros(numPairsPerQuery, 3);
dataMatrix = zeros(numPairsPerQuery * numPoints, 3);

for iterT = 1:T,
	inds = randperm(numPoints);
	for iterPoints = 1:numPoints,
		a = inds(iter);
		dataMatrix = pairToDataMatrix(pairMatrix, a);
		[b c] = adkl_query(M, numPairsPerQuery, a, dataMatrix, mu,...
			numSamples, pairsStatic);
		dab = Dgnd(a, b);
		dac = Dgnd(a, c);
		dataMatrixTemp(:, 1) = a;
		dataMatrixTemp(dab <= dac, 2) = b(dab <= dac);
		dataMatrixTemp(dab > dac, 2) = c(dab > dac);
		dataMatrixTemp(dab <= dac, 3) = c(dab <= dac);
		dataMatrixTemp(dab > dac, 3) = b(dab > dac);
		M = adkl_local_optimize(M, dataMatrixTemp, mu, eta * ones(numPairsPerQuery, 1), numEpochs);
		pairMatrix = pairMatrix | dataToPairMatrix(dataMatrixTemp);
	end;
	
% 	inds = randperm(numPoints);
% 	for iterPoints = 1:numPoints,
% 		a = inds(iter);
% 		dataMatrix = pairToDataMatrix(pairMatrix, a);
% 		[b c] = adkl_query(M, numPairsPerQuery, a, dataMatrix, mu,...
% 			numSamples, pairsStatic);
% 		dab = Dgnd(a, b);
% 		dac = Dgnd(a, c);
% 		dataMatrixTemp(:, 1) = a;
% 		dataMatrixTemp(dab <= dac, 2) = b(dab <= dac);
% 		dataMatrixTemp(dab > dac, 2) = c(dab > dac);
% 		dataMatrixTemp(dab <= dac, 3) = c(dab <= dac);
% 		dataMatrixTemp(dab > dac, 3) = b(dab > dac);
% 		dataMatrix((iter - 1) * numPairsPerQuery + (1:R), :) = dataMatrixTemp;
% 	end;	
% 	pairMatrix = pairMatrix | dataToPairMatrix(dataMatrix);
% 	M = adkl_local_optimize(M, dataMatrixTemp, mu, eta * ones(numPairsPerQuery * numPoints, 1), numEpochs);

% 	inds = randperm(numPoints);
% 	for iterPoints = 1:numPoints,
% 		a = inds(iter);
% 		dataMatrix = pairToDataMatrix(pairMatrix, a);
% 		[b c] = adkl_query(M, numPairsPerQuery, a, dataMatrix, mu,...
% 			numSamples, pairsStatic);
% 		dab = Dgnd(a, b);
% 		dac = Dgnd(a, c);
% 		dataMatrixTemp(:, 1) = a;
% 		dataMatrixTemp(dab <= dac, 2) = b(dab <= dac);
% 		dataMatrixTemp(dab > dac, 2) = c(dab > dac);
% 		dataMatrixTemp(dab <= dac, 3) = c(dab <= dac);
% 		dataMatrixTemp(dab > dac, 3) = b(dab > dac);
% 		dataMatrix((iter - 1) * numPairsPerQuery + (1:R), :) = dataMatrixTemp;
% 	end;	
% 	pairMatrix = pairMatrix | dataToPairMatrix(dataMatrix);
%	dataMatrix = pairToDataMatrix(pairMatrix, 1:numPoints);
%	dataMatrix = dataMatrix(randperm(size(dataMatrix, 1)), :);
% 	M = adkl_local_optimize(M, dataMatrixTemp, mu, eta * ones(numPairsPerQuery * numPoints, 1), numEpochs);

end;

end

function M = adkl_local_optimize(Minit, dataMatrix, mu, etaSchedule, numEpochs)

M = adkl_projection(Minit);
numTriplets = size(dataMatrix, 1);
for iterEpoch = 1:numEpochs,
	inds = randperm(numTriplets);
	for iter = 1:numTriplets,
		M = adkl_projection(M...
			- etaSchedule(iter) * adkl_gradient(M, dataMatrix(inds(iter), :), mu));
	end;
end;
	
end

function [obj deriv] = adkl_gradient(M, dataMatrix, mu)

deriv = zeros(size(M));
numTriplets = size(dataMatrix, 1);

obj = 0;
for iterTriplet = 1:numTriplets,
	M1 = M(:, dataMatrix(iterTriplet, 1));
	M2 = M(:, dataMatrix(iterTriplet, 2));
	M3 = M(:, dataMatrix(iterTriplet, 3));
	% Not efficient for large numTriplets, must calculate submatrix of
	% distances ahead of time. 
	dab = sum((M1 - M2) .^ 2);
	dac = sum((M1 - M3) .^ 2);
	p = (mu + dac) / (2 * mu + dab + dac);
	obj = obj + log(1 / p);
	
	dsq = (2 * mu + dab + dac) ^ 2;
	ddac = - 1 / p * (mu + dab) / dsq;
	ddab = 1 / p * (mu + dac) / dsq;
	
	deriv(:, dataMatrix(iterTriplet, 1)) = deriv(:, dataMatrix(iterTriplet, 1))...
		+ 2 * ddac * (M1 - M3) + 2 * ddab * (M1 - M2);
	deriv(:, dataMatrix(iterTriplet, 2)) = deriv(:, dataMatrix(iterTriplet, 2))...
		+ 2 * ddab * (M2 - M1);
	deriv(:, dataMatrix(iterTriplet, 3)) = deriv(:, dataMatrix(iterTriplet, 3))...
		+ 2 * ddac * (M3 - M1);
end;
obj = obj / numTriplets;
deriv = deriv / numTriplets;

end

function Mproj = adkl_projection(M)

Mproj = bsxfun(@rdivide, M, sqrt(sum(M .^ 2, 1)));

end

function [b c] = adkl_query(M, numPairs, a, dataMatrix, mu, numSamples, pairs)

numPoints = size(M, 2);
distMat = l2_distance(M);

% tau = ones(numPoints, 1);
% numTriplets = size(dataMatrix, 1);
% for iterTriplet = 1:numTriplets,
% 	dab = distMat(:, dataMatrix(iterTriplet, 2));
% 	dac = distMat(:, dataMatrix(iterTriplet, 3));
% 	tau = tau .* (mu + dac) ./ (2 * mu + dab + dac);
% end;

dab = distMat([1:(a - 1), (a + 1):numPoints], dataMatrix(:, 2)');
dac = distMat([1:(a - 1), (a + 1):numPoints], dataMatrix(:, 3)');
logtau = sum(log(mu + dac) - log(2 * mu + dab + dac), 2);
tau = exp(logtau);
tau = tau / sum(tau);
% logtau = log(tau);

if ((nargin < 7) || isempty(pairs)),
	pairs = nchoosek(numPoints - 1, 2);
end;
% pairs(pairs >= a) = pairs(pairs >= a) + 1;
randInds = randperm(size(pairs, 1));
sampleInds = randInds(1:numSamples);
samplePairs = pairs(sampleInds, :);
dab = distMat([1:(a - 1), (a + 1):numPoints], samplePairs(:, 1)');
dac = distMat([1:(a - 1), (a + 1):numPoints], samplePairs(:, 2)');
loglb = log(mu + dac) - log(2 * mu + dab + dac);
lb = exp(loglb);
taub = bsxfun(@times, lb, tau);
% logtaub = bsxfun(@plus, loglb, logtau);
% taub = exp(logtaub);
p = sum(taub, 1);
taub = bsxfun(@rdivide, taub, p);
tauc = bsxfun(@rdivide, bsxfun(@times, - expm1(loglb), tau), 1 - p);

infoGain = p .* sum(plogp(taub), 1) + (1 - p) .* sum(plogp(tauc), 1);
[foo ordInds] = sort(infoGain, 2, 'descend');
b = samplePairs(ordInds(1:numPairs), 1);
b(b >= a) = b(b >= a) + 1;
c = samplePairs(ordInds(1:numPairs), 2);
c(c >= a) = c(c >= a) + 1;

end
