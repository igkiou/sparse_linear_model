%%
numCities = 10;
numItersFPC = 5000;
numItersAPG = 10000;
numItersSAPG = 10000;
etaFPC = 0.1;
etaAPG = 10;
etaSAPG = 10;

%%
% the 10 cities dataset
cities = [0, 587, 1212, 701, 1936, 604, 748, 2139, 2182, 543;
	  			587, 0, 920, 940, 1745, 1188, 713, 1858, 1737, 597;
	  			1212, 920, 0, 879, 831, 1726, 1631,949, 1021, 1494;
	  			701, 940, 879, 0, 1374, 968, 1420, 1645, 1891, 1220;
	  			1936, 1745, 831, 1374, 0, 2339, 2451, 347, 959, 2300;
	  			604, 1188, 1726, 968, 2339, 0, 1092, 2594, 2734, 923;
	  			748, 713, 1631, 1420, 2451, 1092, 0, 2571, 2408, 205;
	  			2139, 1858, 949, 1645, 347, 2594, 2571,  0, 678, 2442;
	  			2182, 1737, 1021, 1891, 959, 2734, 2408, 678, 0, 2329;
	  			543, 597, 1494, 1220, 2300, 923, 205, 2442, 2329, 0];

dataMatrix = getGroundTruthTriples(numCities, cities, 1);

%%
tauValues = [100 50 10 5 1 0.5 0.1 0.05 0.01] / size(dataMatrix, 1);
numTauVals = length(tauValues);

%%
% TODO: Make mfile and mex versions be compatible again.

for iterTau = 1:numTauVals,
	fprintf('Now running iter %d out of %d.\n', iterTau, numTauVals);
	
	dataMatrix(:, 5) = 1;
	[X, spread, indo, G, slack] = yanmds(dataMatrix, nImages, tauValues(iterTau) * size(dataMatrix, 1)); % 
	dataMatrix(:, 5) = - 1;
	temp = getViolations(G, [], dataMatrix(:, 1:4), dataMatrix(:, 5));
	rank_ya(iterTau) = rank(G);
	viol_ya(iterTau) = sum(temp > 0);
	slack_ya(iterTau) = sum(temp);
	loss_ya(iterTau) = slack_ya(iterTau) / size(dataMatrix, 1) + tauValues(iterTau) * sum(svd(G));
	kernel_ya(:, iterTau) = vec(G);

	dataMatrix(:, 5) = - 1;
	tauList = getContinuationList(tauValues(iterTau), 100, 1);
% 	[L, converged, Z, viol_fpc] = nmmds_fpc(nImages, dataMatrix, tauList, [], 0.01, 100000);
	K = nmmds_fpc_mex(nImages, dataMatrix, [], tauList, [], etaFPC, numItersFPC);
	temp = getViolations(K, [], dataMatrix(:, 1:4), dataMatrix(:, 5));
	rank_fpc(iterTau) = rank(K);
	viol_fpc(iterTau) = sum(temp > 0);
	slack_fpc(iterTau) = sum(temp);
	loss_fpc(iterTau) = slack_fpc(iterTau) / size(dataMatrix, 1) + tauValues(iterTau) * sum(svd(K));
	kernel_fpc(:, iterTau) = vec(K);

	dataMatrix(:, 5) = - 1;
	tauList = getContinuationList(tauValues(iterTau), 100, 1);
% 	[L, converged, Z, viol_fpc] = nmmds_fpc(nImages, dataMatrix, tauList, [], 0.01, 100000);
	L = nmmds_apg_mex(nImages, dataMatrix, [], tauList, [], etaAPG, numItersAPG);
	temp = getViolations(L, [], dataMatrix(:, 1:4), dataMatrix(:, 5));
	rank_apg(iterTau) = rank(L);
	viol_apg(iterTau) = sum(temp > 0);
	slack_apg(iterTau) = sum(temp);
	loss_apg(iterTau) = slack_apg(iterTau) / size(dataMatrix, 1) + tauValues(iterTau) * sum(svd(L));
	kernel_apg(:, iterTau) = vec(L);

	dataMatrix(:, 5) = - 1;
	tauList = getContinuationList(tauValues(iterTau), 100, 1);
	F = nmmds_smooth_apg_mex(nImages, dataMatrix, [], tauList, [], etaSAPG, numItersSAPG);
	temp = getViolations(F, [], dataMatrix(:, 1:4), dataMatrix(:, 5));
	rank_smooth_apg(iterTau) = rank(F);
	viol_smooth_apg(iterTau) = sum(temp > 0);
	slack_smooth_apg(iterTau) = sum(temp);
	loss_smooth_apg(iterTau) = sum(temp .^ 2) / size(dataMatrix, 1) + tauValues(iterTau) * sum(svd(F));
	kernel_smooth_apg(:, iterTau) = vec(F);
end;
