X1tX1 = gramTrainTrain;
clear gramTrainTrain
X1sq = diag(X1tX1);
distanceMatrix = bsxfun(@plus, X1sq, bsxfun(@minus, X1sq', 2 * X1tX1));
clear X1tX1 X1sq
distanceMatrix = max(distanceMatrix, distanceMatrix');
distanceMatrix = distanceMatrix - diag(diag(distanceMatrix));
[Srt,Ind] = sort(distanceMatrix, 1);
clear Srt
numNeighbors = 5;
WEu = zeros(size(distanceMatrix));
for iter = 1:size(distanceMatrix,1),
	WEu(Ind(1:(numNeighbors + 1),iter),iter) = 1;
end;
WEu = max(WEu, WEu');
clear Ind
