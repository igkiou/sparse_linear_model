function Do = dictionaryOuterProduct(Ds, Df)

[sDimSq, sNum] = size(Ds);
[fDim, fNum] = size(Df);
sDim = sqrt(sDimSq);
Do = zeros(sDimSq * fDim, sNum * fNum);
for iterS = 1:sNum,
	for iterF = 1:fNum,
		Do(:, (iterS - 1) * fNum + iterF) = vec(Ds(:, iterS) * Df(:, iterF)');
	end;
end
		
