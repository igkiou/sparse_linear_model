function tauList = getContinuationList(targetValue, maxValue, numValues, rate)

if ((nargin >= 4) && ~isempty(rate)),
	numValues = floor(log(maxValue / targetValue) / log(1 / rate));
else
	numValues = numValues - 1;
	rate = (maxValue / targetValue) ^ (- 1 / numValues);
end;
tauList = zeros(numValues + 1, 1);
tauList(end) = targetValue;
for iter = 1:numValues,
	tauList(end - iter) = targetValue * rate ^ (- iter);
end;
