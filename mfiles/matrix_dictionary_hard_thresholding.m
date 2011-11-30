function [Dr normValue] = matrix_dictionary_hard_thresholding(D, rank, M, N)

K = size(D, 2);

Dr = zeros(size(D));
MTemp = zeros(M, N);
MrTemp = zeros(M, N);
normValue = 0;
normTemp = 0;
for iterK = 1:K,
	MTemp = reshape(D(:, iterK), [M N]);
	[MrTemp normTemp] = nuclear_hard_thresholding(MTemp, rank);
	Dr(:, iterK) = MrTemp(:);
	normValue = normValue + normTemp;
end;
	
