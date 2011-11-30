function [Dr normValue] = matrix_dictionary_proximal(D, tau, M, N)

K = size(D, 2);

Dr = zeros(size(D));
MTemp = zeros(M, N);
MrTemp = zeros(M, N);
normValue = 0;
normTemp = 0;
for iterK = 1:K,
	MTemp = reshape(D(:, iterK), [M N]);
	[MrTemp normTemp] = nuclear_proximal(MTemp, tau);
	Dr(:, iterK) = MrTemp(:);
	normValue = normValue + normTemp;
end;
