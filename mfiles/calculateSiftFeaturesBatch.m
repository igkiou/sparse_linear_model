function [SIFTVectors SIFTVectorsUnnorm] = calculateSiftFeaturesBatch(dataVectors, nrml_threshold)

c_num = size(dataVectors, 2);
SIFTVectors = zeros(c_num, 128);
SIFTVectorsUnnorm = zeros(c_num, 128);
if (max(dataVectors(:)) > 1),
	dataVectors = dataVectors / 255;
end;

for jj = 1:c_num,
	disp(sprintf('Now processing figure no. %d out of %d', jj, c_num));

	I = dataVectors(:, jj);
	I = reshape(I, [32 32])';
	gridX = 1;
	gridY = 1;
	
	% find SIFT descriptors
	SIFTVectorsUnnorm(jj, :) = getSiftGrid(I, [], gridX, gridY, 32, 0.8);
	SIFTVectors(jj, :) = normalizeSiftFeatures(SIFTVectorsUnnorm(jj, :), nrml_threshold);
end;    
