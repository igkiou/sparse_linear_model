function data_matrix = preprocess_data(raw_data_matrix, params)
%% parse inputs

if (nargin <= 3) || (~isstruct(params)),
	params = sparseClassificationParams;
end;

verboseMode = params.allVerboseMode;

whitening = params.dataWhitening;
useGradient = params.dataUseGradient;
resizeRatio = params.dataResizeRatio;
baseSize = params.dataBaseSize;

winSize = ceil(baseSize * resizeRatio);

data_matrix = raw_data_matrix;
dataSamples = size(data_matrix, 2);

%% perform gradient calculation and rescaling

if (resizeRatio ~= 1),
	
	if(verboseMode == 1),
		fprintf('Started rescaling.\n');
	end;

	temp_matrix = zeros(winSize ^ 2, dataSamples);
	for iter = 1:dataSamples,
		I = reshape(data_matrix(:, iter), [baseSize baseSize]);
		I = imresize(I, resizeRatio);
		temp_matrix(:, iter) = I(:);
	end;
	data_matrix = temp_matrix;
	
	if(verboseMode == 1),
		fprintf('Finished rescaling.\n');
	end;
end;

%% perform whitening

if(verboseMode == 1),
	fprintf('Started performing whitening.\n');
end;

if (whitening == 1)	
	data_matrix = data_matrix - repmat(mean(data_matrix), [size(data_matrix, 1) 1]);
	data_matrix = safeDivide(data_matrix, repmat(sqrt(sum(data_matrix .^ 2)), [size(data_matrix, 1) 1]));
elseif (whitening == 2)	
	data_matrix = safeDivide(data_matrix, repmat(sum(data_matrix), [size(data_matrix, 1) 1]));
elseif (whitening == 3)	
	data_matrix = safeDivide(data_matrix, repmat(sqrt(sum(data_matrix .^ 2)), [size(data_matrix, 1) 1]));
end;

if(verboseMode == 1),
	fprintf('Finished performing whitening.\n');
end;
