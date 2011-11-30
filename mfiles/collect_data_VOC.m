function collect_data_VOC(filename, data_path, params)
%% read input arguments

if (nargin <= 2) || (~isstruct(params)),
	params = sparseClassificationParams;
end;

winSize = params.dataWinSize;
results_path = params.dataResultsPath;

if (nargin < 1) || (any(isnan(filename))),
	filename = sprintf('VOC_%d', winSize ^ 2);
end;
if (nargin <= 1) || (any(isnan(data_path))),
	data_path = sprintf('/home/igkiou/MATLAB/VOC2007');
end;

%% load file names
contents = ls(sprintf('%s/VOCtest/JPEGImages/', data_path));
tempNames = regexp(contents, '\s*', 'split');
names = tempNames(1:(end - 1));
dataNum1 = length(tempNames) - 1;
contents = ls(sprintf('%s/VOCtrainval/JPEGImages/', data_path));
tempNames = regexp(contents, '\s*', 'split');
names = [names tempNames(1:(end - 1))];
dataNum2 = length(tempNames) - 1;
dataNum = dataNum1 + dataNum2;

% patches per image to keep
patchesPerImg = 500;

%% load no white data

% initialize the matrix to hold the patches
data_nowhite = zeros(winSize ^ 2, patchesPerImg * dataNum);
totalPatches = 0;

% start collecting data
for iter=(1:dataNum)
	fprintf('Image %d.\n', iter);
	
	% load the image
	if (iter <= dataNum1)
		I = imread(sprintf('%s/VOCtest/JPEGImages/%s', data_path, names{iter}));
	else
		I = imread(sprintf('%s/VOCtrainval/JPEGImages/%s', data_path, names{iter}));
	end;
	
	% turn to gray and double
	I = rgb2gray(I);
	I = im2double(I);
	
	% collect patches and whiten
	X_nowhite = im2col(I, [winSize winSize], 'distinct');
	
	% select a random permutation of patches
	extractPatches = size(X_nowhite, 2);
	seq = randperm(extractPatches);
	takePatches = seq(1:min(extractPatches, patchesPerImg));
	
	% update data matrices
	data_nowhite(:, (totalPatches + 1):(totalPatches + length(takePatches))) = X_nowhite(:, takePatches);
	
	% update number of patches so far
	totalPatches = totalPatches + length(takePatches);
end;

%% save no white data

% remove any zero entries
% data_nowhite = data_nowhite(:, 1:totalPatches);
% data_white = data_white(:, 1:totalPatches);

% randomly permute data
seq = randperm(totalPatches);
data_nowhite_short = data_nowhite(:, seq(1:min(50000,length(seq))));

eval(sprintf('save %s/%s data_nowhite data_nowhite_short -V7.3', results_path, filename));
clear data_nowhite
clear data_nowhite_short

%% load white data 

% initialize the matrix to hold the patches
data_white = zeros(winSize ^ 2, patchesPerImg * dataNum);
totalPatches = 0;

% start collecting data
for iter=(1:dataNum)
	fprintf('Image %d.\n', iter);
	
	% load the image
	if (iter <= dataNum1)
		I = imread(sprintf('%s/VOCtest/JPEGImages/%s', data_path, names{iter}));
	else
		I = imread(sprintf('%s/VOCtrainval/JPEGImages/%s', data_path, names{iter}));
	end;
	
	% turn to gray and double
	I = rgb2gray(I);
	I = im2double(I);
	
	% collect patches and whiten
	X_nowhite = im2col(I, [winSize winSize], 'distinct');
	X_white = X_nowhite - repmat(mean(X_nowhite), [size(X_nowhite, 1) 1]);
	X_white = safeDivide(X_white, repmat(sqrt(sum(X_white .^ 2)), [size(X_white, 1) 1]));
	
	% select a random permutation of patches
	extractPatches = size(X_nowhite, 2);
	seq = randperm(extractPatches);
	takePatches = seq(1:min(extractPatches, patchesPerImg));
	
	% update data matrices
	data_white(:, (totalPatches + 1):(totalPatches + length(takePatches))) = X_white(:, takePatches);
	
	% update number of patches so far
	totalPatches = totalPatches + length(takePatches);
end;

%% save white data

% remove any zero entries
% data_nowhite = data_nowhite(:, 1:totalPatches);
% data_white = data_white(:, 1:totalPatches);

% randomly permute data
seq = randperm(totalPatches);

data_white_short = data_white(:, seq(1:min(50000,length(seq))));
eval(sprintf('save %s/%s data_white data_white_short -APPEND -V7.3', results_path, filename));
clear data_white
clear data_white_short
