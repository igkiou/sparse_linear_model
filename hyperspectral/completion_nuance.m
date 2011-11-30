SOURCE = '~/MATLAB/datasets_all/hyperspectral/subset/';
fileNames = {...
	'imgb0',...
	'imgb5',...
	'imgd3',...
	'imgd4',...
	'imgd7',...
	'imgf2',...
	'imgf5',...
	'imgf7',...
	};

largeM = 1040;
largeN = 1392;
samplingFactor = 3;
tau = 20;
wavelengths = 420:10:720;
Y = getKernelFactor(wavelengths, tau);
W = eye(31);
percentage = 0.2;
M = largeM / 2 ^ samplingFactor;
N = largeN / 2 ^ samplingFactor;
O = 31;
numEl = M * N * O;

numFiles = length(fileNames);
for iterFiles = 1:numFiles,
	fprintf('Now running file %s, number %d out of %d.\n', fileNames{iterFiles}, iterFiles, numFiles);
	load(sprintf('%s/%s.mat', SOURCE, fileNames{iterFiles}), 'ref');
	cube = ref;
	cube = cube / maxv(cube);
	cubedown = downSampleCube(cube, samplingFactor);
	numObs = ceil(percentage * numEl);
	inds = randperm(numEl);
	inds = inds(1:numObs);
	inds = sort(inds);
	[a b] = ind2sub([M * N O], inds);
	observationMat(:, 1) = a;
	observationMat(:, 2) = b;
	observationMat(:, 3) = cubedown(inds);

	B = matrix_completion_apg_mex(M * N, O, observationMat, [], [], [], [], 50000);
	cube_recovered = reshape(B, [M N O]);
	save(sprintf('matrix_per%g_def_%s_sub%d.mat', percentage,...
		fileNames{iterFiles}, samplingFactor),...
		'cube_recovered', 'observationMat');
	
	B = operator_completion_apg_mex(M * N, O, observationMat, Y, [], [], [], [], 50000);
	cube_recovered = reshape(B, [M N O]);
	save(sprintf('operator_per%g_tau%g_def_%s_sub%d.mat', percentage, tau,...
		fileNames{iterFiles}, samplingFactor),...
		'cube_recovered', 'observationMat');
end;
