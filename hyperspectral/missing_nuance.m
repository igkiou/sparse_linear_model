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
vec = 1:31;
vec(4:8:31) = [];
wavelengths = 420:10:720;
wavelengths = wavelengths(vec);
Y = getKernelFactor(wavelengths, tau);
W = eye(31);
percentage = 0.4;
M = largeM / 2 ^ samplingFactor;
N = largeN / 2 ^ samplingFactor;
O = length(vec);
numEl = M * N * O;

numFiles = length(fileNames);
for iterFiles = 1:numFiles,
	fprintf('Now running file %s, number %d out of %d.\n', fileNames{iterFiles}, iterFiles, numFiles);
	load(sprintf('%s/%s.mat', SOURCE, fileNames{iterFiles}), 'ref');
	cube = ref;
	cube = cube / maxv(cube);
	cube = cube(:, :, vec);
	cubedown = downSampleCube(cube, samplingFactor);
	numObs = ceil(percentage * numEl);
	inds = randperm(numEl);
	inds = inds(1:numObs);
	inds = sort(inds);
	[a b] = ind2sub([M * N O], inds);
	observationMat(:, 1) = a;
	observationMat(:, 2) = b;
	observationMat(:, 3) = cubedown(inds);

	B = operator_completion_apg_mex(M * N, O, observationMat, Y, [], [], [], [], 50000);
	cube_recovered = reshape(B, [M N O]);
	save(sprintf('operator_per%g_tau%g_def_%s_sub%d_sp%d.mat', percentage, tau,...
		fileNames{iterFiles}, samplingFactor, length(vec)),...
		'cube_recovered', 'observationMat', 'wavelengths');
end;
