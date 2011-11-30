SOURCE = '~/MATLAB/datasets_all/multispectral/';
fileNames = {...
	'balloons_ms',...
	'beads_ms'...
	'sponges_ms',...
	'oil_painting_ms',...
	'flowers_ms',...
	'cd_ms',...
	'fake_and_real_peppers_ms',...
	'photo_and_face_ms',...
	};

largeM = 512;
largeN = 512;
samplingFactor = 2;
tau = 20;
vec = 1:31;
vec(4:8:31) = [];
wavelengths = 400:10:700;
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
	cube = getCube(sprintf('%s/%s/%s', SOURCE, fileNames{iterFiles}, fileNames{iterFiles}), 'png', 1:31, 0);
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
