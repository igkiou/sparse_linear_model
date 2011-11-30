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
wavelengths = 400:10:700;
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
	cube = getCube(sprintf('%s/%s/%s', SOURCE, fileNames{iterFiles}, fileNames{iterFiles}), 'png', 1:31, 0);
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
