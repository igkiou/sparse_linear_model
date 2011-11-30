SOURCE = '~/MATLAB/datasets_all/hyperspectral/moving/tif_files/';
fileNames = {...
	'gsd1',...
	'lawgate3',...
	'yard3',...
	'lawlibrary7'...
};

largeM = 1040;
largeN = 1392;
samplingFactor = 2;
kappa = 0.1;
tau = 20;
wavelength = 420:10:720;
kernelMat = kernel_gram(wavelength, [], 'h', tau);
[U S] = eig(kernelMat);
Y = U * sqrt(S);
W = eye(31);

M = largeM / 2 ^ samplingFactor;
N = largeN / 2 ^ samplingFactor;

numFiles = length(fileNames);
for iterFiles = 1:numFiles,
	fprintf('Now running file %s, number %d out of %d.\n', fileNames{iterFiles}, iterFiles, numFiles);
	cube = getCube(sprintf('%s/%s', SOURCE, fileNames{iterFiles}), [], 420:10:720, 1);
	cube = cube / maxv(cube);
	cubedown = downSampleCube(cube, samplingFactor);

	[B A] = robust_pca_apg_mex(reshape(cubedown, [M * N, 31]), [], [], [], [], [],	100000);
	cube_background = reshape(B, [M N 31]);
	cube_foreground = reshape(A, [M N 31]);
	save(sprintf('robust_pca_def_%s_sub%d.mat',...
		fileNames{iterFiles}, samplingFactor),...
		'cube_background', 'cube_foreground');
	
	[B A] = robust_operator_pca_apg_mex(reshape(cubedown, [M * N, 31]), Y, [], [], [], [], [],	100000);
	cube_background = reshape(B, [M N 31]);
	cube_foreground = reshape(A, [M N 31]);
	save(sprintf('robust_oppca_tau%g_def_%s_sub%d.mat', tau,...
		fileNames{iterFiles}, samplingFactor),...
		'cube_background', 'cube_foreground');
end;
