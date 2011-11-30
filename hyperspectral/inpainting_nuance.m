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
samplingFactor = 2;
tau = 20;
wavelengths = 420:10:720;
Y = getKernelFactor(wavelengths, tau);
W = eye(31);
ps = 0.2;
M = largeM / 2 ^ samplingFactor;
N = largeN / 2 ^ samplingFactor;

numFiles = length(fileNames);
for iterFiles = 1:numFiles,
	fprintf('Now running file %s, number %d out of %d.\n', fileNames{iterFiles}, iterFiles, numFiles);
	load(sprintf('%s/%s.mat', SOURCE, fileNames{iterFiles}), 'ref');
	cube = ref;
	cube = cube / maxv(cube);
	cubedown = downSampleCube(cube, samplingFactor);
	noisevec = addBernoulliSignSupportNoise(M * N * 31, 1, ps);
	cubedownnoise = cubedown + reshape(noisevec, size(cubedown));
	
	[B A] = robust_pca_apg_mex(reshape(cubedownnoise, [M * N, 31]), [], [], [], [], [],	100000);
	cube_background = reshape(B, [M N 31]);
	cube_foreground = reshape(A, [M N 31]);
	save(sprintf('robust_pca_ps%g_def_%s_sub%d.mat', ps,...
		fileNames{iterFiles}, samplingFactor),...
		'cube_background', 'cube_foreground', 'noisevec');
	
	[B A] = robust_operator_pca_apg_mex(reshape(cubedownnoise, [M * N, 31]), Y, [], [], [], [], [],	100000);
	cube_background = reshape(B, [M N 31]);
	cube_foreground = reshape(A, [M N 31]);
	save(sprintf('robust_oppca_ps%g_tau%g_def_%s_sub%d.mat', ps, tau,...
		fileNames{iterFiles}, samplingFactor),...
		'cube_background', 'cube_foreground', 'noisevec');
end;

