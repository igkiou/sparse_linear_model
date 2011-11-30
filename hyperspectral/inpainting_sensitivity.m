fileNames = {...
	'~/MATLAB/datasets_all/hyperspectral/moving/tif_files/gsd1',...
% 	'~/MATLAB/datasets_all/hyperspectral/moving/tif_files/lawgate3',...
% 	'~/MATLAB/datasets_all/hyperspectral/moving/tif_files/lawlibrary7',...
% 	'~/MATLAB/datasets_all/hyperspectral/moving/tif_files/yard3',...
% 	'~/MATLAB/datasets_all/hyperspectral/moving/tif_files/lawgate1',...
% 	'~/MATLAB/datasets_all/hyperspectral/moving/tif_files/lawlibrary5',...
% 	'~/MATLAB/datasets_all/hyperspectral/moving/tif_files/lawlibrary9',...
% 	'~/MATLAB/datasets_all/hyperspectral/moving/tif_files/lawlibrary10',...
% 	'~/MATLAB/datasets_all/hyperspectral/moving/tif_files/yard2',...
% 	'~/MATLAB/datasets_all/hyperspectral/moving/tif_files/yard5'...
	};

largeM = 1040;
largeN = 1392;
subsample = 3;
kappa = 0.1;
tau = 50;
wavelength = 420:10:720;
kernelMat = kernel_gram(wavelength, [], 'h', tau);
[U S] = eig(kernelMat);
Y = U * sqrt(S);
W = eye(31);

M = largeM / 2 ^ subsample;
N = largeN / 2 ^ subsample;

numFiles = length(fileNames);
for iterFiles = 1:numFiles,
	fprintf('Now running file %s, number %d out of %d.\n', fileNames{iterFiles}, iterFiles, numFiles);
	cube = getCube(fileNames{iterFiles}, [], 420:10:720);
	cube = cube / maxv(cube);
	cubedown = downSampleCube(cube, subsample);
	cubedown = compensateCube(cubedown, getCompensationFunction('sensitivity'));
	[B A] = robust_weighted_operator_pca_apg_mex(reshape(cubedown, [M * N, 31]), Y, W);
% 	[B A] = robust_weighted_operator_pca_apg_mex(reshape(cubedown, [M * N, 31]), Y, W, [], [], kappa);
	cube_background = reshape(B, [M N 31]);
	cube_foreground = reshape(A, [M N 31]);
	save(sprintf('robust_oppca_tau%g_def_%s_sens_sub%d.mat', tau, ...
		fileNames{iterFiles}(54:end), subsample), 'cube_background', 'cube_foreground');
end;
