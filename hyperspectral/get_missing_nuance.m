SOURCE = '~/MATLAB/datasets_all/hyperspectral/subset/';
MATSOURCE = '~/MATLAB/sparse_linear_model/hyperspectral/sp_matfiles';
TARGET = '~/MATLAB/sparse_linear_model/hyperspectral/sp_figures';
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
vec_miss = 4:8:31;
vec(vec_miss) = [];
wavelengths_all = 420:10:720;
wavelengths = wavelengths_all(vec);
wavelengths_miss = wavelengths_all(vec_miss);
Y = getKernelFactor(wavelengths, tau);
L = Y' * Y;
k = kernel_gram(wavelengths, wavelengths_miss);
W = eye(31);
perVec = 0.4;
M = largeM / 2 ^ samplingFactor;
N = largeN / 2 ^ samplingFactor;

numFiles = length(fileNames);
numPer = length(perVec);
MSE = zeros(2 * length(perVec), numFiles);
MSE_band = zeros(2 * length(perVec), numFiles);

for iterFiles = 1:numFiles,
	fprintf('Now running file %s, number %d out of %d.\n', fileNames{iterFiles}, iterFiles, numFiles);
	load(sprintf('%s/%s.mat', SOURCE, fileNames{iterFiles}), 'ref');
	cube = ref;
	cube = cube / maxv(cube);
	cubedown = downSampleCube(cube, samplingFactor);
	maxcube = maxv(cubedown) * 4;
	rgbIm = cube2rgb(cubedown, getCompensationFunction('sensitivity'))/maxcube;
% 	figure; imshow(rgbIm)
% 	eval(sprintf('print -depsc2 %s/%s_sub%d_orig.eps', TARGET, fileNames{iterFiles}, samplingFactor));
% 	eval(sprintf('imwrite(rgbIm, ''%s/%s_sub%d_orig.png'')', TARGET, fileNames{iterFiles}, samplingFactor));
% 	close
	for iterPer = 1:numPer,
		per = perVec(iterPer);
				
		load(sprintf('%s/operator_per%g_tau%g_def_%s_sub%d_sp%d.mat', MATSOURCE, per, tau,...
		fileNames{iterFiles}, samplingFactor, length(vec)),...
		'cube_recovered');
	
		B1 = reshape(cube_recovered, [M * N, length(vec)]) * Y';
		B2 = reshape(cube_recovered, [M * N, length(vec)]) * diag(1./sqrt(diag(L))) * Y' * k;
		B = zeros(M * N, 31);
		B(:, vec) = B1;
		B(:, vec_miss) = B2;
		cube_inferred = reshape(B, [M, N, 31]);
		cube1 = cube_inferred;
	
		MSE((iterPer - 1) * 2 + 2, iterFiles) = sum((cubedown(:) - cube1(:)) .^ 2) / numel(cubedown);
		gkiou = cubedown(:, :, vec_miss) - cube1(:, :, vec_miss);
		MSE_band((iterPer - 1) * 2 + 2, iterFiles) =...
			sum(gkiou(:) .^ 2) / numel(cubedown(:, :, vec_miss));
		rgbIm = cube2rgb(cube1, getCompensationFunction('sensitivity'))/maxcube;
% 		figure; imshow(rgbIm)
% 		eval(sprintf('print -depsc2 %s/%s_sub%d_per%g_op%g.eps', TARGET, fileNames{iterFiles}, samplingFactor, per, tau));
% 		eval(sprintf('imwrite(rgbIm, ''%s/%s_sub%d_per%g_op%g.png'')', TARGET, fileNames{iterFiles}, samplingFactor, per, tau));
% 		close
	end
end;
