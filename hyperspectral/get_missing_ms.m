SOURCE = '~/MATLAB/datasets_all/multispectral';
MATSOURCE = '~/MATLAB/sparse_linear_model/hyperspectral/sp_matfiles';
TARGET = '~/MATLAB/sparse_linear_model/hyperspectral/sp_figures';
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
vec_miss = 4:8:31;
vec(vec_miss) = [];
wavelengths_all = 400:10:700;
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
	cube = getCube(sprintf('%s/%s/%s', SOURCE, fileNames{iterFiles}, fileNames{iterFiles}), 'png', 1:31, 0);
	cube = cube / maxv(cube);
	cubedown = downSampleCube(cube, samplingFactor);
	maxcube = maxv(cubedown) * 4;
	rgbIm = cube2rgb2(cubedown)/maxcube;
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
		rgbIm = cube2rgb2(cube1)/maxcube;
% 		figure; imshow(rgbIm)
% 		eval(sprintf('print -depsc2 %s/%s_sub%d_per%g_op%g.eps', TARGET, fileNames{iterFiles}, samplingFactor, per, tau));
% 		eval(sprintf('imwrite(rgbIm, ''%s/%s_sub%d_per%g_op%g.png'')', TARGET, fileNames{iterFiles}, samplingFactor, per, tau));
% 		close
	end
end;
