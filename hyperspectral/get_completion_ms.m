SOURCE = '~/MATLAB/datasets_all/multispectral';
MATSOURCE = '~/MATLAB/sparse_linear_model/hyperspectral/ms_comp_matfiles';
TARGET = '~/MATLAB/sparse_linear_model/hyperspectral/ms_comp_figures_norm';
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
perVec = [0.2 0.4];
M = largeM / 2 ^ samplingFactor;
N = largeN / 2 ^ samplingFactor;

numFiles = length(fileNames);
numPer = length(perVec);
MSE = zeros(2 * length(perVec), numFiles);

for iterFiles = 1:numFiles,
	fprintf('Now running file %s, number %d out of %d.\n', fileNames{iterFiles}, iterFiles, numFiles);
	cube = getCube(sprintf('%s/%s/%s', SOURCE, fileNames{iterFiles}, fileNames{iterFiles}), 'png', 1:31, 0);
	cube = cube / maxv(cube);
	cubedown = downSampleCube(cube, samplingFactor);
	maxcube = maxv(cubedown) * 4;
	rgbIm = cube2rgb2(cubedown)/maxcube;
	figure; imshow(rgbIm)
	eval(sprintf('print -depsc2 %s/%s_sub%d_orig.eps', TARGET, fileNames{iterFiles}, samplingFactor));
	eval(sprintf('imwrite(rgbIm, ''%s/%s_sub%d_orig.png'')', TARGET, fileNames{iterFiles}, samplingFactor));
	close
	for iterPer = 1:numPer,
		per = perVec(iterPer);
		
		load(sprintf('%s/matrix_per%g_def_%s_sub%d.mat', MATSOURCE, per,...
		fileNames{iterFiles}, samplingFactor),...
		'cube_recovered');
		MSE((iterPer - 1) * 2 + 1, iterFiles) = sum((cubedown(:) - cube_recovered(:)) .^ 2) / numel(cubedown);
		rgbIm = cube2rgb2(cube_recovered)/maxcube;
		figure; imshow(rgbIm);
		eval(sprintf('print -depsc2 %s/%s_sub%d_per%g_matrix.eps', TARGET, fileNames{iterFiles}, samplingFactor, per));
		eval(sprintf('imwrite(rgbIm, ''%s/%s_sub%d_per%g_matrix.png'')', TARGET, fileNames{iterFiles}, samplingFactor, per));
		close
			
		load(sprintf('%s/operator_per%g_tau%g_def_%s_sub%d.mat', MATSOURCE, per, tau,...
		fileNames{iterFiles}, samplingFactor),...
		'cube_recovered');
		cube1 = applyKernelFactor(cube_recovered, Y);
		MSE((iterPer - 1) * 2 + 2, iterFiles) = sum((cubedown(:) - cube1(:)) .^ 2) / numel(cubedown);
		rgbIm = cube2rgb2(cube1)/maxcube;
		figure; imshow(rgbIm)
		eval(sprintf('print -depsc2 %s/%s_sub%d_per%g_op%g.eps', TARGET, fileNames{iterFiles}, samplingFactor, per, tau));
		eval(sprintf('imwrite(rgbIm, ''%s/%s_sub%d_per%g_op%g.png'')', TARGET, fileNames{iterFiles}, samplingFactor, per, tau));
		close
	end
end;
