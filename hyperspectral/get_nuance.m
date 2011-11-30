SOURCE = '~/MATLAB/datasets_all/hyperspectral/subset/';
MATSOURCE = '~/MATLAB/sparse_linear_model/hyperspectral/nuance_matfiles';
TARGET = '~/MATLAB/sparse_linear_model/hyperspectral/nuance_figures_norm';
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
psVec = [0.05 0.1 0.2];
M = largeM / 2 ^ samplingFactor;
N = largeN / 2 ^ samplingFactor;

numFiles = length(fileNames);
numPs = length(psVec);
MSE = zeros(2 * length(psVec), numFiles);

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
	for iterPs = 1:numPs,
		ps = psVec(iterPs);
		load(sprintf('%s/robust_pca_ps%g_def_%s_sub%d.mat', MATSOURCE, ps,...
			fileNames{iterFiles}, samplingFactor),...
			'cube_background', 'noisevec');

		cubedownnoise = cubedown + reshape(noisevec, size(cubedown));
		rgbIm = cube2rgb(cubedownnoise, getCompensationFunction('sensitivity'))/maxcube;
% 		figure; imshow(rgbIm / maxv(rgbIm));
% 		eval(sprintf('print -depsc2 %s/%s_sub%d_ps%g_noisy.eps', TARGET, fileNames{iterFiles}, samplingFactor, ps));
% 		eval(sprintf('imwrite(rgbIm / maxv(rgbIm), ''%s/%s_sub%d_ps%g_noisy.png'')', TARGET, fileNames{iterFiles}, samplingFactor, ps));
% 		close
	
		MSE((iterPs - 1) * 2 + 1, iterFiles) = sum((cubedown(:) - cube_background(:)) .^ 2) / numel(cubedown);
		rgbIm = cube2rgb(cube_background, getCompensationFunction('sensitivity'))/maxcube;
% 		figure; imshow(rgbIm);
% 		eval(sprintf('print -depsc2 %s/%s_sub%d_ps%g_pca.eps', TARGET, fileNames{iterFiles}, samplingFactor, ps));
% 		eval(sprintf('imwrite(rgbIm, ''%s/%s_sub%d_ps%g_pca.png'')', TARGET, fileNames{iterFiles}, samplingFactor, ps));
% 		close

		errorImg = sqrt(sum((cubedown - cube_background).^2,3));
% 		figure; imshow(errorImg); colormap jet
% 		eval(sprintf('print -depsc2 %s/%s_sub%d_ps%g_pca_error.eps', TARGET, fileNames{iterFiles}, samplingFactor, ps));
		eval(sprintf('imwrite(im2uint8(errorImg), jet(256), ''%s/%s_sub%d_ps%g_pca_error.png'')', TARGET, fileNames{iterFiles}, samplingFactor, ps));
% 		close

		load(sprintf('%s/robust_oppca_ps%g_tau%g_def_%s_sub%d.mat', MATSOURCE, ps, tau,...
		fileNames{iterFiles}, samplingFactor),...
		'cube_background');
		cube1 = applyKernelFactor(cube_background, Y);
		MSE((iterPs - 1) * 2 + 2, iterFiles) = sum((cubedown(:) - cube1(:)) .^ 2) / numel(cubedown);
		rgbIm = cube2rgb(cube1, getCompensationFunction('sensitivity'))/maxcube;
% 		figure; imshow(rgbIm)
% 		eval(sprintf('print -depsc2 %s/%s_sub%d_ps%g_oppca%g.eps', TARGET, fileNames{iterFiles}, samplingFactor, ps, tau));
% 		eval(sprintf('imwrite(rgbIm, ''%s/%s_sub%d_ps%g_oppca%g.png'')', TARGET, fileNames{iterFiles}, samplingFactor, ps, tau));
% 		close

		errorImg = sqrt(sum((cubedown - cube1).^2,3));
% 		figure; imshow(errorImg); colormap jet
% 		eval(sprintf('print -depsc2 %s/%s_sub%d_ps%g_oppca%g_error.eps', TARGET, fileNames{iterFiles}, samplingFactor, ps, tau));
		eval(sprintf('imwrite(im2uint8(errorImg), jet(256), ''%s/%s_sub%d_ps%g_oppca%g_error.png'')', TARGET, fileNames{iterFiles}, samplingFactor, ps, tau));
% 		close
	end
end;
