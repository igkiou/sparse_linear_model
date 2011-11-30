%%
wavelength = 420:10:720;
numWavelength = length(wavelength);
a = colormap(jet(numWavelength)); close; 
M = 260;
N = 348;
K = 3;
istart = 13;
jstart = 3;
width = 7;
height = 5;
fps = 5;

%%
% load(sprintf('/home/igkiou/MATLAB/translucency/acquisition/textMasks_%d_%d.mat',...
% 	istart, jstart), 'textMasks');

SOURCE = '~/MATLAB/datasets_all/hyperspectral/moving/tif_files';
MATSOURCE = '~/MATLAB/sparse_linear_model/hyperspectral/video_matfiles';
TARGET = '~/MATLAB/sparse_linear_model/hyperspectral/video_avifiles';
fileNames = {...
	'gsd1',...
	'lawgate3',...
	'yard3',...
	'lawlibrary7',...
 	'yard5'...
};
numFiles = length(fileNames);

samplingFactor = 2;
kappa = 0.1;
tau = 20;
kernelMat = kernel_gram(wavelength, [], 'h', tau);
[U S] = eig(kernelMat);
Y = U * sqrt(S);
W = eye(31);
colorVec = jet(numWavelength);
colorVecNorm = bsxfun(@rdivide, colorVec, sqrt(sum(colorVec .^ 2, 2)));

compensationFunction = getCompensationFunction('sensitivity');
for iterFile = 1:numFiles,
	fprintf('Now running scene %s, number %d out of %d.\n', fileNames{iterFile}, iterFile, numFiles);
	
	% load orig file
	cube = getCube(sprintf('%s/%s', SOURCE, fileNames{iterFile}), 'tif', wavelength, 1);
	cube = cube / maxv(cube);
	cubedown = downSampleCube(cube, samplingFactor);
	cubedown = compensateCube(cubedown, compensationFunction);
	
	% write rainbow version of orig
	writeVideo(sprintf('%s/%s_sub%d_orig_rainbow.avi', TARGET, fileNames{iterFile}, samplingFactor),...
				cubedown, [], fps, [], colorVecNorm);
	
	% write gray annot version of orig
	aviobj = avifile(sprintf('%s/%s_sub%d_orig_annot.avi', TARGET, fileNames{iterFile}, samplingFactor), 'fps', fps);
	for iter = 1:numWavelength,
		P = cubedown(:, :, iter);
		P = repmat(P, [1 1 3]);
		for iter2 = 1:iter,
			P(istart + height * (numWavelength - iter2) + (1:height), jstart + (1:width), :) =...
				repmat(reshape(colorVec(iter2, :), [1 1 3]), [height, width, 1]);
		end;
		Q = im2frame(im2uint8(P));
		aviobj = addframe(aviobj, Q);
	end;
	aviobj = close(aviobj);
	
	% load pca file
	load(sprintf('%s/robust_pca_def_%s_sub%d.mat', MATSOURCE,...
	fileNames{iterFile}, samplingFactor),...
	'cube_background', 'cube_foreground');
	cube_background = compensateCube(cube_background, compensationFunction);
	cube_foreground = compensateCube(cube_foreground, compensationFunction);

	% write rainbow version of pca
	writeVideo(sprintf('%s/%s_sub%d_pca_back_rainbow.avi', TARGET, fileNames{iterFile}, samplingFactor),...
				cube_background, [], fps, [], colorVecNorm);
	writeVideo(sprintf('%s/%s_sub%d_pca_fore_rainbow.avi', TARGET, fileNames{iterFile}, samplingFactor),...
				cube_foreground, [], fps, [], colorVecNorm);
	
	% write gray annot version of pca
	aviobj = avifile(sprintf('%s/%s_sub%d_pca_back_annot.avi', TARGET, fileNames{iterFile}, samplingFactor), 'fps', fps);
	for iter = 1:numWavelength,
		P = cube_background(:, :, iter);
		P = repmat(P, [1 1 3]);
		for iter2 = 1:iter,
			P(istart + height * (numWavelength - iter2) + (1:height), jstart + (1:width), :) =...
				repmat(reshape(colorVec(iter2, :), [1 1 3]), [height, width, 1]);
		end;
		Q = im2frame(im2uint8(P));
		aviobj = addframe(aviobj, Q);
	end;
	aviobj = close(aviobj);
	
	aviobj = avifile(sprintf('%s/%s_sub%d_pca_fore_annot.avi', TARGET, fileNames{iterFile}, samplingFactor), 'fps', fps);
	for iter = 1:numWavelength,
		P = cube_foreground(:, :, iter);
		P = repmat(P, [1 1 3]);
		for iter2 = 1:iter,
			P(istart + height * (numWavelength - iter2) + (1:height), jstart + (1:width), :) =...
				repmat(reshape(colorVec(iter2, :), [1 1 3]), [height, width, 1]);
		end;
		Q = im2frame(im2uint8(P));
		aviobj = addframe(aviobj, Q);
	end;
	aviobj = close(aviobj);
	
	% load oppca file
	load(sprintf('%s/robust_oppca_tau%g_def_%s_sub%d.mat', MATSOURCE, tau,...
	fileNames{iterFile}, samplingFactor),...
	'cube_background', 'cube_foreground');
	cube_background = applyKernelFactor(cube_background, Y);
	cube_background = compensateCube(cube_background, compensationFunction);
	cube_foreground = compensateCube(cube_foreground, compensationFunction);

	% write rainbow version of pca
	writeVideo(sprintf('%s/%s_sub%d_oppca_tau%g_back_rainbow.avi', TARGET, fileNames{iterFile}, samplingFactor, tau),...
				cube_background, [], fps, [], colorVecNorm);
	writeVideo(sprintf('%s/%s_sub%d_oppca_tau%g_fore_rainbow.avi', TARGET, fileNames{iterFile}, samplingFactor, tau),...
				cube_foreground, [], fps, [], colorVecNorm);
	
	% write gray annot version of pca
	aviobj = avifile(sprintf('%s/%s_sub%d_oppca_tau%g_back_annot.avi', TARGET, fileNames{iterFile}, samplingFactor, tau),...
				'fps', fps);
	for iter = 1:numWavelength,
		P = cube_background(:, :, iter);
		P = repmat(P, [1 1 3]);
		for iter2 = 1:iter,
			P(istart + height * (numWavelength - iter2) + (1:height), jstart + (1:width), :) =...
				repmat(reshape(colorVec(iter2, :), [1 1 3]), [height, width, 1]);
		end;
		Q = im2frame(im2uint8(P));
		aviobj = addframe(aviobj, Q);
	end;
	aviobj = close(aviobj);
	
	aviobj = avifile(sprintf('%s/%s_sub%d_oppca_tau%g_fore_annot.avi', TARGET, fileNames{iterFile}, samplingFactor, tau),...
				'fps', fps);
	for iter = 1:numWavelength,
		P = cube_foreground(:, :, iter);
		P = repmat(P, [1 1 3]);
		for iter2 = 1:iter,
			P(istart + height * (numWavelength - iter2) + (1:height), jstart + (1:width), :) =...
				repmat(reshape(colorVec(iter2, :), [1 1 3]), [height, width, 1]);
		end;
		Q = im2frame(im2uint8(P));
		aviobj = addframe(aviobj, Q);
	end;
	aviobj = close(aviobj);
end;
