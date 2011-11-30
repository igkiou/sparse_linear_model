function [trainPatches testPatches] = getTexturePatches(textureClass, patchSize, slideType, useNorm, largeExp)

if (nargin < 2),
	patchSize = 12;
end;
if (nargin < 3),
	slideType = 'distinct';
end;
if (nargin < 4),
	useNorm = 1;
end;
if (nargin < 5),
	largeExp = 1;
end;

if (largeExp == 1),
% 	eval(sprintf('Ilarge = im2double(imread(''~/MATLAB/datasets_all/brodatz/textures/D%d.gif''));', textureClass));
	load(sprintf('~/MATLAB/datasets_all/brodatz/textures/D%d.mat', textureClass));
	L = Ilarge(257:end,:);
	UR = Ilarge(1:256, 257:end);
	trainPatches = im2col(L, [patchSize patchSize], slideType);
	trainPatches = [trainPatches im2col(UR, [patchSize patchSize], slideType)];
	UL = Ilarge(1:256, 1:256);
	testPatches = im2col(UL, [patchSize patchSize], slideType);
else
	if (textureClass == 5),
		Itrain = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D5D92_1.pgm'));
		Itest = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D5D92.pgm'));
		Itest = Itest(1:256, 1:256);
	elseif (textureClass == 92),
		Itrain = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D5D92_2.pgm'));
		Itest = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D5D92.pgm'));
		Itest = Itest(1:256, 257:512);
	elseif (textureClass == 4),
		Itrain = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D4D84_1.pgm'));
		Itest = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D4D84.pgm'));
		Itest = Itest(1:256, 1:256);
	elseif (textureClass == 84),
		Itrain = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D4D84_2.pgm'));
		Itest = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D4D84.pgm'));
		Itest = Itest(1:256, 257:512);
	elseif (textureClass == 8),
		Itrain = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D8D84_1.pgm'));
		Itest = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D8D84.pgm'));
		Itest = Itest(1:256, 1:256);
	elseif (textureClass == 12),
		Itrain = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D12D17_1.pgm'));
		Itest = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D12D17.pgm'));
		Itest = Itest(1:256, 1:256);
	elseif (textureClass == 17),
		Itrain = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D12D17_2.pgm'));
		Itest = im2double(imread('~/MATLAB/datasets_all/brodatz/training/2-texture/D12D17.pgm'));
		Itest = Itest(1:256, 257:512);
	end;
	trainPatches = im2col(Itrain, [patchSize patchSize], slideType);
	testPatches = im2col(Itest, [patchSize patchSize], slideType);
end;

if (useNorm == 1)
	trainPatches = normcols(trainPatches);
	testPatches = normcols(testPatches);
end;
