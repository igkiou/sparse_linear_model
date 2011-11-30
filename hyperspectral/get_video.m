% lawgate3
cube = getCube('~/MATLAB/datasets_all/hyperspectral/moving/tif_files/lawgate3', 'tif', 420:10:720, 1);
cube = cube / maxv(cube);
cubedown = downSampleCube(cube, 2);
im = cube2rgb(cubedown, getCompensationFunction('sensitivity'));
figure; imshow(im/3.25)

load video_matfiles/robust_oppca_tau20_def_lawgate3_sub2
imo = cube2rgb(cube_foreground, getCompensationFunction('sensitivity'));
figure; imshow(imo / 2);
cube_background = applyKernelFactor(cube_background, getKernelFactor(420:10:720, 20));
imo = cube2rgb(cube_background, getCompensationFunction('sensitivity'));
figure; imshow(imo / 3);

load video_matfiles/robust_pca_def_lawgate3_sub2
imb = cube2rgb(cube_foreground, getCompensationFunction('sensitivity'));
figure; imshow(imb / 2);
imb = cube2rgb(cube_background, getCompensationFunction('sensitivity'));
figure; imshow(imb / 3);

% gsd1
cube = getCube('~/MATLAB/datasets_all/hyperspectral/moving/tif_files/gsd1', 'tif', 420:10:720, 1);
cube = cube / maxv(cube);
cubedown = downSampleCube(cube, 2);
im = cube2rgb(cubedown, getCompensationFunction('sensitivity'));
figure; imshow(im/3.5)

load video_matfiles/robust_oppca_tau20_def_gsd1_sub2
imo = cube2rgb(cube_foreground, getCompensationFunction('sensitivity'));
figure; imshow(imo / 3);
cube_background = applyKernelFactor(cube_background, getKernelFactor(420:10:720, 20));
imo = cube2rgb(cube_background, getCompensationFunction('sensitivity'));
figure; imshow(imo / 3);

load video_matfiles/robust_pca_def_gsd1_sub2
imb = cube2rgb(cube_foreground, getCompensationFunction('sensitivity'));
figure; imshow(imb / 3);
imb = cube2rgb(cube_background, getCompensationFunction('sensitivity'));
figure; imshow(imb / 3);

% yard5
cube = getCube('~/MATLAB/datasets_all/hyperspectral/moving/tif_files/yard5', 'tif', 420:10:720, 1);
cube = cube / maxv(cube);
cubedown = downSampleCube(cube, 2);
im = cube2rgb(cubedown, getCompensationFunction('sensitivity'));
figure; imshow(im.^1.5/6.5)

load video_matfiles/robust_oppca_tau20_def_yard5_sub2
imo = cube2rgb(cube_foreground, getCompensationFunction('sensitivity'));
figure; imshow(imo / 2);
cube_background = applyKernelFactor(cube_background, getKernelFactor(420:10:720, 20));
imo = cube2rgb(cube_background, getCompensationFunction('sensitivity'));
figure; imshow(imo / 3);

load video_matfiles/robust_pca_def_yard5_sub2
imb = cube2rgb(cube_foreground, getCompensationFunction('sensitivity'));
figure; imshow(imb / 2);
imb = cube2rgb(cube_background, getCompensationFunction('sensitivity'));
figure; imshow(imb / 3);
