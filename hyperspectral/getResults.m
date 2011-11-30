name = 'yard4';
wavelength = 420:10:720;
wl = wavelength;
tau = 10;
kernelMat = kernel_gram(wavelength, [], 'h', tau);
[U S] = eig(kernelMat);
Y10 = U * sqrt(S);
tau = 100;
kernelMat = kernel_gram(wavelength, [], 'h', tau);
[U S] = eig(kernelMat);
Y100 = U * sqrt(S);
tau = 1000;
kernelMat = kernel_gram(wavelength, [], 'h', tau);
[U S] = eig(kernelMat);
Y1000 = U * sqrt(S);

cube = getCube(sprintf('/home/igkiou/MATLAB/datasets_all/hyperspectral/moving/tif_files/%s', name), wl, 1);
cube = cube / maxv(cube);
cubergb = cube2rgb(cube, getCompensationFunction('sensitivity'));
figure; imshow((cubergb/4).^(1/2.2))

cubedown = downSampleCube(cube, 3);
cubedownrgb = cube2rgb(cubedown, getCompensationFunction('sensitivity'));
cubedownrgb = circshift(cubedownrgb, [-1 -1 0]);
figure; imshow((cubedownrgb/4).^(1/2.2))
% figure; imshow((cubedownrgb).^(1/2.2)/maxv((cubedownrgb).^(1/2.2)), [])

load(sprintf('matfiles/robust_pca_default_%s_sub3.mat', name));
cubedownrgb = cube2rgb(imflip(cube_background), getCompensationFunction('sensitivity'));
figure; imshow((cubedownrgb/4).^(1/2.2))
% figure; imshow((cubedownrgb).^(1/2.2)/maxv((cubedownrgb).^(1/2.2)), [])

load(sprintf('matfiles/robust_oppca_tau10_def_%s_sub3.mat', name));
cubedownrgb = cube2rgb(imflip(reshape(reshape(cube_background,[130*174 31])*Y10',[130 174 31])), getCompensationFunction('sensitivity'));
figure; imshow((cubedownrgb/4).^(1/2.2))
% figure; imshow((cubedownrgb).^(1/2.2)/maxv((cubedownrgb).^(1/2.2)), [])

load(sprintf('matfiles/robust_oppca_tau100_def_%s_sub3.mat', name));
cubedownrgb = cube2rgb(imflip(reshape(reshape(cube_background,[130*174 31])*Y100',[130 174 31])), getCompensationFunction('sensitivity'));
figure; imshow((cubedownrgb/4).^(1/2.2))
% figure; imshow((cubedownrgb).^(1/2.2)/maxv((cubedownrgb).^(1/2.2)), [])

load(sprintf('matfiles/robust_oppca_tau1000_def_%s_sub3.mat', name));
cubedownrgb = cube2rgb(imflip(reshape(reshape(cube_background,[130*174 31])*Y1000',[130 174 31])), getCompensationFunction('sensitivity'));
figure; imshow((cubedownrgb/4).^(1/2.2))
% figure; imshow((cubedownrgb).^(1/2.2)/maxv((cubedownrgb).^(1/2.2)), [])
