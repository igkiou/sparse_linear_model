I = imread('/home/igkiou/MATLAB/datasets_all/BSDS300/images/test/102061.jpg');
I = rgb2gray(I);
I = im2double(I);
X = im2col(I,[8 8],'distinct');
figure;imshow(I);

mexmethod = 0; % 0: orig, 1: mex, 2: gradient 3: exact
dorandom = 0;
dolearn = 1;
dooptim = 0;
useinit = 1;

ompparam.L = 6;
ompparam.eps = 0.0001;
lassoparam.lambda = 0.0001;
% lassoparam.lambda = 0.001;
lassoparam.mode = 1;
method = 'lasso'; % 'lasso'
measurements = 18;
if (strcmp(method, 'omp') || strcmp(method, 'komp')),
	param = ompparam;
elseif (strcmp(method, 'lasso')),
	param = lassoparam;
end;

% load dictionaries/ksvd_VOC_64_256_nowhite
% D = learntDict;
% 
% if (useinit == 1),
% 	if (~exist('initPhi', 'var')),
% 		initPhi = random_sensing(learntDict, measurements);
% 	end;
% else
% 	initPhi = [];
% end;
% 
% if (dorandom == 1),
% 	Phi = random_sensing(D, measurements);
% 	Y = sparse_reconstruct(X, Phi, D, method, param);
% 	J = col2im(Y, [8 8], [size(I,1) size(I,2)], 'distinct');
% 	figure;imshow(J);
% 	title(sprintf('RMSE = %g, SSIM = %g', rmse(I, J), ssim_original(I, J)));
% end;
% 
% if (dolearn == 1),
% 	if (mexmethod == 0)
% 		Phi = learn_sensing(D, measurements, initPhi);	
% 	elseif (mexmethod == 1)
% 		Phi = learn_sensing_mex(D, measurements, initPhi);
% 	elseif (mexmethod == 2)
% 		Phi = learn_sensing_gradient(D, measurements, initPhi);
% 	elseif (mexmethod == 3)
% 		Phi = learn_sensing_exact(D, measurements);
% 	end;
% 	Y = sparse_reconstruct(X, Phi, D, method, param);
% 	J = col2im(Y, [8 8], [size(I,1) size(I,2)], 'distinct');
% 	figure;imshow(J);
% 	title(sprintf('RMSE = %g, SSIM = %g', rmse(I, J), ssim_original(I, J)));
% end;
% 
% if (dooptim == 1),
% 	load(sprintf('dictionaries/coupledksvd_VOC_64_256_nowhite_meas%d', measurements), 'learntProjMat');
% 	Phi = learntProjMat;
% 	Y = sparse_reconstruct(X, Phi, D, method, param);
% 	J = col2im(Y, [8 8], [size(I,1) size(I,2)], 'distinct');
% 	figure;imshow(J);
% 	title(sprintf('RMSE = %g, SSIM = %g', rmse(I, J), ssim_original(I, J)));
% end;

load(sprintf('dictionaries/coupledksvd_VOC_64_256_nowhite_meas%d', measurements));
D = learntDict;

if (dorandom == 1),
	Phi = random_sensing(D, measurements);
	Y = sparse_reconstruct(X, Phi, D, method, param);
	J = col2im(Y, [8 8], [size(I,1) size(I,2)], 'distinct');
	figure;imshow(J);
	title(sprintf('RMSE = %g, SSIM = %g', rmse(I, J), ssim_original(I, J)));
end;

if (dolearn == 1),
	if (mexmethod == 0)
		Phi = learn_sensing(D, measurements, initPhi);	
	elseif (mexmethod == 1)
		Phi = learn_sensing_mex(D, measurements, initPhi);
	elseif (mexmethod == 2)
		Phi = learn_sensing_gradient(D, measurements, initPhi);
	elseif (mexmethod == 3)
		Phi = learn_sensing_exact(D, measurements);
	end;
	Y = sparse_reconstruct(X, Phi, D, method, param);
	J = col2im(Y, [8 8], [size(I,1) size(I,2)], 'distinct');
	figure;imshow(J);
	title(sprintf('RMSE = %g, SSIM = %g', rmse(I, J), ssim_original(I, J)));
end;

if (dooptim == 1),
	Phi = learntProjMat;
	Y = sparse_reconstruct(X, Phi, D, method, param);
	J = col2im(Y, [8 8], [size(I,1) size(I,2)], 'distinct');
	figure;imshow(J);
	title(sprintf('RMSE = %g, SSIM = %g', rmse(I, J), ssim_original(I, J)));
end;
