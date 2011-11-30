function database = calculateSiftFeatures(className, numPics, gridSpacing, patchSize, maxImSize, nrml_threshold)

database = [];

database.cname = {}; % name of each class
database.label = []; % label of each class
database.path = {}; % contain the pathes for each image of each class
database.nclass = 2;

rt_img_dir = ['/home/igkiou/MATLAB/datasets_all/Graz_02/', className, '/'];
rt_data_dir = ['/home/igkiou/MATLAB/sparse_linear_model/results/Graz_experiments/sift_features/', className, '/'];

siftpath = rt_data_dir;
if ~isdir(siftpath),
	mkdir(siftpath);
end;

frames = dir([rt_img_dir, '*.bmp']);
c_num = length(frames);           
if (c_num < numPics),
	warning('Category %s contains less than NUMPICS = %d images.', className, numPics);
else
	c_num = numPics;
end;

database.imnum = c_num;

for jj = 1:c_num,
	imgpath = [rt_img_dir, frames(jj).name];
	disp(sprintf('Now processing figure no. %d out of %d: %s', jj, c_num, imgpath));

	I = imread(imgpath);
	if ndims(I) == 3,
		I = im2double(rgb2gray(I));
	else
		I = im2double(I);
	end;
	if (~strcmp(className, 'none')),
		gtpath = strrep(imgpath, [className, '/'], 'groundtruth/');
		gtpath = strrep(gtpath, '.bmp', '_gt.jpg');
		GT = imread(gtpath);
	else
		GT = ones(size(I));
	end;

	[im_h, im_w] = size(I);

	if max(im_h, im_w) > maxImSize,
		I = imresize(I, maxImSize/max(im_h, im_w), 'bicubic');
		[im_h, im_w] = size(I);
	end;

	% make grid sampling SIFT descriptors
	remX = mod(im_w-patchSize,gridSpacing);
	offsetX = floor(remX/2)+1;
	remY = mod(im_h-patchSize,gridSpacing);
	offsetY = floor(remY/2)+1;

	[gridX,gridY] = meshgrid(offsetX:gridSpacing:im_w-patchSize+1, offsetY:gridSpacing:im_h-patchSize+1);

	% find SIFT descriptors
	[siftArr labelArr] = getSiftGrid(I, GT, gridX, gridY, patchSize, 0.8);
	siftArr = normalizeSiftFeatures(siftArr, nrml_threshold);

	feaSet.feaArr = siftArr';
	feaSet.feaLabels = labelArr';
	feaSet.x = gridX(:) + patchSize / 2 - 0.5;
	feaSet.y = gridY(:) + patchSize / 2 - 0.5;
	feaSet.width = im_w;
	feaSet.height = im_h;

	[pdir, fname] = fileparts(frames(jj).name);                        
	fpath = [rt_data_dir, strrep(frames(jj).name, '.bmp', '.mat')];

	save(fpath, 'feaSet');
	database.path = [database.path, fpath];
end;    
