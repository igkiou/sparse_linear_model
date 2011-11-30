SOURCE = '~/MATLAB/datasets_all/hyperspectral/mat_files/';
MATSOURCE = '~/MATLAB/sparse_linear_model/hyperspectral/sp_matfiles';
TARGET = '~/MATLAB/sparse_linear_model/hyperspectral/sp_figures';
fileNames = {'img1', 'imga6', 'imgb3', 'imgb8', 'imgc5', 'imgd3', 'imge0', 'imge5.mat',	'imgf3', 'imgf8',...
'img2', 'imga7', 'imgb4', 'imgb9', 'imgc7', 'imgd4', 'imge1', 'imge6.mat',	'imgf4', 'imgh0',...
'imga1', 'imgb0', 'imgb5', 'imgc1', 'imgc8', 'imgd7', 'imge2', 'imge7.mat',	'imgf5', 'imgh1',...
'imga2', 'imgb1', 'imgb6', 'imgc2', 'imgc9', 'imgd8', 'imge3', 'imgf1.mat',	'imgf6', 'imgh2',...
'imga5', 'imgb2', 'imgb7', 'imgc4', 'imgd2', 'imgd9', 'imge4', 'imgf2.mat',	'imgf7', 'imgh3',...
};

largeM = 1040;
largeN = 1392;
numFiles = length(fileNames);
sv = zeros(31, numFiles);
for iterFiles = 1:numFiles,
	fprintf('Now running file %s, number %d out of %d.\n', fileNames{iterFiles}, iterFiles, numFiles);
	load(sprintf('%s/%s', SOURCE, fileNames{iterFiles}), 'ref');
	cube = ref;
	cube = cube / maxv(cube);
	B = reshape(cube, [largeM * largeN, 31]);
	sv(:, iterFiles) = svd(B);
end;

E = sum(sv .^ 2, 1);
a = cumsum(sv.^2, 1) ./repmat(E, [31 1]);
mean_a = mean(a, 2);

figure;clf;set(gca,'fontsize',20);hold on
plot(1:31, mean_a, 'g-','LineWidth',4);
plot(1:31, a, '-','LineWidth',2);
plot(1:31, mean_a, 'g-','LineWidth',4);
legend('mean')
xlabel('Number of singular values');
ylabel('Portion of total energy')
axis([1 8 0.94 1.01])
grid on;
