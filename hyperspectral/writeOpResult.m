function writeOpResult(fileName, Y, compensation, fps, flipFlag, SOURCE, TARGET)

if ((nargin < 6) || (isempty(SOURCE)))
	SOURCE = '.';
end;

if ((nargin < 7) || (isempty(TARGET)))
	TARGET = '.';
end;

load(sprintf('%s/%s.mat', SOURCE, fileName));
[M N O] = size(cube_background);
cube_background = reshape(reshape(cube_background, [M * N, O]) * Y', [M, N, O]);
writeVideo(sprintf('%s/%s_fore.avi', TARGET, fileName), cube_foreground, compensation, fps, flipFlag);
writeVideo(sprintf('%s/%s_back.avi', TARGET, fileName), cube_background, compensation, fps, flipFlag);
