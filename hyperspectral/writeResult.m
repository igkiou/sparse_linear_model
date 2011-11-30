function writeResult(fileName, compensation, fps, flipFlag, SOURCE, TARGET)

if ((nargin < 5) || (isempty(SOURCE)))
	SOURCE = '.';
end;

if ((nargin < 5) || (isempty(TARGET)))
	TARGET = '.';
end;

load(sprintf('%s/%s.mat', SOURCE, fileName));
writeVideo(sprintf('%s/%s_fore.avi', TARGET, fileName), cube_foreground, compensation, fps, flipFlag);
writeVideo(sprintf('%s/%s_back.avi', TARGET, fileName), cube_background, compensation, fps, flipFlag);
