function [Y, class_name] = oneofc(label)
%
% Usage: [Y, class_name] = oneofc(label)
% 0ne-of-c coding for multi-class classification
%
% Example:
%
% [Y, class_name] = oneofc([3, 2, 1, 2, 4])
%  Y =
% 
%      1     0     0     0
%      0     1     0     0
%      0     0     1     0
%      0     1     0     0
%      0     0     0     1
% 
% 
%  class_name =
% 
%      3     2     1     4
%
% See also: 
%      oneofc_inv.m
% Kai Yu, kai.yu@gmx.net, Siemens AG

% TODO: Also write converter to 1:numLabels
N = length(label);

class_num = 0;
class_name = [];
class_column = zeros(N, 1)-1;

for i = 1 : N
    if i == 1
        class_num = 1;
        class_name = [class_name, label(1)];
    end
    for j = 1 : class_num
        if label(i) == class_name(j) 
            class_column(i) = j;
        end
    end
    if class_column(i) == -1
        class_num = class_num + 1;
        class_name = [class_name, label(i)];
        class_column(i) = class_num;
    end
end

Y = -1 * ones(N, class_num);
for i = 1 : N
    Y(i, class_column(i)) = 1;
end
Y = Y';
