function label = oneofc_inv(Y, class_name)

% Usage: label = oneofc_inv(Y, class_name)
%
% Covert 1-of-c coding to an one-dim output vector. For each row,
% we pick up the index (idx) of the column having the largest entry in
% this row, and pick the the class_name(idx) as the output.
%
% Inputs: 
%      Y          - (n x c) matrix
%      class_name - (n x 1) vector
%
% Outputs:
%      label - (n x 1) vector
%
% Example: 
% Y =
%      1     0     0     0
%      0     1     0     0
%      0     0     1     0
%      0     1     0     0
%      0     0     0     1
% class_name =
% 
%      3     2     1     4
%      
% oneofc_inv(Y, class_name)
% 
% ans =
% 
%      3     2     1     2     4
%
% See also, 
%      oneofc.m
%
% kai.yu@siemens.com
% Kai Yu, Siemens AG

[M, N] = size(Y);

error( nargchk(1, 2, nargin));
if nargin < 2
    class_name = [];
end
if isempty(class_name)
    class_name = [1:M];
end

% idx = Y*[1:M]';
[dummy, idx] = max(Y);

label = zeros(1, N);
label = class_name(idx);
