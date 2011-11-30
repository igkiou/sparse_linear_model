function Phi = random_sensing(input1, numSamples)
%% parse inputs

if (isscalar(input1)),
	signalSize = input1; % if input1 is the length of the signal
else
	signalSize = size(input1, 1); % if input1 is a dictionary
end;

%% create random matrix
% randn('seed',0);
Phi = randn(numSamples, signalSize);
Phi = Phi / sqrt(numSamples);