function c=PlotColour(index,lineflag)
%PLOTCOLOUR Returns colour and marker string for PLOT
% c=PLOTCOLOUR(index) where index is an integer, return
% next marker string in sequence

%(C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk

list=['b.';'rx';'ko';'cs';'g*';'md';'y+';'bo';'rs';'k.';'c*';];

index=mod(index,size(list,1));
if index==0, index=size(list,1);end
c=list(index,:);


if exist('lineflag') & lineflag == 1
   c=c(1);
end

