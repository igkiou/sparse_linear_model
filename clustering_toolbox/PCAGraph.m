function PCAGraph(X,dims,labels)
%PCAGRAPH Plots data projected onto its first 2 principal components
% PCAGraph(X,dims,labels), where X = data, dims = no. of components
% to plot (2 or 3) and labels = class label of each entity (optional).

%(C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk

if nargin == 0
   help(mfilename);return
end

if nargin < 2
   dims=2;
end

[U,S,V]=svd(X);

W=diag(S);

Xcen=X-repmat(mean(X),length(X),1);			%centre data
Y=Xcen * V';

clf
if nargin < 3				%no labels
   if dims == 2
      plot(Y(:,1),Y(:,2),'.');
   else
      plot3(Y(:,1),Y(:,2),Y(:,3),'.');
	end
else							%plot each class with different marker
   for i = 1:max(labels)
      subset=(labels==i);
      if dims == 2
	      plot(Y(subset,1),Y(subset,2),PlotColour(i));hold on
      else
      	plot3(Y(subset,1),Y(subset,2),Y(subset,3),PlotColour(i));hold on
		end

  	end
end


