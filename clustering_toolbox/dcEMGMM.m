function [mix, class,likelihood]=dcEMGMM(DataSet,k, GType, InitType, InitCentres)
% DCEMGMM - estimates Gaussian Mixture Model using the EM algorithm
% Syntax: [mix, class,likelihood]=DCEMGMM(DataSet,k, GType, InitType, InitCentres)
% Clusters preference data using GMMEM algorithm from Netlab
% DataSet is nxd matrix or filename of file containing nxd matix
% k = number of clusters (components) to use
% GType = component shape; choose from {'spherical','diag','full'}
% InitType = 0 -> random data point selection
% InitType = 1 -> random + kmeans	(default)
% InitCentres = k*d matrix of initial cluster centres (defaults to random)
%
% "mix" is a structure defining the mixture model. "class" is a vector listing
% the estimated class label for each record. "likelihood" is the negative
% log likelihood of the model.
%
% NB This reqires the Netlab toolbox to be available.
%
% Copyright (C) David Corney 2000		D.Corney@cs.ucl.ac.uk

if size(DataSet,1)==1
   x=load(DataSet);
else
   x=DataSet;
end

if exist('InitCentres')==1 & size(InitCentres,1)~=k
   error('Try_kmeans: Number of "initial centres" doesn''t match "k"');
end

if nargin<5
   InitCentres = ChooseInitialCentres(x,k);
end

[n,d]=size(x);

if exist('GType')~=1
   GType = 'spherical';
   %GType = 'diag';
   %GType = 'full';
end

if exist('InitType')~=1
   InitType = 1;
end

mix = gmm(d, k, GType);	%full/diag/spherical

if k == 1
   mix.centres = mean(x);	%one cluster - no need to optimise
else   
   options = foptions;
	options(1) = -1; 		%no display for kmeans
   options(14) = 5;		%max iterations for initalisation
   mix = gmminit(mix, x, options);	%create structure and do k-means to initialise centres
   
   if InitType == 0		%overwrite centres with specified (or random) data points
		mix.centres = InitCentres;
   end
   
   options(1)  = -1;			% 1->Prints out error values.
   options(14) = 1000;		% Max. number of iterations.
   options(5)  = 1;     	% prevent covariance collapse
   mix.init_centres = mix.centres;		%Store starting point for later reference
   [mix, options, errlog]  = gmmem(mix, x, options);
	NumIts=size(errlog,1);
	if NumIts == options(14)
   	warning(' EM ran out of iterations!',1);
	end
end

%Derive class labels from model
centre=mix.centres;
[post, a] = gmmpost(mix, x);

[y,class]=max(post,[],2);

if nargout >=3
   likelihood=-sum(log(gmmprob(mix, x)));
end