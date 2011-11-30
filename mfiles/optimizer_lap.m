function L=optimizer_lap(XLXt,numSamples,L,rD,dict,ReguAlpha,ReguBeta,Numiters)
%
% function [L,Det]=lmnn(x,y,Kg,'Parameter1',Value1,'Parameter2',Value2,...);
%
% Input:
%
% x = input matrix (dxn, each column is an input vector) 
% y = labels (row vector 1xn)
% (*optional*) Kg = attract Kg nearest similar labeled vectos
% (*optional*)  L = initial transformation matrix (e.g eye(size(x,1)))
% (*optional*) rD = reduced dimension (default no reduction)
% (*optional*) dict = dictionary to be used (default no dictionary)
% 
% Parameters:
% stepsize = (default 1e-09)
% tempid = (def 0) saves state every 10 iterations in temp#.mat
% save = (def 0) save the initial computation 
% skip = (def 0) loads the initial computation instead of
%        recomputing (only works if previous run was on exactly the same data)
% correction = (def 15) how many steps between each update 
%              The number of impostors are fixed for until next "correction"
% factor = (def 1.1) multiplicative factor by which the
%         "correction" gap increases
% obj = (def 1) if 1, solver solves in L, if 0, solver solves in L'*L
% thresho = (def 1e-9) cut off for change in objective function (if
%           improvement is less, stop)
% thresha = (def 1e-22) cut off for stepsize, if stepsize is
%           smaller stop
% validation = (def 0) fraction of training data to be used as
%               validation set (best output is stored in Det.bestL)
% validationstep = (def 50) every "valcount" steps do validation
% maxiter = maximum number of iterations (default: 10000)
% scale = (def. 0) if 1, all data gets re-scaled s.t. average
%         distance to closest neighbor is 1
% quiet = {0,1} surpress output (default=0)  
% notree = (default=0) set to 1 if you do not want to use metric trees
%
%
% Output:
%
% L = linear transformation xnew=L*x
%    
% Det.obj = objective function over time
% Det.nimp = number of impostors over time
% Det.pars = all parameters used in run
% Det.time = time needed for computation
% Det.iter = number of iterations
% Det.verify = verify (results of validation - if used)
%  
% Version 1.0
% copyright by Kilian Q. Weinbergerr (2005)
% University of Pennsylvania
% contact kilianw@seas.upenn.edu
%
 
% NOTE: changed initializations
if(isempty(L))
 fprintf('Initial starting point not specified.\n');
 if(rD==size(x,1)),
	fprintf('Starting with identity matrix.\n');  
	L=eye(size(x,1));
 else
	fprintf('Starting with random matrix.\n');  
	L=randn(rd,size(x,1));
 end;
end;

if (rD ~= size(L,1)),
	error('Invalid initial L, does not agree with reduced dimension rD');
end;

% checks
% NOTE: change dimension control of x
D=size(L,2);
if(size(XLXt,1)~=D), 
	error('data and L must have matching dimensions!\n');
end;

% NOTE: added dimension control of dict and creation of related variables
if(isempty(dict)),
	error('No dict provided!\n');
end;
if(size(dict,1)~=D),
	error('dict and L must have matching dimensions!\n');
end;

DDt = dict * dict';
DDt2 = DDt ^ 2;
DDt3 = DDt ^ 3;
[V Lv] = eig(DDt);
VL = V * Lv;
diagL = diag(Lv);

% set parameters
pars.stepsize=1e-07;
pars.minstepsize=0;
pars.tempid=1;
% pars.maxiter=11000;
if (~exist('Numiters', 'var')),
	pars.maxiter = 1000;
else
	pars.maxiter = Numiters;
end;
pars.factor=1.1;
pars.correction=15;
pars.thresho=1e-7;
pars.thresha=1e-22;
pars.ifraction=1;
pars.scale=0;
pars.obj=1;
pars.quiet=0;
pars.classsplit=0;
pars.validation=0;
pars.validationstep=25;
pars.earlystopping=0;
pars.valrand=1;


pars.aggressive=0;
pars.stepgrowth=1.01;
pars.weight1=ReguAlpha; % two supervised terms
pars.weight2=ReguBeta; % supervised vs unsupervised
pars.maximp=100000;
pars.maximp0=1000000;
pars.treesize=50;
pars.notree=1;

pars.targetlabels=[];

tempname=sprintf('temp%i.mat',pars.tempid);

if(~pars.quiet)
	pars %#ok
end;


% TODO: Change so that distances are computed in sparse domain
obj=zeros(1,pars.maxiter);

correction=pars.correction;
stepsize=pars.stepsize;
lastcor=1; %#ok

df=zeros(size(L));
dflap=vec(zeros(size(L)));
dforth=vec(zeros(size(L)));
correction = 1;
% Main Loop
for iter=1:pars.maxiter
 % save old position
 Lold=L;dfold=df;
if(iter>1),
% 	[foo Gvec] = eig_lsqr_obj_grad_mex_large(vec(L), XXt, YXt, trYYt, DDt2, DDt3, VL, diagL, pars.weight1, pars.weight2);
% 	[foo Gvec dflsqr dforth] = orth_lsqr_obj_grad(vec(L), X, Y, DDt, pars.weight1, pars.weight2);
	[foo Gvec objlap objorth dflap dforth] = eig_lap_obj_grad(vec(L), rD, XLXt, numSamples, DDt, DDt2, VL, Lv, pars.weight1, pars.weight2);
	G = reshape(Gvec, [rD D]);
	df = G;
	L=step(L,G,stepsize,pars);
end;

if(~pars.quiet),
	fprintf('%i.',iter);
end;

if(any(any(isnan(df))))
  fprintf('Gradient has NaN value!\n');
  keyboard;
end;

%obj(iter)=objv;
% NOTE: changed objective
% obj(iter) = eig_lsqr_obj_grad_mex_large(vec(L), XXt, YXt, trYYt, DDt2, DDt3, VL, diagL, pars.weight1, pars.weight2);
obj(iter) = eig_lap_obj_grad(vec(L), rD, XLXt, numSamples, DDt, DDt2, VL, Lv, pars.weight1, pars.weight2);

if(isnan(obj(iter)))
 fprintf('Obj is NAN!\n');
 keyboard;
end;

delta=obj(iter)-obj(max(iter-1,1));
if(~pars.quiet),
	fprintf(['Obj:%2.2f Delta:%2.4f max(G):%2.4f max(L):%2.4f max(O):%2.4f\n'],obj(iter),delta,max(max(abs(df))),max(max(abs(dflap))),max(max(abs(dforth))));
end;

if(iter>1 && delta>0 && correction~=pars.correction) 
 stepsize=stepsize*0.5;
 fprintf('***correcting stepsize***\n');
 if(stepsize<pars.minstepsize),
	 stepsize=pars.minstepsize;
 end;
 if(~pars.aggressive)
  L=Lold;
  df=dfold;
  obj(iter)=obj(iter-1);
 end;
% correction=1;
 hitwall=1; %#ok
else 
  if(correction~=pars.correction),
	  stepsize=stepsize*pars.stepgrowth;
  end;
 hitwall=0; %#ok
end;

if(iter>10)
 if (max(abs(diff(obj(iter-3:iter))))<pars.thresho*obj(iter)  || stepsize<pars.thresha)
  if(~pars.quiet),
	  fprintf('Stepsize too small. No more progress!\n');
  end;
 end;
end;

if(pars.tempid>=0 && mod(iter,50)==0),
	save(tempname,'L','iter','obj','pars'); 
end;


end;


function L=step(L,G,stepsize,pars)

% do step in gradient direction
if(size(L,1)~=size(L,2)),
	pars.obj=1;
end;
switch(pars.obj)
  case 0    % updating Q
     Q=L'*L;
     Q=Q-stepsize.*G;
   case 1   % updating L
%      G=2.*(L*G);
     L=L-stepsize.*G;     
     return;
  case 2    % multiplicative update
     Q=L'*L;
     Q=Q-stepsize.*G+stepsize^2/4.*G*inv(Q)*G; %#ok
     return;
  case 3
     Q=L'*L;
	 Q=Q-stepsize.*G;
	 Q=diag(Q);
 	 L=diag(sqrt(max(Q,0)));
     return;
  otherwise
   error('Objective function has to be 0,1,2\n');
end;

% decompose Q
[L,dd]=eig(Q);
dd=real(diag(dd));
L=real(L);
% reassemble Q (ignore negative eigenvalues)
j=find(dd<1e-10);
if(~isempty(j)) 
	if(~pars.quiet),
		fprintf('[%i]',length(j));
	end;
end;
dd(j)=0;
[temp,ii]=sort(-dd); %#ok
L=L(:,ii);
dd=dd(ii);
% Q=L*diag(dd)*L';
L=(L*diag(sqrt(dd)))';

%for i=1:size(L,1)
% if(L(i,1)~=0) L(i,:)=L(i,:)./sign(L(i,1));end;
%end;
