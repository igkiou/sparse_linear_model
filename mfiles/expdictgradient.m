function [D, A] = expdictgradient(X, params)
%DICTGRADIENT Coupled K-SVD dictionary and sensing matrix training.
%  [D,PHI,ALPHA] = DICTGRADIENT(PARAMS) runs the coupled K-SVD training
%  algorithm on the specified set of signals, returning the simultaneously
%  trained dictionary D and sensing matrix PHI, as well as the signal
%  representation matrix ALPHA.  
%
%  DICTGRADIENT alternately optimizes D and PHI, with the other matrix being
%  considered fixed. For the optimization of PHI given a fixed D, the
%  optimization problem solved by DICTGRADIENT is given by
%
%      min  |L - L*Gamma'*Gamma*L'|_F^2
%     Gamma
%
%	where [V L] = eig(D) and Gamma = PHI * V. For the optimization of D
%	given a fixed PHI, the optimization problem solved by DICTGRADIENT is
%	given by
%
%   min alpha*|X-D*ALPHA|_F^2+|Y-PHI*D*ALPHA|_F^2  s.t.  |A_i|_0 <= T
%   D,ALPHA
%
%  where X is the set of training signals, Y = PHI*X + H, H is additive
%  Gaussian noise, A_i is the i-th column of ALPHA, T is the target
%  sparsity, and alpha is a regularization parameter. 
%
%  Required fields in PARAMS:
%  --------------------------
%
%    'codinglambda' - Sparse coding regularization parameter.
%
%    'initdict' / 'dictsize' - Initial dictionary / no. of atoms to train.
%      At least one of these two should be present in PARAMS.
%
%      'dictsize' specifies the number of dictionary atoms to train. If it
%      is specified without the parameter 'initdict', the dictionary is
%      initialized with dictsize randomly selected training signals.
%
%      'initdict' specifies the initial dictionary for the training. It
%      should be either a matrix of size NxL, where N=size(X,1), or an
%      index vector of length L, specifying the indices of the examples to
%      use as initial atoms. If 'dictsize' and 'initdict' are both present,
%      L must be >= dictsize, and in this case the dictionary is
%      initialized using the first dictsize columns from initdict. If only
%      'initdict' is specified, dictsize is set to L.
%
%    'initsens' / 'senssize' - Initial sensing matrix / no. of compressed
%      samples. At least one of these two should be present in PARAMS.
%
%      'senssize' specifies the number of rows in the sensing matrix PHI.
%      If it is specified without the parameter 'initsens', the sensing
%      matrix is initialized as a random Gaussian matrix.
%
%      'initsens' specifies the initial sensing matrix for the training. It
%      should be a matrix of size MxN, where N=size(X,1). If 'senssize'
%      and 'initsens' are both present, M must be >= senssize, and in this
%      case the dictionary is initialized using the first senssize rows
%      from sensdict. If only 'initsens' is specified, senssize is set to
%      M. 
%
%	  'noisestd' specifies the standard deviation of the additive Gaussian
%	  noise added to the projected X, as a percentage of the amplitude
%	  of each coefficient.
%
%	  'alpha' specifies the trade-off between the two types of
%	  reconstruction error.
%
%
%  Optional fields in PARAMS:
%  --------------------------
%
%    'iternum' - Number of training iterations.
%      Specifies the number of iterations to perform. If not
%      specified, the default is 10.
%
%
%  Optional fields in PARAMS - advanced:
%  -------------------------------------
%
%    'muthresh' - Mutual incoherence threshold.
%      This parameter can be used to control the mutual incoherence of the
%      trained dictionary, and is typically between 0.9 and 1. At the end
%      of each iteration, the trained dictionary is "cleaned" by discarding
%      atoms with correlation > muthresh. The default value for muthresh is
%      0.99. Specifying a value of 1 or higher cancels this type of
%      cleaning completely. Note: the trained dictionary is not guaranteed
%      to have a mutual incoherence less than muthresh. However, a method
%      to track this is using the VERBOSE parameter to print the number of
%      replaced atoms each iteration; when this number drops near zero, it
%      is more likely that the mutual incoherence of the dictionary is
%      below muthresh.
%
%
%   Summary of all fields in PARAMS:
%   --------------------------------
%
%   Required:
%     'codinglambda'		   sparse coding regularization
%     'initdict' / 'dictsize'  initial dictionary / dictionary size
%
%   Optional (default values in parentheses):
%     'iternum'                number of training iterations (10)
%	  'blockratio'			   portion of samples in block (0)
%     'muthresh'               mutual incoherence threshold (0.99)
%	  'printinfo'			   print progress messages (0)
%	  'errorinfo'			   print objective function value progress (0)
%
%
%  References:
%  [1] M. Aharon, M. Elad, and A.M. Bruckstein, "The K-SVD: An Algorithm
%      for Designing of Overcomplete Dictionaries for Sparse
%      Representation", the IEEE Trans. On Signal Processing, Vol. 54, no.
%      11, pp. 4311-4322, November 2006.
%  [2] M. Elad, R. Rubinstein, and M. Zibulevsky, "Efficient Implementation
%      of the K-SVD Algorithm using Batch Orthogonal Matching Pursuit",
%      Technical Report - CS, Technion, April 2008.
%  [3] J. Duarte-Carvajalino, and G. Sapiro, "Learning to Sense Sparse
%	   Signals: Simultaneous Sensing Matrix and Sparsifying Dictionary
%	   Optimization", the IEEE Trans. On Image Processing, Vol. 18, no. 7,
%      pp. 1395-1408, July 2009.
%
%  See also LEARN_SENSING, RANDOM_SENSING, OMP, OMP2.

%%%%% parse input parameters %%%%%

% exponential family %

if (isfield(params,'expFamily')),
  family = params.expFamily;
else
  family = 'P';
end;

% coding lambda %

if (isfield(params,'codinglambda')),
  codinglambda = params.codinglambda;
else
  error('Sparse coding regularization parameter not specified'); %#ok
end;

% iteration count %

if (isfield(params,'iternum')),
  iternum = params.iternum;
else
  iternum = 10;
end;

% block ratio %

if (isfield(params,'blockratio')),
  blockratio = params.blockratio;
else
  blockratio = 0;
end;

% save path %

if (isfield(params,'savepath')),
	savepath = params.savepath;
else
	savepath = NaN;
end;

% status messages %

if (isfield(params,'printinfo')),
  printinfo = params.printinfo;
else
  printinfo = 0;
end;

if (isfield(params,'errorinfo')),
  errorinfo = params.errorinfo;
else
  errorinfo = 0;
end;

% dictionary clearing %

if (isfield(params,'dictclear')),
  dictclear = params.dictclear;
else
  dictclear = 1;
end;

% mutual incoherence limit %

if (isfield(params,'muthresh')),
  muthresh = params.muthresh;
else
  muthresh = 0.99;
end;

if (muthresh < 0),
  error('invalid muthresh value, must be non-negative'); %#ok
end;

% use count limit %

if (isfield(params,'usedthresh')),
  usedthresh = params.usedthresh;
else
  usedthresh = 0;
end;

if (usedthresh < 0),
  error('invalid usedthresh value, must be non-negative'); %#ok
end;

% determine dictionary size %

if (isfield(params,'initdict')),
  if (any(size(params.initdict)==1) && all(iswhole(params.initdict(:)))),
    dictsize = length(params.initdict);
  else
    dictsize = size(params.initdict,2);
  end;
end;

if (isfield(params,'dictsize')),    % this superceedes the size determined by initdict
  dictsize = params.dictsize;
end;

if (size(X, 2) < dictsize),
  warning('Number of training signals is smaller than number of atoms to train'); %#ok
end;

% initialize the dictionary %

% X_ids = find(~any(abs(X) < 1e-6));   % ensure no X elements with zero entries are chosen
X_ids = 1:size(X, 2);
total_ids = length(X_ids);
perm_ids = randperm(total_ids);
pos_mean = mean(X(X~=0));

if (isfield(params,'initdict') && ~any(isnan(params.initdict(:)))),
  if (any(size(params.initdict)==1) && all(iswhole(params.initdict(:)))),
	[foo D] = link_func_dual(X(:,params.initdict(1:dictsize)), family);
  else
    if (size(params.initdict,1)~=size(X,1) || size(params.initdict,2)<dictsize),
      error('Invalid initial dictionary'); %#ok
    end;
    D = params.initdict(:,1:dictsize);
  end;
  used_ids = 0;
else
  [foo D] = link_func_dual(X(:,X_ids(perm_ids(1:dictsize))), family);
  while (any(isinf(D(:)))),
	  inf_ids = find(isinf(D(:)));
	  tempVals = poissrnd(pos_mean, size(inf_ids));
	  [foo, D(inf_ids)] = link_func_dual(tempVals, family);
  end;
  used_ids = dictsize;
end;

% normalize the dictionary %

D = normcols(D);

% precalculate useful quantities %

[signalsize numsamples] = size(X); %#ok
if ((blockratio > 0) && (blockratio < 1)),
  blocksamples = ceil(blockratio * numsamples);
else
  blocksamples = numsamples;
end;

%%%%%%%%%%%%%%%%%  main loop  %%%%%%%%%%%%%%%%%

if (printinfo),
  disp(sprintf('Starting learning exponential sparsifying dictionary with dictclear=%d.\n', dictclear)); %#ok
end;


for iter = 1:iternum
	
  if (printinfo),
	  disp(sprintf('Iteration %d / %d begins\n', iter, iternum)); %#ok
  end;
 
  %%%%%%%%  perform dictionary optimization  %%%%%%%% 
  
  %%%%%  block creation %%%%%
  
  if (printinfo),
	  disp(sprintf('Creating blocks\n')); %#ok
  end;
  
  if ((blockratio > 0) && (blockratio < 1)),
	  inds = randperm(numsamples);
	  blockinds = inds(1:blocksamples);
	  Xblock = X(:, blockinds);
  else
	  Xblock = X;
  end;
  
  %%%%%  sparse coding  %%%%%
  
  if (printinfo),
	  disp(sprintf('Sparse coding\n')); %#ok
  end;
  
  A = l1exp_featuresign_mex(Xblock, D, codinglambda);
  
  %%%%%  dictionary update  %%%%%
  
  if (printinfo),
	  disp(sprintf('Updating dictionary\n')); %#ok
  end;
  
%   D = l2exp_learn_basis_gradient_projection_backtracking(X, A, D, family);
  D = l2exp_learn_basis_gradient_projection_backtracking_mex(Xblock, A, D, family);
  
  %%%%%  compute error  %%%%%
  
  if (errorinfo)
	basisObj = basis_exp_obj_grad(D, Xblock, A, family);
	totalObj = basisObj + codinglambda * sum(abs(A(:)));
	[foo XblockRecon] = link_func(D * A, family);
	reconObj = norm(Xblock - XblockRecon, 'fro') ^ 2;
  end;
   
  %%%%%  clear dictionary  %%%%%
  
  if ((dictclear == 1) && (mod(iter, 10) == 0)),
	  if (printinfo),
		  disp(sprintf('Clearing dictionary\n')); %#ok
	  end;
	  
	  if ((blockratio > 0) && (blockratio < 1)),
		  A = l1exp_featuresign_mex(X, D, codinglambda);
	  end;
	  
	  used_ids_old = used_ids;
	  [D perm_ids used_ids] = cleardict_custom(D, X, A, family, muthresh, usedthresh, X_ids, perm_ids, used_ids, pos_mean);
	  
	  if (printinfo),
		  disp(sprintf('Finished clearing dictionary, replaced atoms %d / %d\n', used_ids - used_ids_old, dictsize)); %#ok
	  end;
  end;
  
  %%%%%%%%  save intermediate results  %%%%%%%% 	
  
  if (printinfo),
	  disp(sprintf('Saving intermediate results\n')); %#ok
  end;
  
  if (~any(isnan(savepath(:)))),
	save(sprintf('%s_iter%d.mat', savepath, iter), 'D');
  end;
  
  %%%%%  print info  %%%%%
  
  if (printinfo),
	  if (errorinfo),
		  disp(sprintf('basisObj = %g, totalObj = %g, reconObj = %g\n', basisObj, totalObj, reconObj)); %#ok
	  end;
	  disp(sprintf('Iteration %d / %d complete\n', iter, iternum)); %#ok
  end;
  
end;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           cleardict                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [D perm_ids used_ids] = cleardict_custom(D, X, A, family, muthresh, usedthresh, X_ids, perm_ids, used_ids, pos_mean)

usecount = sum(abs(A) > 1e-7, 2);
dictsize = size(D, 2);
for j = 1:dictsize
  
  % compute G(:,j)
%   Gj = D' * D(:, j);
%   Gj(j) = 0;
  
  % replace atom
%   if ((max(Gj .^ 2) > muthresh^2 || usecount(j) <= usedthresh)),
  if (usecount(j) <= usedthresh),
	used_ids = used_ids + 1;
	[foo D(:, j)] = link_func_dual(X(:, X_ids(perm_ids(used_ids))), family);
	while (any(isinf(D(:, j)))),
	  inf_ids = find(isinf(D(:, j)));
	  tempVals = poissrnd(pos_mean, size(inf_ids));
	  [foo, D(inf_ids, j)] = link_func_dual(tempVals, family);
	end;
	D(:, j) = D(:, j) / norm(D(:, j));
  end;
end;

end
