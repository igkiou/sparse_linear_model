function [D, Phi] = coupledmemory(X, param)
%COUPLEDMEMORY Coupled batch dictionary training.
%	[D,PHI] = COUPLEDMEMORY(X, PARAM) runs a coupled variant of the online
%	sparse dictionary training of [1], as adapted for use in batch mode. It
%	runs the algorithm on the specified set of signals X, returning the
%	simultaneously trained dictionary D and sensing matrix PHI. PARAM is a
%	struct containing settings for various parameters of the algorithm, as
%	described below.
%
%	COUPLEDMEMORY alternately optimizes D and PHI, with the other matrix
%	being considered fixed. For the optimization of PHI given a fixed D,
%	the optimization problem solved by COUPLEDMEMORY is given by
%
%      min  |L - L*Gamma'*Gamma*L'|_F^2
%     Gamma
%
%	where [V L] = eig(D) and Gamma = PHI * V. The algorithm of [2] is used
%	to solve this problem. For the optimization of D given a fixed PHI, the
%	optimization problem solved by COUPLEDMEMORY is given by
%
%		1) if param.mode=1
%	min_{D in C} (1/n) sum_{i=1}^n  ||alpha_i||_1  s.t.  ...
%	kappa*||x_i-D*alpha_i||_2^2+(1-kappa)*||y_i-PHI*D*alpha_i||_2^2 <= lambda
%		2) if param.mode=2
%	min_{D in C} (1/n) sum_{i=1}^n (1/2)(kappa*||x_i-D*alpha_i||_2^2 + ...
%	(1-kappa)*||y_i-PHI*D*alpha_i||_2^2) + lambda||alpha_i||_1
%
%	where X is a double m x n matrix containing n training signals of size
%	m, columnwise with n columns, Y = PHI*X + H, H is additive Gaussian
%	noise, and indices _i indicate the i-th column of the corresponding
%	matrix. 
%
%	C is a convex set verifying
%	C={  D in Real^{m x p}  s.t.  forall j,  ||d_j||_2^2 <= 1 }
%
%	Potentially, n can be very large with this algorithm.
%
%  Required fields in PARAM:
%  --------------------------
%
%	'lambda' - Regularization parameter, as described above.
%
%	'kappa' - Parameter balancing the reconstruction costs from the
%	original and the projected data, as described above.
%
%	'initdict' / 'dictsize' - Initial dictionary / no. of atoms to train.
%	At least one of these two should be present in PARAM.
%
%		'dictsize' specifies the number of dictionary atoms to train. If it
%		is specified without the parameter 'initdict', the dictionary is
%		initialized with dictsize randomly selected training signals.
%
%		'initdict' specifies the initial dictionary for the training. It
%		should be either a matrix of size NxL, where N=size(X,1), or an
%		index vector of length L, specifying the indices of the examples to
%		use as initial atoms. If 'dictsize' and 'initdict' are both
%		present, L must be >= dictsize, and in this case the dictionary is
%		initialized using the first dictsize columns from initdict. If only
%		'initdict' is specified, dictsize is set to L.
%
%	'initsens' / 'senssize' - Initial sensing matrix / no. of compressed
%	samples. At least one of these two should be present in PARAM.
%
%		'senssize' specifies the number of rows in the sensing matrix PHI.
%		If it is specified without the parameter 'initsens', the sensing
%		matrix is initialized as a random Gaussian matrix.
%
%		'initsens' specifies the initial sensing matrix for the training.
%		It should be a matrix of size MxN, where N=size(X,1). If
%		'senssize' and 'initsens' are both present, M must be >= senssize,
%		and in this case the dictionary is initialized using the first
%		senssize rows from sensdict. If only 'initsens' is specified,
%		senssize is set to M. 
%
%	'noisestd' specifies the standard deviation of the additive Gaussian
%	noise added to the projected data, as a percentage of the amplitude
%	of each coefficient.
%
%  Optional fields in PARAM:
%  --------------------------
%
%	'iternum' - Number of iterations. If not specified, the default is 10
%	iterations. 
%
%	'iternum2' - Number of iterations of inner gradient descent loop. If
%	not specified, the default is 10 iterations.
%	
%	'mode' - Specifies the optimization problem solved, as described above.
%	Its default value is 2.
%
%	'modeParam' - Optimization mode.
%		1) if param.modeParam=0, the optimization uses the 
%		parameter free strategy of [3].
%		2) if param.modeParam=1, the optimization uses the 
%		parameters rho as in [1].
%		3) if param.modeParam=2, the optimization uses exponential 
%		decay weights with updates of the form 
%		A_{t} <- rho A_{t-1} + alpha_t alpha_t^T.
%
%	'rho' - Tuning parameter (see [1]).
%
%	'clean' - Prune automatically the dictionary from unused elements. True
%	by default. 
%
%	'mumThreads' - Number of threads for exploiting multi-core /
%	multi-cpus. By default, it takes the value -1, which automatically
%	selects all the available CPUs/cores. 
%
%	'memusage' - This parameter controls memory usage of the function, by
%	switching between the online ('low') and memory ('high') variants of
%	the algorithm of [1]. By default, it takes the value 'low'. 'high' is
%	not supported yet.
%
%	Summary of all fields in PARAM:
%	--------------------------------
%
%	Required:
%	  'lambda'				   regularization parameter
%	  'kappa'				   term balancing quadratic costs
%     'initdict' / 'dictsize'  initial dictionary / dictionary size
%     'initsens' / 'senssize'  initial sens. matrix / sens. matrix size
%	  'noisestd'			   std of noise as percentage of amplitude
%
%	Optional (default values in parentheses):
%     'iternum'                number of training iterations (10)
%     'iternum2'               number of inner loop iterations (10)
%	  'sensmethod'			   Phi training method, 'orig'/'conj' ('conj')
%     'mode'                   optimization problem solved (2)
%     'modeParam'              optimization mode (0)
%     'rho'                    tuning parameter
%     'clean'                  prune unused dictionary elements (1)
%     'memusage'               'low, or 'high' ('low')
%
%
%  References:
%  [1] J. Mairal, F. Bach, J. Ponce, and G. Sapiro, "Online Learning for
%	   Matrix Factorization and Sparse Coding ", the Journal of Machine
%	   Learning Research, Vol. 11, pp. 19-60, 2010.
%  [2] J. Duarte-Carvajalino, and G. Sapiro, "Learning to Sense Sparse
%	   Signals: Simultaneous Sensing Matrix and Sparsifying Dictionary
%	   Optimization", the IEEE Trans. On Image Processing, Vol. 18, no. 7,
%      pp. 1395-1408, July 2009.
%  [3] J. Mairal, F. Bach, J. Ponce, and G. Sapiro, "Online Dictionary
%  Learning for Sparse Coding ", Proceedings of the 26th Annual
%  International Conference on Machine Learning, pp. 689-696, 2009. 
%
%  See also LEARN_SENSING, RANDOM_SENSING, MEXTRAINDL_MEMORY, MEXTRAINDL.

global MEM_LOW MEM_HIGH memusage

MEM_LOW = 1;
MEM_HIGH = 2;

%%%%% parse input parameters %%%%%

% regularization parameter %

if (isfield(param,'lambda'))
  lambda = param.lambda;
else
  error('Regularization parameter not specified');
end
trainparam.lambda = lambda;

% regularization parameter %

if (isfield(param,'kappa'))
  kappa = param.kappa;
else
  error('Cost balancing parameter not specified');
end

% noise std %

if (isfield(param,'noisestd'))
  noisestd = param.noisestd;
else
  error('Noise standard deviation not specified');
end

% iteration count %

if (isfield(param,'iternum'))
  iternum = param.iternum;
else
  iternum = 10;
end

% inner loop iteration count %

if (isfield(param,'iternum2'))
  iternum2 = param.iternum2;
else
  iternum2 = 10;
end

% optimization problem %

if (isfield(param,'mode'))
  mode = param.mode;
else
  mode = 2;
end
trainparam.mode = mode;

% optimization problem %

if (isfield(param,'modeParam'))
  modeParam = param.modeParam;
  trainparam.modeParam = modeParam;
end

% tuning parameter %

if (isfield(param,'rho'))
  rho = param.rho;
  trainparam.rho = rho;
end

% pruning setting %

if (isfield(param,'clean'))
  clean = param.clean;
  trainparam.clean = clean;
end

% memory usage %

if (isfield(param,'memusage'))
  switch lower(param.memusage)
    case 'low'
      memusage = MEM_LOW;
    case 'high'
      memusage = MEM_HIGH;
    otherwise
      error('Invalid memory usage mode');
  end
else
  memusage = MEM_LOW;
end

% determine dictionary size %

if (isfield(param,'initdict')),
  if (any(size(param.initdict)==1) && all(iswhole(param.initdict(:)))),
    dictsize = length(param.initdict);
  else
    dictsize = size(param.initdict,2);
  end
end
if (isfield(param,'dictsize'))    % this superceedes the size determined by initdict
  dictsize = param.dictsize;
end

if (size(X,2) < dictsize)
  error('Number of training signals is smaller than number of atoms to train');
end

% initialize the dictionary %

if (isfield(param,'initdict') && ~any(isnan(param.initdict(:)))),
  if (any(size(param.initdict)==1) && all(iswhole(param.initdict(:))))
    D = X(:,param.initdict(1:dictsize));
  else
    if (size(param.initdict,1)~=size(X,1) || size(param.initdict,2)<dictsize)
      error('Invalid initial dictionary');
    end
    D = param.initdict(:,1:dictsize);
  end
else
  X_ids = find(colnorms_squared(X) > 1e-6);   % ensure no zero data elements are chosen
  perm = randperm(length(X_ids));
  D = X(:,X_ids(perm(1:dictsize)));
end

% normalize the dictionary %

D = normcols(D);

trainparam.K = dictsize;

% determine sensing matrix size %

if (isfield(param,'initsens'))
  senssize = size(param.initsens,1);
end
if (isfield(param,'senssize'))    % this superceedes the size determined by initsens
  senssize = param.senssize;
end

% initialize the sensing matrix %

if (isfield(param,'initsens'))
  if (size(param.initsens,2)~=size(X,1) || size(param.initsens,1)<senssize)
    error('Invalid initial dictionary');
  end
  Phi = param.initsens(1:senssize,:);
else
  Phi = random_sensing(size(X, 1), senssize);
end

% determine sensing matrix training method %

if (isfield(params, 'sensmethod')),
   if (~strcmp(params.sensmethod, 'orig') && ~strcmp(params.sensmethod, 'conj')),
		error('Invalid sensmethod');
   end;
   sensmethod = params.sensmethod;
else
   sensmethod = 'conj';
end;

% set other parameters %

trainparam.iter = iternum2;
trainparam.batchsize = size(X, 2);

% create matrix to be used in main loop %

aI = kappa * eye(size(X, 1));

% other initializations %

% At = zeros(dictsize, dictsize);
% Bt = zeros((size(X, 1) + senssize), dictsize);

%%%%%%%%%%%%%%%%%  main loop  %%%%%%%%%%%%%%%%%


for iter = 1:iternum
	
	disp(sprintf('Iter %d.', iter));
  
  %%%%%%%%  perform sensing matrix optimization  %%%%%%%% 
  	disp(sprintf('Optimizing for Phi.'));
  
  if (strcmp(sensmethod, 'orig')),
	  [Phi eigObj] = learn_sensing_mex(D, senssize, Phi);
  elseif (strcmp(sensmethod, 'conj')),
	  [Phi eigObj] = learn_sensing_eig_conj_mex(D, senssize, Phi);
  end;);

  %%%%%%%%  perform dictionary optimization  %%%%%%%% 
  
  %%%%%  create equivalent data and dictionary %%%%%
  Y = Phi * X;
  Y = Y + noisestd * Y .* randn(size(Y, 1), size(Y, 2));
  Xeq = [kappa * X; (1 - kappa) * Y];
  Deq = [aI; (1 - kappa) * Phi] * D;
  trainparam.D = Deq;
  
%   %%%%%  sparse code %%%%%
%   
%   lassoparam.lambda = trainparam.lambda;
%   lassoparam.mode = trainparam.mode;
%   
%   normfactors = spdiag(1 ./ sqrt(sum(Deq .^ 2)));
%   Deqtemp = Deq * normfactors;
%   Alpha = mexLasso(Xeq, Deqtemp, lassoparam);
%   Alpha = normfactors * Alpha;
%   
%   %%%%%  run one iteration of online learning algorithm %%%%%
%   
%   At = At + Alpha * Alpha';
%   Bt = Bt + Xeq * Alpha';
% 
%   for inneriter = 1:iternum2,
% 	  for j = 1:dictsize
% 	    u_j = (Bt(:, j) - Deq * At(:, j)) / At(j, j) + Deq(:, j);
% 		Deq(:, j) = u_j / max(1, norm(u_j));
% 	  end;
%   end;
	disp(sprintf('Optimizing for codes and dictionary.'));  
  if(iter == 1)
	[Deq, model] = mexTrainDL(Xeq, trainparam);
  else
	[Deq, model] = mexTrainDL(Xeq, trainparam, model);  
  end;
	  
  %%%%%  dictionary update  %%%%%
	disp(sprintf('Dictionary update.'));    
  PhitPhi = (1 - kappa) ^ 2 * Phi' * Phi;
  cholPhi = chol(kappa ^ 2 * eye(size(PhitPhi)) + PhitPhi);
  D = (cholPhi \ (cholPhi' \ ([kappa * eye(size(Phi, 2)) (1 - kappa)* Phi'] * Deq)));
  D = normcols(D);

  %%%%%  clear dictionary  %%%%%
  
%   D = cleardict(D,Alpha,X,muthresh,unused_sigs,replaced_atoms);

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           cleardict                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [D,cleared_atoms] = cleardict(D,Alpha,X,muthresh,unused_sigs,replaced_atoms)

use_thresh = 4;  % at least this number of samples must use the atom to be kept

dictsize = size(D,2);

% compute error in blocks to conserve memory
err = zeros(1,size(X,2));
blocks = [1:3000:size(X,2) size(X,2)+1];
for i = 1:length(blocks)-1
  err(blocks(i):blocks(i+1)-1) = sum((X(:,blocks(i):blocks(i+1)-1)-D*Alpha(:,blocks(i):blocks(i+1)-1)).^2);
end

cleared_atoms = 0;
usecount = sum(abs(Alpha)>1e-7, 2);

for j = 1:dictsize
  
  % compute G(:,j)
  Gj = D'*D(:,j);
  Gj(j) = 0;
  
  % replace atom
  if ( (max(Gj.^2)>muthresh^2 || usecount(j)<use_thresh) && ~replaced_atoms(j) )
    [y,i] = max(err(unused_sigs));
    D(:,j) = X(:,unused_sigs(i)) / norm(X(:,unused_sigs(i)));
    unused_sigs = unused_sigs([1:i-1,i+1:end]);
    cleared_atoms = cleared_atoms+1;
  end
end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            misc functions            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Y = colnorms_squared(X)

% compute in blocks to conserve memory
Y = zeros(1,size(X,2));
blocksize = 2000;
for i = 1:blocksize:size(X,2)
  blockids = i : min(i+blocksize-1,size(X,2));
  Y(blockids) = sum(X(:,blockids).^2);
end

end
