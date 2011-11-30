function [D, Phi, A] = coupledksvd(X, params)
%COUPLEDKSVD Coupled K-SVD dictionary and sensing matrix training.
%  [D,PHI,ALPHA] = COUPLEDKSVD(PARAMS) runs the coupled K-SVD training
%  algorithm on the specified set of signals, returning the simultaneously
%  trained dictionary D and sensing matrix PHI, as well as the signal
%  representation matrix ALPHA.  
%
%  COUPLEDKSVD alternately optimizes D and PHI, with the other matrix being
%  considered fixed. For the optimization of PHI given a fixed D, the
%  optimization problem solved by COUPLEDKSVD is given by
%
%      min  |L - L*Gamma'*Gamma*L'|_F^2
%     Gamma
%
%	where [V L] = eig(D) and Gamma = PHI * V. For the optimization of D
%	given a fixed PHI, the optimization problem solved by COUPLEDKSVD is
%	given by
%
%   min alpha*|X-D*ALPHA|_F^2+|Y-PHI*D*ALPHA|_F^2  s.t.  |A_i|_0 <= T
%   D,ALPHA
%
%  where X is the set of training signals, Y = PHI*X + H, H is additive
%  Gaussian noise, A_i is the i-th column of ALPHA, T is the target
%  sparsity, and alpha is a regularization parameter. 
%
%
%  Required fields in PARAMS:
%  --------------------------
%
%    'Tdata' - Sparse coding target.
%      Specifies the number of coefficients (Tdata) for coding each signal.
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
%      Specifies the number of iterations to perform. If not specified, the
%      default is 10. 
%
%    'memusage' - Memory usage.
%      This parameter controls memory usage of the function. 'memusage'
%      should be one of the strings 'high', 'normal' (default) or 'low'.
%      When set to 'high', the fastest implementation of OMP is used, which
%      involves precomputing both G=D'*D and DtX=D'*X. This increases
%      speed but also requires a significant amount of memory. When set to
%      'normal', only the matrix G is precomputed, which requires much less
%      memory but slightly decreases performance. Finally, when set to
%      'low', neither matrix is precomputed. This should only be used when
%      the trained dictionary is highly redundant and memory resources are
%      very low, as this will dramatically increase runtime. See function
%      OMP for more details.
%
%    'exact' - Exact K-SVD update.
%      Specifies whether the exact or approximate dictionary update
%      should be used. By default, the approximate computation is used,
%      which is significantly faster and requires less memory. Specifying a
%      nonzero value for 'exact' causes the exact computation to be used
%      instead, which slows down the method but provides slightly improved
%      results. The exact update uses SVD to solve the rank-1 minimization
%      problem, while the approximate upate performs alternate-optimization
%      to solve this problem.
%
%
%  Optional fields in PARAMS - advanced:
%  -------------------------------------
%
%    'muthresh' - Mutual incoherence threshold.
%      This parameter can be used to control the mutual incoherence of the
%      trained dictionary, and is typically between 0.9 and 1. At the end;
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
%     'Tdata'		           sparse-coding target
%     'initdict' / 'dictsize'  initial dictionary / dictionary size
%     'initsens' / 'senssize'  initial sens. matrix / sens. matrix size
%	  'noisestd'			   std of noise as percentage of amplitude
%	  'alpha'				   error balancing parameter
%
%   Optional (default values in parentheses):
%     'iternum'                number of training iterations (10)
%	  'sensmethod'			   Phi training method, 'orig'/'conj' ('orig')
%     'memusage'               'low, 'normal' or 'high' ('normal')
%     'exact'                  exact update instead of approximate (0)
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


global MEM_LOW MEM_NORMAL MEM_HIGH memusage
global ompparams exactsvd

MEM_LOW = 1;
MEM_NORMAL = 2;
MEM_HIGH = 3;


%%%%% parse input parameters %%%%%

ompparams = {'checkdict','on'};

% coding mode %

if (isfield(params,'Tdata'))
  thresh = params.Tdata;
else
  error('Data sparse-coding target not specified');
end;

% noise std %

if (isfield(params,'noisestd'))
  noisestd = params.noisestd;
else
  error('Noise standard deviation not specified');
end;

% error balancing parameter %

if (isfield(params,'alpha'))
  alpha = params.alpha;
else
  error('Error balancing parameter not specified');
end;

% memory usage %

if (isfield(params,'memusage'))
  switch lower(params.memusage)
    case 'low'
      memusage = MEM_LOW;
    case 'normal'
      memusage = MEM_NORMAL;
    case 'high'
      memusage = MEM_HIGH;
    otherwise
      error('Invalid memory usage mode');
  end;
else
  memusage = MEM_NORMAL;
end;

% iteration count %

if (isfield(params,'iternum'))
  iternum = params.iternum;
else
  iternum = 10;
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

% mutual incoherence limit %

if (isfield(params,'muthresh'))
  muthresh = params.muthresh;
else
  muthresh = 0.99;
end;
if (muthresh < 0)
  error('invalid muthresh value, must be non-negative');
end;

% exact svd computation %

exactsvd = 0;
if (isfield(params,'exact') && params.exact~=0)
  exactsvd = 1;
end;

% determine dictionary size %

if (isfield(params,'initdict')),
  if (any(size(params.initdict)==1) && all(iswhole(params.initdict(:)))),
    dictsize = length(params.initdict);
  else
    dictsize = size(params.initdict,2);
  end;
end;
if (isfield(params,'dictsize'))    % this superceedes the size determined by initdict
  dictsize = params.dictsize;
end;

if (size(X,2) < dictsize)
  error('Number of training signals is smaller than number of atoms to train');
end;

% initialize the dictionary %

if (isfield(params,'initdict') && ~any(isnan(params.initdict(:)))),
  if (any(size(params.initdict)==1) && all(iswhole(params.initdict(:))))
    D = X(:,params.initdict(1:dictsize));
  else
    if (size(params.initdict,1)~=size(X,1) || size(params.initdict,2)<dictsize)
      error('Invalid initial dictionary');
    end;
    D = params.initdict(:,1:dictsize);
  end;
else
  X_ids = find(colnorms_squared(X) > 1e-6);   % ensure no zero X elements are chosen
  perm = randperm(length(X_ids));
  D = X(:,X_ids(perm(1:dictsize)));
end;

% normalize the dictionary %

D = normcols(D);

% determine sensing matrix size %

if (isfield(params,'initsens'))
  senssize = size(params.initsens,1);
end;
if (isfield(params,'senssize'))    % this superceedes the size determined by initsens
  senssize = params.senssize;
end;

% initialize the sensing matrix %

if (isfield(params,'initsens'))
  if (size(params.initsens,2)~=size(X,1) || size(params.initsens,1)<senssize)
    error('Invalid initial sensing matrix');
  end;
  Phi = params.initsens(1:senssize,:);
else
  Phi = random_sensing(size(X, 1), senssize);
end;

% determine sensing matrix training method %

if (isfield(params, 'sensmethod')),
   if (~strcmp(params.sensmethod, 'orig') && ~strcmp(params.sensmethod, 'conj')),
		error('Invalid sensmethod');
   end;
   sensmethod = params.sensmethod;
else
   sensmethod = 'orig';
end;

% create matrix to be used in main loop %

[signalsize numsamples] = size(X);
aI = alpha * eye(size(X, 1));

%%%%%%%%%%%%%%%%%  main loop  %%%%%%%%%%%%%%%%%


for iter = 1:iternum
  
  if (printinfo),
	  disp(sprintf('Iteration %d / %d begins\n', iter, iternum)); %#ok
  end;
  
  %%%%%%%%  perform sensing matrix optimization  %%%%%%%% 
  
  if (printinfo),
	  disp(sprintf('Optimizing Phi\n')); %#ok
  end;
  
  if (strcmp(sensmethod, 'orig')),
	  [Phi eigObj] = learn_sensing_exact(D, senssize, Phi);
  elseif (strcmp(sensmethod, 'conj')),
	  [Phi eigObj] = learn_sensing_eig_conj_mex(D, senssize, Phi);
  end;

  %%%%%  precalculate quantities used in dictioanry optim %%%%%
  PhitPhi = Phi' * Phi;
  cholPhi = chol(alpha ^ 2 * eye(signalsize) + PhitPhi);
  extPhi = [aI; Phi];
  
  %%%%%%%%  perform dictionary optimization  %%%%%%%% 
  
  %%%%%  create equivalent X and dictionary %%%%%
  
  if (printinfo),
	  disp(sprintf('Creating equivalent X\n')); %#ok
  end;
  
  Y = Phi * X;
  Y = Y + noisestd * Y .* randn(senssize, numsamples);
  Xeq = [alpha * X; Y];
%   Deq = [aI; Phi] * D;
  Deq = extPhi * D;
  
  %%%%%  sparse coding  %%%%%
  
  if (printinfo),
	  disp(sprintf('Sparse coding\n')); %#ok
  end;
  
  normfactors = spdiag(1 ./ sqrt(sum(Deq .^ 2)));
  A = sparsecode(Xeq, Deq * normfactors, thresh);
  A = normfactors * A;
  
  %%%%%  dictionary update  %%%%%
  
  if (printinfo),
	  disp(sprintf('Updating dictionary\n')); %#ok
  end;
  
  replaced_atoms = zeros(1,dictsize);  % mark each atom replaced by optimize_atom
  
  unused_sigs = 1:numsamples;  % tracks the signals that were used to replace "dead" atoms.
                                 % makes sure the same signal is not selected twice
  
  p = randperm(dictsize);
  for j = 1:dictsize
    [D(:,p(j)),gamma_j,X_indices,unused_sigs,replaced_atoms] = ...
		coupled_optimize_atom(Xeq,Deq,extPhi,cholPhi,p(j),A,unused_sigs,replaced_atoms);
    A(p(j),X_indices) = gamma_j;
% 	Deq(:,p(j)) = [aI; Phi] * D(:,p(j));
	Deq(:,p(j)) = extPhi * D(:,p(j));
  end;
  
  %%%%%  compute error  %%%%%
  
  if (errorinfo)
    reconObj = compute_err(D, Phi, A, X, Y, alpha);
  end;
   
  %%%%%  clear dictionary  %%%%%
  
  if (printinfo),
	  disp(sprintf('Clearing dictionary\n')); %#ok
  end;
  
  D = coupled_cleardict(D, A, X, Y, alpha, muthresh, unused_sigs, replaced_atoms);
  
  %%%%%  print info  %%%%%
  
  if (printinfo),
	  if (errorinfo),
		  disp(sprintf('eigObj = %g, reconObj = %g\n', eigObj, reconObj)); %#ok
	  end;
	  disp(sprintf('Iteration %d / %d complete\n', iter, iternum)); %#ok
  end;
  
end;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%        coupled_optimize_atom         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [atom,gamma_j,X_indices,unused_sigs,replaced_atoms] = coupled_optimize_atom(X,D,extPhi,cholPhi,j,A,unused_sigs,replaced_atoms)

global exactsvd

% X samples which use the atom, and the corresponding nonzero
% coefficients in A
[gamma_j, X_indices] = sprow(A, j);

if (length(X_indices) < 1)
	maxsignals = 5000;
	perm = randperm(length(unused_sigs));
	perm = perm(1:min(maxsignals,end));
	E = sum((X(:,unused_sigs(perm)) - D*A(:,unused_sigs(perm))).^2);
	[d, i] = max(E);
	atom = X(:,unused_sigs(perm(i)));
	atom = atom./norm(atom);
	gamma_j = zeros(size(gamma_j));
	unused_sigs = unused_sigs([1:perm(i)-1,perm(i)+1:end;]);
	replaced_atoms(j) = 1;
else
	smallA = A(:,X_indices);
	Dj = D(:,j);
	if (exactsvd),
		[atom,s,gamma_j] = svds(X(:,X_indices) - D*smallA + Dj*gamma_j, 1);
		gamma_j = s*gamma_j;
	else
		atom = collincomb(X,X_indices,gamma_j') - D*(smallA*gamma_j') + Dj*(gamma_j*gamma_j');
		atom = atom/norm(atom);
		gamma_j = rowlincomb(atom,X,1:size(X,1),X_indices) - (atom'*D)*smallA + (atom'*Dj)*gamma_j;
	end;
end;

% atom = (cholPhi \ (cholPhi' \ ([alpha * eye(size(Phi, 2)) Phi'] * atom)));
atom = (cholPhi \ (cholPhi' \ (extPhi' * atom)));
atomnorm = norm(atom);
if (atomnorm > eps)
	atom = atom / atomnorm;
	gamma_j = atomnorm * gamma_j;
end;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             sparsecode               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = sparsecode(X,D,thresh)

global MEM_HIGH MEM_NORMAL memusage
global ompparams

% normfactors = spdiag(1 ./ sqrt(sum(D .^ 2)));
% D = D * normfactors;

  %%%%%  precompute G %%%%%
  
  G = [];
  if (memusage >= MEM_NORMAL)
    G = D'*D;
  end;
  
  %%%%%  sparse coding  %%%%%

if (memusage < MEM_HIGH),
  A = omp(D,X,G,thresh,ompparams{:});
else
  A = omp(D'*X,G,thresh,ompparams{:});  
end;

% A = normfactors * A;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             compute_err              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function err = compute_err(D, Phi, A, X, Y, alpha)
  
mat1 = X - D * A;
mat2 = Y - Phi * D * A;
err = (alpha * norm(mat1, 'fro') ^ 2 + norm(mat2, 'fro') ^ 2);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          coupled_cleardict           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function D = coupled_cleardict(D, Phi, A, X, Y, alpha, muthresh, unused_sigs, replaced_atoms)

use_thresh = 4;  % at least this number of samples must use the atom to be kept

dictsize = size(D,2);

% compute error in blocks to conserve memory
err = zeros(1, size(X, 2));
blocks = [1:3000:size(X, 2) size(X, 2) + 1];
for i = 1:length(blocks) - 1
  err(blocks(i):blocks(i + 1) - 1) = ...
	  alpha * sum((X(:, blocks(i):blocks(i + 1) - 1) - D * A(:, blocks(i):blocks(i + 1) - 1)) .^ 2)...
	  + sum((Y(:, blocks(i):blocks(i + 1) - 1) - Phi * D * A(:, blocks(i):blocks(i + 1) - 1)) .^ 2);
end;

usecount = sum(abs(A) > 1e-7, 2);

for j = 1:dictsize
  
  % compute G(:,j)
  Gj = D' * D(:, j);
  Gj(j) = 0;
  
  % replace atom
  if ((max(Gj .^ 2) > muthresh^2 || usecount(j) < use_thresh) && ~replaced_atoms(j))
    [y, i] = max(err(unused_sigs));
    D(:, j) = X(:, unused_sigs(i)) / norm(X(:, unused_sigs(i)));
    unused_sigs = unused_sigs([1:i - 1, i + 1:end;]);
  end;
end;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%           cleardict                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function D = cleardict(D, A, X, muthresh, unused_sigs, replaced_atoms)

use_thresh = 4;  % at least this number of samples must use the atom to be kept

dictsize = size(D,2);

% compute error in blocks to conserve memory
err = zeros(1, size(X, 2));
blocks = [1:3000:size(X, 2) size(X, 2) + 1];
for i = 1:length(blocks) - 1
  err(blocks(i):blocks(i + 1) - 1) = sum((X(:, blocks(i):blocks(i + 1) - 1) - D * A(:, blocks(i):blocks(i + 1) - 1)) .^ 2);
end

usecount = sum(abs(A) > 1e-7, 2);

for j = 1:dictsize
  
  % compute G(:,j)
  Gj = D' * D(:, j);
  Gj(j) = 0;
  
  % replace atom
  if ((max(Gj .^ 2) > muthresh^2 || usecount(j) < use_thresh) && ~replaced_atoms(j))
    [y, i] = max(err(unused_sigs));
    D(:, j) = X(:, unused_sigs(i)) / norm(X(:, unused_sigs(i)));
    unused_sigs = unused_sigs([1:i - 1, i + 1:end]);
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
end;

end
