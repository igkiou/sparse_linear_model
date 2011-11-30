function [D, A] = ksvd(X, params)
%KSVD K-SVD dictionary training.
%  [D,GAMMA] = KSVD(PARAMS) runs the K-SVD dictionary training algorithm on
%  the specified set of signals, returning the trained dictionary D and the
%  signal representation matrix GAMMA.
%
%  KSVD has two modes of operation: sparsity-based and error-based. For
%  sparsity-based minimization, the optimization problem is given by
%
%      min  |X-D*GAMMA|_F^2      s.t.  |A_i|_0 <= T
%    D,A
%
%  where X is the set of training signals, A_i is the i-th column of
%  A, and T is the target sparsity. For error-based minimization, the
%  optimization problem is given by
%
%      min  |A|_0      s.t.  |X_i - D*A_i|_2 <= EPSILON
%    D,A
%
%  where X_i is the i-th training signal, and EPSILON is the target error.
%
%
%  Required fields in PARAMS:
%  --------------------------
%
%    'Tdata' / 'Edata' - Sparse coding target.
%      Specifies the number of coefficients (Tdata) or the target error in
%      L2-norm (Edata) for coding each signal. If only one is present, that
%      value is used. If both are present, Tdata is used, unless the field
%      'codemode' is specified (below).
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
%
%  Optional fields in PARAMS:
%  --------------------------
%
%    'iternum' - Number of training iterations.
%      Specifies the number of K-SVD iterations to perform. If not
%      specified, the default is 10.
%
%    'memusage' - Memory usage.
%      This parameter controls memory usage of the function. 'memusage'
%      should be one of the strings 'high', 'normal' (default) or 'low'.
%      When set to 'high', the fastest implementation of OMP is used, which
%      involves precomputing both G=D'*D and DtX=D'*X. This increasese
%      speed but also requires a significant amount of memory. When set to
%      'normal', only the matrix G is precomputed, which requires much less
%      memory but slightly decreases performance. Finally, when set to
%      'low', neither matrix is precomputed. This should only be used when
%      the trained dictionary is highly redundant and memory resources are
%      very low, as this will dramatically increase runtime. See function
%      OMP for more details.
%
%    'codemode' - Sparse-coding target mode.
%      Specifies whether the 'Tdata' or 'Edata' fields should be used for
%      the sparse-coding stopping criterion. This is useful when both
%      fields are present in PARAMS. 'codemode' should be one of the
%      strings 'sparsity' or 'error'. If it is not present, and both fields
%      are specified, sparsity-based coding takes place.
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
%    'maxatoms' - Maximal number of atoms in signal representation.
%      When error-based sparse coding is used, this parameter can be used
%      to specify a hard limit on the number of atoms in each signal
%      representation (see parameter 'maxatoms' in OMP2 for more details).
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
%     'Tdata' / 'Edata'        sparse-coding target
%     'initdict' / 'dictsize'  initial dictionary / dictionary size
%
%   Optional (default values in parentheses):
%     'testX'               validation X (none)
%     'iternum'                number of training iterations (10)
%     'memusage'               'low, 'normal' or 'high' ('normal')
%     'codemode'               'sparsity' or 'error' ('sparsity')
%     'exact'                  exact update instead of approximate (0)
%     'maxatoms'               max # of atoms in error sparse-coding (none)
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
%
%  See also KSVDDENOISE, OMPDENOISE, OMP, OMP2.


%  Ron Rubinstein
%  Computer Science Department
%  Technion, Haifa 32000 Israel
%  ronrubin@cs
%
%  August 2009


global CODE_SPARSITY CODE_ERROR codemode
global MEM_LOW MEM_NORMAL MEM_HIGH memusage
global ompparams exactsvd
% global ompfunc

CODE_SPARSITY = 1;
CODE_ERROR = 2;

MEM_LOW = 1;
MEM_NORMAL = 2;
MEM_HIGH = 3;


%%%%% parse input parameters %%%%%

ompparams = {'checkdict','on'};

% coding mode %

if (isfield(params,'codemode'))
  switch lower(params.codemode)
    case 'sparsity'
      codemode = CODE_SPARSITY;
      thresh = params.Tdata;
    case 'error'
      codemode = CODE_ERROR;
      thresh = params.Edata;
    otherwise
      error('Invalid coding mode specified');
  end
elseif (isfield(params,'Tdata'))
  codemode = CODE_SPARSITY;
  thresh = params.Tdata;
elseif (isfield(params,'Edata'))
  codemode = CODE_ERROR;
  thresh = params.Edata;

else
  error('Data sparse-coding target not specified');
end


% max number of atoms %

if (codemode==CODE_ERROR && isfield(params,'maxatoms'))
  ompparams{end+1} = 'maxatoms';
  ompparams{end+1} = params.maxatoms;
end


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
  end
else
  memusage = MEM_NORMAL;
end


% iteration count %

if (isfield(params,'iternum'))
  iternum = params.iternum;
else
  iternum = 10;
end


% omp function %

% if (codemode == CODE_SPARSITY)
%   ompfunc = @omp;
% else
%   ompfunc = @omp2;
% end


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

% X norms %

XtX = [];
if (codemode==CODE_ERROR && memusage==MEM_HIGH)
  XtX = colnorms_squared(X);
end


% mutual incoherence limit %

if (isfield(params,'muthresh'))
  muthresh = params.muthresh;
else
  muthresh = 0.99;
end
if (muthresh < 0)
  error('invalid muthresh value, must be non-negative');
end


% exact svd computation %

exactsvd = 0;
if (isfield(params,'exact') && params.exact~=0)
  exactsvd = 1;
end


% determine dictionary size %

if (isfield(params,'initdict'))
  if (any(size(params.initdict)==1) && all(iswhole(params.initdict(:))))
    dictsize = length(params.initdict);
  else
    dictsize = size(params.initdict,2);
  end
end
if (isfield(params,'dictsize'))    % this superceedes the size determined by initdict
  dictsize = params.dictsize;
end

if (size(X,2) < dictsize)
  error('Number of training signals is smaller than number of atoms to train');
end


% initialize the dictionary %

if (isfield(params,'initdict') && ~any(isnan(params.initdict(:)))),
  if (any(size(params.initdict)==1) && all(iswhole(params.initdict(:)))),
    D = X(:,params.initdict(1:dictsize));
  else
    if (size(params.initdict,1)~=size(X,1) || size(params.initdict,2)<dictsize),
      error('Invalid initial dictionary');
    end
    D = params.initdict(:,1:dictsize);
  end
else
  X_ids = find(colnorms_squared(X) > 1e-6);   % ensure no zero X elements are chosen
  perm = randperm(length(X_ids));
  D = X(:,X_ids(perm(1:dictsize)));
end


% normalize the dictionary %

D = normcols(D);

%%%%%%%%%%%%%%%%%  main loop  %%%%%%%%%%%%%%%%%


for iter = 1:iternum
  
  if (printinfo),
	  disp(sprintf('Iteration %d / %d begins\n', iter, iternum)); %#ok
  end;
	
  G = [];
  if (memusage >= MEM_NORMAL)
    G = D'*D;
  end
  
  %%%%%  sparse coding  %%%%%
  
  if (printinfo),
	  disp(sprintf('Sparse coding\n')); %#ok
  end;
  
  A = sparsecode(X,D,XtX,G,thresh);
  
  %%%%%  dictionary update  %%%%%
  
  if (printinfo),
	  disp(sprintf('Updating dictionary\n')); %#ok
  end;
  
  replaced_atoms = zeros(1,dictsize);  % mark each atom replaced by optimize_atom
  
  unused_sigs = 1:size(X,2);  % tracks the signals that were used to replace "dead" atoms.
                                 % makes sure the same signal is not selected twice
  
  p = randperm(dictsize);
  for j = 1:dictsize
    [D(:,p(j)),gamma_j,X_indices,unused_sigs,replaced_atoms] = optimize_atom(X,D,p(j),A,unused_sigs,replaced_atoms);
    A(p(j),X_indices) = gamma_j;
  end
  
  %%%%%  compute error  %%%%%
  
  if (errorinfo)
    reconObj = compute_err(D, Phi, Alpha, X, Y, kappa);
  end; 
  
  %%%%%  clear dictionary  %%%%%
  
  if (printinfo),
	  disp(sprintf('Clearing dictionary\n')); %#ok
  end;
  
  D = cleardict(D,A,X,muthresh,unused_sigs,replaced_atoms);
  
  %%%%%  print info  %%%%%
  
  if (printinfo),
	  if (errorinfo),
		  disp(sprintf('reconObj = %g\n', reconObj)); %#ok
	  end;
	  disp(sprintf('Iteration %d / %d complete\n', iter, iternum)); %#ok
  end;
  
end;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%            optimize_atom             %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [atom,gamma_j,X_indices,unused_sigs,replaced_atoms] = optimize_atom(X,D,j,A,unused_sigs,replaced_atoms)

global exactsvd

% X samples which use the atom, and the corresponding nonzero
% coefficients in A
[gamma_j, X_indices] = sprow(A, j);

if (length(X_indices) < 1)
  maxsignals = 5000;
  perm = randperm(length(unused_sigs));
  perm = perm(1:min(maxsignals,end));
  E = sum((X(:,unused_sigs(perm)) - D*A(:,unused_sigs(perm))).^2);
  [d,i] = max(E);
  atom = X(:,unused_sigs(perm(i)));
  atom = atom./norm(atom);
  gamma_j = zeros(size(gamma_j));
  unused_sigs = unused_sigs([1:perm(i)-1,perm(i)+1:end]);
  replaced_atoms(j) = 1;
  return;
end

smallA = A(:,X_indices);
Dj = D(:,j);

if (exactsvd)

  [atom,s,gamma_j] = svds(X(:,X_indices) - D*smallA + Dj*gamma_j, 1);
  gamma_j = s*gamma_j;
  
else
  
  atom = collincomb(X,X_indices,gamma_j') - D*(smallA*gamma_j') + Dj*(gamma_j*gamma_j');
  atom = atom/norm(atom);
  gamma_j = rowlincomb(atom,X,1:size(X,1),X_indices) - (atom'*D)*smallA + (atom'*Dj)*gamma_j;

end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             sparsecode               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function A = sparsecode(X,D,XtX,G,thresh)

global CODE_SPARSITY CORE_ERROR codemode
global MEM_HIGH memusage
global ompparams

% NOTE: removed ompfunc and used directly omp/omp2.

if (memusage < MEM_HIGH)
%	A = ompfunc(D,X,G,thresh,ompparams{:});
	if (codemode == CODE_SPARSITY),
		A = omp(D,X,G,thresh,ompparams{:});
	elseif (codemode == CORE_ERROR),
		A = omp2(D,X,G,thresh,ompparams{:});
	end;
else  % memusage is high
  
  if (codemode == CODE_SPARSITY)
%     A = ompfunc(D'*X,G,thresh,ompparams{:});
	  A = omp(D'*X,G,thresh,ompparams{:});
    
  elseif (codemode == CODE_ERROR)
%     A = ompfunc(D'*X,XtX,G,thresh,ompparams{:});
	  A = omp2(D'*X,XtX,G,thresh,ompparams{:});
  end
  
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             compute_err              %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function err = compute_err(D,A,X)
  
global CODE_SPARSITY codemode

if (codemode == CODE_SPARSITY)
  err = sqrt(sum(reperror2(X,D,A))/numel(X));
else
  err = nnz(A)/size(X,2);
end

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


function err2 = reperror2(X,D,A)

% compute in blocks to conserve memory
err2 = zeros(1,size(X,2));
blocksize = 2000;
for i = 1:blocksize:size(X,2)
  blockids = i : min(i+blocksize-1,size(X,2));
  err2(blockids) = sum((X(:,blockids) - D*A(:,blockids)).^2);
end

end


function Y = colnorms_squared(X)

% compute in blocks to conserve memory
Y = zeros(1,size(X,2));
blocksize = 2000;
for i = 1:blocksize:size(X,2)
  blockids = i : min(i+blocksize-1,size(X,2));
  Y(blockids) = sum(X(:,blockids).^2);
end

end
