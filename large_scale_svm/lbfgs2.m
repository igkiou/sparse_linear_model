function [retval xstarbest xstarfinal history]= lbfgs2(x0, options,  f, sf, varargin)
%
% [retval xstar2 fstar2 gstar2 varargout]= lbfgs(x0, options,  f, sf, varargin)
% use LBFGS algorithm to minimize the function f(x,...). 
%
% x0: starting point
% f:  objective function, takes the form f(x0, varargin) and returns
%   [fvalue, gradient, varargout] = f(x0, varargin)
% sf: a post-processing function that can be called by lbfgs(), for
% example, extra reporting etc. It takes following argument
%
%     sf(xstar2, fstar2, gstar2, varargin, varargout)
%


if options.echo == true
    fprintf('-------------------------------------\n');
%     t0 = printnow;
    regurgitate(x0, options, f, sf, varargin);
end

n = length(x0(:));
m = options.m;

s = zeros(n, m);
y = zeros(n, m);
alpha = zeros(m,1);

% we need to do m line search in order to start the iteration for LBFGS
x_idx = x0;
if options.echo == true
    fprintf('Starting initial line search to collect enough gradient information...\n');
end
[f_idx, g_idx] = feval(f, x0, varargin{:});

if options.echo == true
    fprintf('    Initial function value: %f\n', f_idx);
end

if isempty(sf) == 0
  xstarbest = x0; history = [];
  [xstarbest history]=feval(sf, x0, 0, xstarbest, history, varargin{:});
else
  history.obj(1) = f_idx;
end

astarguess = options.wolfe.a1;
astar = options.wolfe.a1;

k = 0;
gstar = g_idx;
while 1
  if k < m  %% HOWMANY pairs of x/gradx info we have
    howmany = k;
  else
    howmany = m;
  end

  if howmany < 1
    gamma_k = 1;
  else
    gamma_k = (1/rou(howmany))/sum(y(:,howmany).^2,1);
  end
  % compute search directions
  q = g_idx(:);
  for i = howmany:-1:1
    alpha(i) = rou(i)* s(:,i)'*q;
    q = q - alpha(i)*y(:,i);
  end
  s_idx = gamma_k*q;
  for i = 1:howmany
    beta = rou(i)*y(:,i)'*s_idx;
    s_idx = s_idx + s(:,i)*(alpha(i)-beta);
  end

  %%% line search
  s_idx = reshape(s_idx, size(x0));
  if k < m
     s(:, k+1) = - x_idx(:);
    y(:, k+1) = - g_idx(:);    
  else
    s(:, 1:end-1) = s(:, 2:end);
    y(:, 1:end-1) = y(:, 2:end);
    s(:, end) = -x_idx(:);
    y(:, end) = -g_idx(:);
  end
  if k ==0
    astar = options.wolfe.a0;
  else
    if astar < options.wolfe.a1 && abs(astar - options.wolfe.a1) < 1e-2*options.wolfe.a1
      astar = (astar + options.wolfe.a1)/2;
    else
      astar = (astar + options.wolfe.amax)/2;
    end
  end

  [astar xstar fstar gstar] =  lineSearchWolfe(x_idx, f_idx, g_idx, -s_idx, ...
                                              astar, options.wolfe.amax, options.wolfe.c1, ...
                                              options.wolfe.c2, ...
                                                  options.wolfe.maxiter, f, varargin{:});
   
  % update memory
  if k <m
    s(:, k+1) = xstar(:) + s(:, k+1);
    y(:, k+1)= gstar(:) + y(:,k+1);
    rou(k+1) = 1 ./ (s(:,k+1)'*y(:,k+1));
  else
    s(:, end) = xstar(:) +s(:, end);
    y(:, end) = gstar(:) +y(:, end);
    rou(1:end-1) = rou(2:end);
    rou(end) = 1 ./ (s(:, end)'*y(:, end));
  end

  k = k+1;
  if options.echo == true
    fprintf('  Iter: %d, f = %f, step size=%f\n', k, fstar, astar);
  end

  if isempty(sf) == 0
    [xstarbest history] = feval(sf, xstar, k, xstarbest, history, varargin{:});
  else
    history.obj(k+1) = fstar;
  end
  if max(abs(gstar(:)))/(1+abs(fstar)) <= options.termination || ...
        max(abs((xstar(:)-x_idx(:)) ./ (xstar(:)+eps))) <= options.xtermination
    if options.echo == true
        fprintf('Termination criteria met...\n');
    end
    
    retval = 0;
    xstarfinal = xstar;
    if isempty(sf)
      xstarbest = xstar;
    end
    return;
  end
  if k >= options.maxiter
    if options.echo == true
        fprintf('Reach the maximum number of iteration...\n');
    end
    retval = 1;
    xstarfinal = xstar;
    if isempty(sf)
      xstarbest = xstar;
    end
    
    return;
  end
  if exist('./killing.me', 'file') == 2
    if options.echo == true
    fprintf('Find the killing.me file..Stop optimization.\n');
    end
    !rm killing.me;
    retval = 2;
    xstarfinal = xstar;
    if isempty(sf)
      xstarbest = xstar;
    end
    
    return;
  end
  f_idx = fstar;
  g_idx = gstar;
  x_idx= xstar;
end
  
return





