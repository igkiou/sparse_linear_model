function [astar xstar fstar gstar varargout] = lineSearchWolfe(x0, f0, g0, s0, a1, amax, c1, c2, maxiter, f, varargin)
% astar = lineSearchWolfe(f0, g0, a1, amax, c1, c2, maxiter, f, varargin) 
% determines a line search step size that satisfies the strong Wolfe
% condition.
%
% the algorithm implements what was described in Numerical Optimization
% (Nocedal and Wright, Springer 1999), on pp.59.
%
% Inputs:
%     x0: current parameter
%     f0: function value at step size 0 
%     g0: gradient of f with respect to its parameter at step size 0
%     s0:  search direction at step size 0 
%     a1: initial step size to start
%     amax: maximum step size allowed
%     c1:  the constant for sufficient reduction (Wolfe condition 1), c2
%     the constant for curvature condition (Wolfe condition 2).
%
%       Note that c1 is often set as 10^(-4) and c2 is set as 0.9 if
%       using Newton type algorithm or 0.1 for nonlinear conjugate
%       gradients.
%
%   maxiter: maxium iteration to search for ste size
%   f:   the function that is used to compute function value and gradient
%   of the function with repect to its parameters. f should be a function
%   handle that takes f(x, varargin) where x is the current parameter value
%   and varargin are other parameters that need to be passed into f(). It
%   should return two values: the firs being the function value and the
%   second being the gradient
%

echo = false;

ai_1 = 0; 
ai = a1;
i = 1;
fi_1 = f0;
linegrad0 = g0(:)'*s0(:);
linegradi_1 = linegrad0;
varargout(1) = {[]};

while 1
  xstar = x0+ai*s0;

  [fi gi ] = feval(f,  xstar, varargin{:});
  linegradi = gi(:)'*s0(:);  
  if fi > (f0+c1*ai*linegrad0) || ( fi >= fi_1 && i > 1)
    if echo == true
        fprintf('Zooming\n');
    end

    [astar xstar fstar gstar ] = zoom(ai_1, ai,x0,f0, g0, s0, c1, c2, ...
                                        linegrad0, fi_1,  linegradi_1,fi, linegradi,f, varargin{:});

    return;
  end

  if abs(linegradi) <= -c2*linegrad0
    astar = ai; 
    fstar = fi; gstar = gi;
    return;
  end
  if linegradi >=0
    if echo == true
        fprintf('Zooming(2)\n');
    end
    [astar xstar fstar gstar ] = zoom(ai, ai_1, x0,f0, g0, s0, c1, c2, ...
                                        linegrad0, fi, linegradi, fi_1, linegradi_1, f, varargin{:});

    return;
  end
  i = i+1;
  if abs(ai - amax) <= 0.01*amax || i > maxiter
    fstar = fi; gstar = gi; astar = ai; 
    if echo == true
        fprintf(['Maxium number of iteration exceeded or getting too close to ' ...
             'the maximum step size..return with best found so far.\n' ]);
    end
    return;
  end
  ai_1 = ai;
  fi_1 = fi;
  linegradi_1 = linegradi;
  
  if echo == true
    fprintf('Extrapolating step length: a_i from %f  ---> ', ai);
  end
  ai = (ai + amax)/2;  % better way to do it later
  if echo == true
    fprintf('%d\n', ai);
  end
  %ai = ai* exp(log(amax/ai)/5); % 5 steps to get to the max, bad idea?
end

function [astar xstar fstar gstar varargout] = zoom(alo, ahi, x0, f0, g0, ...
                                                    s0, c1, c2, linegrad0, ...
                                                    falo,galo, fhi, ghi,  f, varargin)
echo = false;
i=0;
varargout(1) = {[]};
while 1
  %aj = (alo + ahi)/2;
  % find a trial aj using intepolation

  d1 = galo+ghi - 3*(falo-fhi)/(alo-ahi);
  d2 = sqrt(d1*d1 - galo*ghi);
  aj = ahi - (ahi-alo)*(ghi+d2-d1)/(ghi-galo+2*d2);
  if echo == true
    fprintf('Zooming --> aj: %f, alo: %f, ahi: %f\n', aj, alo, ahi);
  end
  if alo < ahi
    if aj < alo || aj > ahi
      aj = (alo+ahi)/2;
    end
  else
    if aj > alo || aj < ahi
      aj = (alo+ahi)/2;
    end
  end
  xstar = x0+aj*s0;
  [fj, gj ] = feval(f, xstar, varargin{:});
  if fj > f0 + c1*aj*linegrad0 || fj > falo
    ahi = aj;
    fhi = fj;
    ghi = gj(:)'*s0(:);
  else
    linegradj = gj(:)'*s0(:);
    if abs(linegradj) <= -c2*linegrad0
      astar = aj;
      fstar = fj; gstar = gj;
      return;
    end

    if linegradj*(ahi - alo) >=0
      ahi = alo;
      fhi = falo;
      ghi = galo;
    end
    alo = aj;
    falo = fj;
    galo = linegradj;
    
  end
  if abs(alo-ahi) <= 0.01*alo || i >= 10
    astar = aj;
    fstar = fj; gstar = gj;
    if echo == true
        fprintf('Zooming exceeds precision or max # of iteration\n');
    end
    return;
  end
  
  i = i+1;
end
      
