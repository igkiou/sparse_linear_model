function [x,f,exitflag,output] = lbfgs_minimize(funObj,x0,options,varargin)
% minFunc(funObj,x0,options,varargin)
%
% Unconstrained optimizer using a line search strategy
%
% Uses an interface very similar to fminunc
%   (it doesn't support all of the optimization toolbox options,
%       but supports many other options).
%
% It computes descent directions using 'lbfgs': Quasi-Newton with 
% Limited-Memory BFGS Updating (uses a predetermined nunber of previous
% steps to form a low-rank Hessian approximation).
%
% Several line search strategies are available for finding a step length satisfying
%   the termination criteria ('LS'):
%   - 0: Backtrack w/ Step Size Halving
%   - 1: Backtrack w/ Quadratic/Cubic Interpolation from new function values
%   - 2: Backtrack w/ Cubic Interpolation from new function + gradient
%   values (default for 'bb' and 'sd')
%   - 3: Bracketing w/ Step Size Doubling and Bisection
%   - 4: Bracketing w/ Cubic Interpolation/Extrapolation with function +
%   gradient values (default for all except 'bb' and 'sd')
%   - 5: Bracketing w/ Mixed Quadratic/Cubic Interpolation/Extrapolation
%   - 6: Use Matlab Optimization Toolbox's line search
%           (requires Matlab's linesearch.m to be added to the path)
%
%   Above, the first three find a point satisfying the Armijo conditions,
%   while the last four search for find a point satisfying the Wolfe
%   conditions.  If the objective function overflows, it is recommended
%   to use one of the first 3.
%   The first three can be used to perform a non-monotone
%   linesearch by changing the option 'Fref'.
%
% Several strategies for choosing the initial step size are avaiable ('LS_init'):
%   - 0: Always try an initial step length of 1 (default) (t = 1)
%   - 1: Use a step similar to the previous step (t = t_old*min(2,g'd/g_old'd_old))
%   - 2: Quadratic Initialization using previous function value and new
%   function value/gradient (use this if steps tend to be very long)
%       (t = min(1,2*(f-f_old)/g))
%   - 3: The minimum between 1 and twice the previous step length
%       (t = min(1,2*t)
%
% Inputs:
%   funObj is a function handle
%   x0 is a starting vector;
%   options is a struct containing parameters
%  (defaults are used for non-existent or blank fields)
%   all other arguments are passed to funObj
%
% Outputs:
%   x is the minimum value found
%   f is the function value at the minimum found
%   exitflag returns an exit condition
%   output returns a structure with other information
%
% Supported Input Options
%   Display - Level of display [ off | final | (iter) | full | excessive ]
%   MaxFunEvals - Maximum number of function evaluations allowed (1000)
%   MaxIter - Maximum number of iterations allowed (500)
%   TolFun - Termination tolerance on the first-order optimality (1e-5)
%   TolX - Termination tolerance on progress in terms of function/parameter changes (1e-9)
%   c1 - Sufficient Decrease for Armijo condition (1e-4)
%   c2 - Curvature Decrease for Wolfe conditions (.2 for cg methods, .9 otherwise)
%   LS_init - Line Search Initialization -see above (2 for cg/sd, 4 for scg, 0 otherwise)
%   LS - Line Search type -see above (2 for bb, 4 otherwise)
%   Fref - Setting this to a positive integer greater than 1
%       will use non-monotone Armijo objective in the line search.
%       (20 for bb, 10 for csd, 1 for all others)
%   useComplex - if 1, use complex differentials when computing numerical derivatives
%       to get very accurate values (default: 0)
%   useMex - where applicable, use mex files to speed things up (default: 1)
%
% Method-specific input options:
% Corr - number of corrections to store in memory (default: 100)
%     (higher numbers converge faster but use more memory)
% Damped - use damped update (default: 0)
%
% Supported Output Options
%   iterations - number of iterations taken
%   funcCount - number of function evaluations
%   algorithm - algorithm used
%   firstorderopt - first-order optimality
%   message - exit message
%   trace.funccount - function evaluations after each iteration
%   trace.fval - function value after each iteration
%
% Author: Mark Schmidt (2006)
% Web: http://www.cs.ubc.ca/~schmidtm
%

if nargin < 3
    options = [];
end

% Get Parameters
[verbose,verboseI,debug,doPlot,maxFunEvals,maxIter,tolFun,tolX,method,...
    corrections,c1,c2,LS_init,LS,Fref,Damped,useMex] = ...
    minFunc_processInputOptions(options);

% Initialize
p = length(x0);
d = zeros(p,1);
x = x0;
t = 1;

% If necessary, form numerical differentiation functions
funEvalMultiplier = 1;

% Evaluate Initial Point
[f,g] = funObj(x,varargin{:});
funEvals = 1;

% Output Log
if verboseI
    fprintf('%10s %10s %15s %15s %15s\n','Iteration','FunEvals','Step Length','Function Val','Opt Cond');
end

% Initialize Trace
trace.fval = f;
trace.funcCount = funEvals;

% Check optimality of initial point
if sum(abs(g)) <= tolFun
    exitflag=1;
    msg = 'Optimality Condition below TolFun';
    if verbose
        fprintf('%s\n',msg);
    end
    if nargout > 3
        output = struct('iterations',0,'funcCount',1,...
            'algorithm',method,'firstorderopt',sum(abs(g)),'message',msg,'trace',trace);
    end
    return;
end

% Perform up to a maximum of 'maxIter' descent steps:
for i = 1:maxIter

    % ****************** COMPUTE DESCENT DIRECTION *****************

	% Update the direction and step sizes

	if i == 1
		d = -g; % Initially use steepest descent direction
		old_dirs = zeros(length(g),0);
		old_stps = zeros(length(d),0);
		Hdiag = 1;
	else
		if Damped
			[old_dirs,old_stps,Hdiag] = dampedUpdate(g-g_old,t*d,corrections,debug,old_dirs,old_stps,Hdiag);
		else
			[old_dirs,old_stps,Hdiag] = lbfgsUpdate(g-g_old,t*d,corrections,debug,old_dirs,old_stps,Hdiag);
		end

		if useMex
			d = lbfgsC(-g,old_dirs,old_stps,Hdiag);
		else
			d = lbfgs(-g,old_dirs,old_stps,Hdiag);
		end
	end
	g_old = g;

    if ~isLegal(d)
        fprintf('Step direction is illegal!\n');
        pause;
        return
    end

    % ****************** COMPUTE STEP LENGTH ************************

    % Directional Derivative
    gtd = g'*d;

    % Check that progress can be made along direction
    if gtd > -tolX
        exitflag=2;
        msg = 'Directional Derivative below TolX';
        break;
    end

    % Select Initial Guess
    if i == 1
		t = min(1,1/sum(abs(g)));
    else
        if LS_init == 0
            % Newton step
            t = 1;
        elseif LS_init == 1
            % Close to previous step length
            t = t*min(2,(gtd_old)/(gtd));
        elseif LS_init == 2
            % Quadratic Initialization based on {f,g} and previous f
            t = min(1,2*(f-f_old)/(gtd));
        elseif LS_init == 3
            % Double previous step length
            t = min(1,t*2);
        end

        if t <= 0
            t = 1;
        end
    end
    f_old = f;
    gtd_old = gtd;

    % Compute reference fr if using non-monotone objective
    if Fref == 1
        fr = f;
    else
        if i == 1
            old_fvals = repmat(-inf,[Fref 1]);
        end

        if i <= Fref
            old_fvals(i) = f;
        else
            old_fvals = [old_fvals(2:end);f];
        end
        fr = max(old_fvals);
    end

    % Line Search
    f_old = f;
    if LS < 3 % Use Armijo Bactracking
        % Perform Backtracking line search
       [t,x,f,g,LSfunEvals] = ArmijoBacktrack(x,t,d,f,fr,g,gtd,c1,LS,tolX,debug,doPlot,1,funObj,varargin{:});
       funEvals = funEvals + LSfunEvals;

	else
        % Find Point satisfying Wolfe
        [t,f,g,LSfunEvals] = WolfeLineSearch(x,t,d,f,g,gtd,c1,c2,LS,25,tolX,debug,doPlot,1,funObj,varargin{:});
        funEvals = funEvals + LSfunEvals;
        x = x + t*d;
    end

    % Output iteration information
    if verboseI
        fprintf('%10d %10d %15.5e %15.5e %15.5e\n',i,funEvals*funEvalMultiplier,t,f,sum(abs(g)));
	end

    % Update Trace
    trace.fval(end+1,1) = f;
    trace.funcCount(end+1,1) = funEvals;

    % Check Optimality Condition
    if sum(abs(g)) <= tolFun
        exitflag=1;
        msg = 'Optimality Condition below TolFun';
        break;
    end

    % ******************* Check for lack of progress *******************

    if sum(abs(t*d)) <= tolX
        exitflag=2;
        msg = 'Step Size below TolX';
        break;
    end


    if abs(f-f_old) < tolX
        exitflag=2;
        msg = 'Function Value changing by less than TolX';
        break;
    end

    % ******** Check for going over iteration/evaluation limit *******************

    if funEvals*funEvalMultiplier > maxFunEvals
        exitflag = 0;
        msg = 'Exceeded Maximum Number of Function Evaluations';
        break;
    end

    if i == maxIter
        exitflag = 0;
        msg='Exceeded Maximum Number of Iterations';
        break;
    end

end

if verbose
    fprintf('%s\n',msg);
end
if nargout > 3
    output = struct('iterations',i,'funcCount',funEvals*funEvalMultiplier,...
        'algorithm',method,'firstorderopt',sum(abs(g)),'message',msg,'trace',trace);
end

end

