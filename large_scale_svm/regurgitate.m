function regurgitate(varargin)
% REGURGITATE(VARARGIN) prints out name/type/values (if scalar, or char string)
% of each elements in argument list VARARGIN
%
% For array (numerical/cell etc), the function prints out just the size of the
% array.
%
% For struct, the function prints out the fields names and values (recursively).
%
% by feisha@cis.upenn.edu
whocalledme = dbstack('-completenames');
fprintf('\n');
fprintf('***** Regurgitate parameters ******  \n');
if length(whocalledme) >=2
  fprintf('Running script/function file %s\n', whocalledme(2).file);
else

  fprintf('Calling from command line\n');
end
fprintf('    begining of arguments\n');
for idx=1:nargin
    printarg(inputname(idx), varargin{idx}, '');
end 
fprintf('    end of arguments\n');
fprintf('*************************************\n');
return


function printarg(name, var, offset)

if isempty(var)
    fprintf([offset '%s:\t[]\n'], name);
    return;
end

if isa(var, 'function_handle')
  ff = functions(var);
  fprintf([offset '%s:\t''@%s''\t(type: function handle)\n'], name, ff.function);
    return;
end


if isstruct(var) % structure, doing recursion on each field
    fprintf([offset '%s:\t(struct)\n'], name);
    feldnames = fieldnames(var);
    for idx= 1:length(feldnames)
            printarg(feldnames{idx}, getfield(var, feldnames{idx}), [offset ...
                                '    ']);
    end
    return
end

if iscell(var)  % numerical array, just print out size
    sizeofvar = size(var);
    fprintf([offset '%s:\t[ %d x %d ](first 2D)\t(type :%s)\n'], name, sizeofvar(1), ...
            sizeofvar(2), class(var));

    return;
end



if isscalar(var) % scalar, print out value
    fprintf([offset '%s:\t%f\t (type: %s)\n'], name, var, class(var));
    return;
end

if isnumeric(var)  % numerical array, just print out size
    sizeofvar = size(var);
    fprintf([offset '%s:\t[ %d x %d ](first 2D)\t(type :%s)\n'], name, sizeofvar(1), ...
            sizeofvar(2), class(var));
    return;
end

if ischar(var) % char, print out value
    fprintf([offset '%s:\t''%s''\t (type: %s)\n'], name, var, class(var));
    return;
end

% when everything fails
fprintf([offset '%s:\t(don''t know what to do)\t(type:%s)\n'], name, class(var));
return;


