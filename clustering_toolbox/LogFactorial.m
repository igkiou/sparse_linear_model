function p = LogFactorial(n)
%LOGFACTORIAL Calculates the Natural log of the factorial of n
% NL = LogFactorial(n)
%

%(C) David Corney (2000)   		D.Corney@cs.ucl.ac.uk

[R,C]=size(n);

if R==1 & C==1
   p=sum(log(1:n));
   return
end

if min(R,C)==1
   for i = 1:length(n)
      p(i)=sum(log(1:n(i)));
   end
   return
end

error('LogFactorial requires scalar or vector input!')

