function [f,df,ddf] = expfun(x);
%  [f,df,ddf] = logexp1(x);
%
%  Implements the nonlinearity:  
%     f(x) = log(1+exp(x)).^pow;
%  Where pow = 1;
%  plus first and second derivatives
%

f = exp(x);
if nargout > 1
    df = f;
end
if nargout > 2
    ddf = f;
end
