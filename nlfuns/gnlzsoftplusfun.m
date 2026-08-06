function [f,df,ddf] = gnlzsoftplusfun(x,pow)
%  [f,df,ddf] = gnlzsoftplusfun(x,pow)
%
%  Implements the nonlinearity:  
%     f(x) = log(1+exp(x)).^pow;
%  plus first and second derivatives


f0 = log(1+exp(x));
f = f0.^pow;

if nargout > 1
    df = pow*f0.^(pow-1).*exp(x)./(1+exp(x));
end
if nargout > 2
    if pow == 1
        ddf = pow*f0.^(pow-1).*exp(x)./(1+exp(x)).^2;
    else
        ddf = pow*f0.^(pow-1).*exp(x)./(1+exp(x)).^2 + ...
              pow*(pow-1)*f0.^(pow-2).*(exp(x)./(1+exp(x))).^2;
    end
end


% Check for small values
if any(x<-30)
    iix = (x<-30);
    f(iix) = exp(x(iix));
    df(iix) = f(iix);
    ddf(iix) = f(iix);
end

% Check for large values
if any(x>500)
    iix = (x>500);
    f(iix) = x(iix);
    df(iix) = 1;
    ddf(iix) = 0;
end