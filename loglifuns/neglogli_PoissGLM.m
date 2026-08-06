function [L,dL,ddL] = neglogli_PoissGLM(wts,Xmat,Yvec,nlfun,dtbin)
% [L,dL,ddL] = neglogli_PoissGLM(wts,Xmat,Yvec,nlfun,dtbin)
%
% Negative log-likelihood of Poisson GLM with arbitrary nonlinearity
%
% INPUT:
%    wts [M x 1] - weights
%   Xmat [N x M] - design matrix
%   Yvec [N x 1] - observed Poisson random variables
%          nlfun - handle for nonlinear function
%  dtbin [1 x 1] - assumed time bin size 
%
% OUTPUT: 
%         L [1 x 1] - negative log-likelihood
%        dL [M x 1] - gradient
%       ddL [M x M] - Hessian (2nd deriv matrix)


% Multiply design matrix by weights
z = Xmat*wts;  

switch nargout

    case {0,1} % --- Compute neglogli only -----------------------
  
        f = nlfun(z); % compute nonlinearity output
        L = -Yvec'*(log(f)+log(dtbin)) + sum(f)*dtbin;

    case 2 % ---  Compute neglogli & Gradient ----------------

        [f,df] = nlfun(z); % compute nonlinearity output and its deriv
        L = -Yvec'*(log(f)+log(dtbin)) + sum(f)*dtbin;

        % grad
        aa = (df*dtbin-(Yvec.*df./f)); % weights for computing grad
        dL = Xmat'*aa;

    case 3 % --- Compute neglogli, Gradient & Hessian --------

        [f,df,ddf] = nlfun(z);  % compute nonlinearity and 1st two derivs
        L = -Yvec'*(log(f)+log(dtbin)) + sum(f)*dtbin;

        % grad
        aa = (df*dtbin-(Yvec.*df./f)); % weights for computing grad
        dL = Xmat'*aa;
        
        % Hessian
        bb = ddf*dtbin-Yvec.*(ddf./f-(df./f).^2); % weights for Hessian
        %ddL = Xmat'*bsxfun(@times,Xmat,bb);
        ddL = Xmat' * (bb .* Xmat);

end
