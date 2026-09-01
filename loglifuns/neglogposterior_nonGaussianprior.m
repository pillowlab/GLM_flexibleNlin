function [negLogPost,grad,H] = neglogposterior(prs,negloglifun,logpriorfun,varargin)
% [negLP,grad,H] = neglogposterior(prs,negloglifun,logprifun,varargin)
%
% Compute negative log-posterior given a negative log-likelihood function
% and zero-mean Gaussian prior with inverse covariance 'Cinv'.
%
% Inputs:
 %   prs [d x 1] - parameter vector
%    negloglifun - handle for negative log-likelihood function
%   Cinv [d x d] - response (spike count per time bin)
%
% Outputs:
%          negLP - negative log posterior
%   grad [d x 1] - gradient 
%      H [d x d] - Hessian (second deriv matrix)

% Compute negative log-posterior by adding quadratic penalty to log-likelihood

switch nargout

    case 1  % evaluate function
        negLogPost = negloglifun(prs) - logpriorfun(prs,varargin{:});
    
    case 2  % evaluate function and gradient
        [negLogPost,grad] = negloglifun(prs);
        [logP,dlogP] = logpriorfun(prs,varargin{:});

        negLogPost = negLogPost - logP;
        grad = grad - dlogP;

    case 3  % evaluate function and gradient
        [negLogPost,grad,H] = negloglifun(prs);
        [logP,dlogP,Hpri] = logpriorfun(prs,varargin{:});
        negLogPost = negLogPost - logP;
        grad = grad - dlogP;
        H = H - Hpri;
end

