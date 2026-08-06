function L = neglogli_PoissGLM_nlin(nlprs,wts,Xmat,Yvec,nlfunptr,dtbin)
% L = neglogli_PoissGLM_nlin(nlprs,wts,Xmat,yvals,nlfunptr,dtbin)
%
% Negative log-likelihood of Poisson GLM with arbitrary nonlinearity, as a
% function of the parameters governing the nonlinearity
%
% INPUT:
%     nlprs [K x 1] - parameters governing the nonlinearity
%       wts [M x 1] - weights
%      Xmat [N x M] - design matrix
%      Yvec [N x 1] - observed Poisson random variables
%             nlfun - handle for nonlinear function
%     dtbin [1 x 1] - assumed time bin size 
%
% OUTPUT: 
%         L [1 x 1] - negative log-likelihood


% Put parameters governing nonlinearity into function
nlfun = @(v)nlfunptr(v,nlprs); 

% Multiply design matrix by weights
z = Xmat*wts;  

% Compute output of nonlinearity
f = nlfun(z);

% Compute negative log-likelihood
L = -Yvec'*(log(f)+log(dtbin)) + sum(f)*dtbin;
