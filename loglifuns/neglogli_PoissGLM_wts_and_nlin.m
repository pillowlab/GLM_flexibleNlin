function L = neglogli_PoissGLM_wts_and_nlin(jointprs,xvals,yvals,nlfunptr,dtbin)
% [L,dL,ddL] = neglogli_PoissGLM_wts_and_nlin(jointprs,xvals,yvals,nlfunptr,dtbin)
%
% Negative log-likelihood of Poisson GLM with arbitrary nonlinearity, as a
% function of the weights AND the parameters governing the nonlinearity
%
% INPUT:
% ------
%  jointprs [M+K x 1] - parameters for weights(length M) and nonlinearity (length K)
%       xvals [N x M] - design matrix
%       yvals [N x 1] - observed Poisson random variables
%            nlfunptr - handle for nonlinear function
%       dtbin [1 x 1] - assumed time bin size 
%
% OUTPUT: 
% --------
%   L [1 x 1] - negative log-likelihood

nwts = size(xvals,2);  % number of weights in weight vector
wts = jointprs(1:nwts);  % weights
nlprs = jointprs(nwts+1:end); % nonlinearity params

% Put parameters governing nonlinearity into nonlinear function
nlfun = @(v)nlfunptr(v,nlprs); 

% Multiply design matrix by weights
z = xvals*wts;  

% Compute output of nonlinearity
f = nlfun(z); 

% Compute log-likelihood
L = -yvals'*(log(f)+log(dtbin)) + sum(f)*dtbin;
