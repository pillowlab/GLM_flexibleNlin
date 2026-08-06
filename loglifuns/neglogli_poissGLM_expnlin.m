function [neglogli, dL, H] = neglogli_poissGLM_expnlin(wts,Xmat,Yvec,dtbin)
% [neglogli, dL, H] = Loss_GLM_logli_expnlin(wts,Xmat);
%
% Compute negative log-likelihood of data undr Poisson GLM model with
% exponential nonlinearity
%
% Inputs:
%     wts [d x 1] - parameter vector
%    Xmat [T x d] - design matrix
%    Yvec [T x 1] - response (spike count per time bin)
%   dtbin [1 x 1] - time bin size used 
%
% Outputs:
%   neglogli   = negative log likelihood of spike train
%   dL [d x 1] = gradient 
%   H  [d x d] = Hessian (second deriv matrix)

% Compute GLM filter output and condititional intensity
vv = Xmat*wts; % filter output
rr = exp(vv)*dtbin; % conditional intensity (per bin)

% ---------  Compute log-likelihood -----------
Trm1 = -vv'*Yvec; % spike term from Poisson log-likelihood
Trm0 = sum(rr);  % non-spike term 
neglogli = Trm1 + Trm0;

% ---------  Compute Gradient -----------------
if (nargout > 1)

    dL = Xmat'*(rr-Yvec);

end

% ---------  Compute Hessian -------------------
if nargout > 2
    H = Xmat' * (rr .* Xmat); % only non-spiking term contributes
end

