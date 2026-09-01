function [logP,dlogP,H] = logprior_normalizedGaussian(prvec,Cinv,D)
% [logP,dlogP,H] = logprior_normalizedGaussian(prvec,Cinv,D)
%
% Evaluates the penalty function - 1/2 x ^T C^{-1} x / (x^T D x), as well
% as its gradient and Hessian 
% 
% If D is the identity matrix, this is equivalent to evaluating a Gaussian
% prior on the normalized parameter vector x.  That is, convert x to a unit
% vector by dividing it by its Euclidean norm, then evaluate the log of a
% Gaussian on this normalized vector (not including the normalizing
% constant).
%
% Inputs:
% -------
%   prvec [n x 1] - parameter vector (last element can be DC)
%    Cinv [n x n] - inverse covariance
%       D [n x n] - matrix for normalizing the parameters
%
% Outputs:
% --------
%       p [1 x 1] - log-prior
%      dp [n x 1] - grad
%       H [n x n] - Hessian


num  = prvec' * Cinv * prvec;
denom  = prvec' * D * prvec;
Cinvx = Cinv * prvec;
Dx = D * prvec;

logP = -0.5 * (num/denom);

dlogP = - (denom*Cinvx - num*Dx) / denom^2;

H = - Cinv/denom + num*D/denom^2 ...
    + 2*(Cinvx*Dx' + Dx*Cinvx')/denom^2 ...
    - 4*num*(Dx*Dx')/denom^3;

%num = wts'*Cinv*wts;  % numerator of penalty
% denom = wts'*D*wts;   % denomenator of penalty
% logP = -0.5 * (num / denom);
% 
% if nargout > 1
%     dlogP = (- denom*Cinv*wts + num*wts)/denom.^2;
% end
% 
% if nargout > 2
%     negCinv = spdiags(-Cinvdiag,0,nx,nx);
%     logdetrm = sum(log(Cinvdiag));
% end
