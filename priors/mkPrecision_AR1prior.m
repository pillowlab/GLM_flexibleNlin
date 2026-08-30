function L = mkPrecision_AR1prior(tau,rho,nw,DCflag)
% L = mkPrecision_AR1prior(tau,rho,nw,DCflag)
%
% Build sparse precision matrix (inverse covariance matrix) corresonding to
% an AR1 or "exponential" covariance function 
%
% Inputs:
% -------
%       tau - time constant of exponential decay (must be > 0)
%       rho - marginal variance (must be > 0)
%        nw - length of param vector 
%    DCflag - boolean for whether to add row/col of zeros for last element
%
% Outputs:
% --------
%   L [nw x nw] or [nw+1 x nw+1 ] - precision matrix 
%
% Notes:
% ------
% Inverse prior covariance matrix, linear case:
%
%  L  =  1/(1-a^2) [ 1 -a
%                      -a 1+a^2 -a                       
%                        .   .   .
%                          -a 1+a^2 -a
%                                -a  1 ]
%
% where a = exp(-1/tau)


% ---  Check intputs --- 
if nargin < 4
    DCflag = 0;   % assume no DC dterm
end

if tau <= 0
    error('abs autocorrelation parameter aa must be > 0');
end

if rho <= 0
    error('rho must be > 0');
end

% --- Build covariance matrix -----

aa = exp(-1/tau); % decay of covariance in 1 time step
const = 1/(rho*(1-aa.^2)); % multiplicative constant 

vdiag = [1;ones(nw-2,1)+aa^2;1]; % diagonal values
voffdiag = -ones(nw,1)*aa;  % off-diagonal values

L = spdiags([voffdiag,vdiag,voffdiag]*const,-1:1,nw,nw);

% --- add row and column of zeros for DC term, if needed ----
if DCflag
    L = blkdiag(L,0);
end
