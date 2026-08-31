function L = mkPrecision_AR1prior_nonunif(tau,rho,tvec,DCflag)
% L = mkPrecision_AR1prior_nonunif(tau, rho, tvec)
%
%   Covariance:
%       L(i,j) = exp(-abs(t(i)-t(j))/tau)
%
%   Inputs:
%   -------
%       tau - positive length scale
%       rho - marginal variance
%      tvec - vector of ordered sample locations
%    DCflag - flag for including extra col & row of zeros (if 1)
%
%
%   Output:
%   -------
%       L - sparse inverse covariance (precision) matrix

tvec = tvec(:);
N = length(tvec);

% Correlation between neighboring samples
rvec = exp(-diff(tvec)/tau);

% Off-diagonal entries
offdiag = -rvec ./ (1-rvec.^2);

% Diagonal entries
d = zeros(N,1);

d(1) = 1 / (1-rvec(1)^2);
d(N) = 1 / (1-rvec(end)^2);

if N > 2
    d(2:N-1) = ...
        1 ./ (1-rvec(1:end-1).^2) + ...
        rvec(2:end).^2 ./ (1-rvec(2:end).^2);
end

% Construct sparse tridiagonal matrix
L = spdiags([[offdiag; 0], d, [0; offdiag]],[-1 0 1], N, N)/rho;

% --- add row and column of zeros for DC term, if needed ----
if (nargin == 4) && DCflag
    L = blkdiag(L,0);
end
