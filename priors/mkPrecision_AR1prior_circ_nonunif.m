function L = mkPrecision_AR1prior_circ_nonunif(tau, rho, tvec, T, DCflag)
% L = mkPrecision_AR1prior_circ_nonunif(tau, rho, tvec, T, DCflag)
% 
% Make precision matrix for a periodic exponential / OU covariance.
%
% Covariance:
%
%   K(i,j) = [exp(-d/tau) + exp(-(T-d)/tau)] ...
%            / [1 + exp(-T/tau)]
%
% where d = abs(t(i)-t(j)).
%
% Inputs:
% -------
%      tau - positive length scale
%      rho - marginal variance
%      t   - sample locations in [0,T), in increasing order
%      T   - circumference of interval
%   DCflag - flag for including extra col & row of zeros (if 1)
%
% Output:
% -------
%   L - sparse cyclic-tridiagonal precision matrix

tvec = tvec(:);
N = length(tvec); % number of elements in covariance

% Gaps between successive points, including wraparound
delta = [diff(tvec); T - tvec(end) + tvec(1)];

% Correlations across each gap
rvec = exp(-delta/tau);

% Edge coefficients
a = 1 ./ (1-rvec.^2);
b = -rvec ./ (1-rvec.^2);

% Each diagonal receives contributions from the edge entering
% the point and the edge leaving it.
d = [a(end); a(1:end-1)] + rvec.^2 .* a;

% Build cyclic tridiagonal matrix
ii = (1:N)';
jj = [2:N 1]';

L = sparse(ii, ii, d, N, N);

% Add each neighboring edge symmetrically
L = L + sparse(ii, jj, b, N, N) ...
              + sparse(jj, ii, b, N, N);

% Normalize so marginal variance is rho
R = exp(-T/tau);
L = ((1+R)/(1-R)) * L / rho;

