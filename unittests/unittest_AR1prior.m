% Unit test to ensure that mkPrecision_AR1prior produces a precision matrix with
% the correct inverse

% set up hyperparametrs
tau = 10;  % length scale
rho = 3;   % marginal varaince
nw = 100;  % number of weights

% Make precision using function
L = mkPrecision_AR1prior(tau,rho,nw);
C = inv(L);  % invert it

% Compare first row with analytic form
Crow1 = full(C(1,:));  % first row of resulting Cov
exactrowcov = rho*exp(-(0:nw-1)/tau);  % analytic form for 1st row

err1 = sum((Crow1-exactrowcov).^2); % error

% ----------------------------------------------------------
% now redo it for larger precision matrix padded for the DC term

includeDC = 1;  % Boolean for including an extra row and col of zeros
L2 = mkPrecision_AR1prior(tau,rho,nw,includeDC);
C2 = inv(L2(1:nw,1:nw));  % invert just the part corresponding to weights
C2row1 = full(C2(1,:));  % extract first row

err2 = sum((C2row1-exactrowcov).^2);  % compute error

if (err1+err2) < 1e-10
    fprintf('--- unittest_AR1prior: PASSED ---\n');
else
    warning('unittest_AR1prior: FAILED');
end

