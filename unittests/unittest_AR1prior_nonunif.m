% Unit test to ensure that mkPrecision_AR1prior_nonunif produces a precision matrix with
% the correct inverse

% set up hyperparametrs
tau = 10;  % length scale
rho = 3;   % marginal varaince
nw = 100;  % number of weights

rowindices = [1:13,15:17,20:50,60:75,95:100];

% Make full precision using standard function
L = mkPrecision_AR1prior(tau,rho,nw);
C = inv(L);  % invert it

% marginalize out missing rows 
B = eye(nw);
B = B(:,rowindices);
Creduced = B'*C*B;
Lreduced = inv(Creduced);

% Now construct precision directly
Lreduced2 = mkPrecision_AR1prior_nonunif(tau,rho,rowindices);

% Plot covariance and precision matrices, if desired
subplot(221);
imagesc(Creduced); axis image; title('cov 1 '); 
subplot(222);
imagesc(inv(Lreduced2)); axis image; title('cov 2 '); 
subplot(223);
imagesc(Lreduced); axis image; title('precision 1');
subplot(224);
imagesc(Lreduced2); axis image; title('precision 2');

err = sum(sum((Lreduced-Lreduced2).^2));

if (err) < 1e-10
     fprintf('--- unittest_AR1prior_nonunif: PASSED ---\n');
 else
     warning('unittest_AR1prior_nonunif: FAILED');
 end

