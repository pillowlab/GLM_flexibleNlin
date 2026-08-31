% demo2_PoissonGLM_regularized.m
%
% Simulate and recover weights from a Poisson GLM with known nonlinearity generalized-softplus
% nonlinearity using graph Laplacian and AR1 regularization 

setpath;  % add necessary paths
clear;  % clear memory

%% 1. Make simulated dataset =============

dtbin = .01; % bin size for representing time (s)

nw = 100;  % number of weights

% Create stimulus 
nsecTrain = 25;   % stimulus length 
nsampsTrain = round(nsecTrain/dtbin);  % number of bins for training data
nsampsTest = 1e6;  % pick large value for size of test set

IIDTRAIN = 0;  % flag to set training data
if IIDTRAIN
    % Use Gaussian white noise for train
    Xtrain = randn(nsampsTrain,nw); % training stimulus
else
    % Use correlated 1/F stimuli for training
    FreqPow = 2;  % power of frequency fall-off
    Xtrain = make1Fstimuli(nw,nsampsTrain,FreqPow); % training stimulus
end

% Set test stimuli
%Xtest = make1Fstimuli(nw,nsampsTest,2); % test stimulus 
Xtest = randn(nsampsTest,nw); % test stimulus 

% Generate weights
sigsmooth = 2.5;  % sigma for smoothing true weights
gfilt = normpdf((1:nw)',nw/2,sigsmooth);  % Gaussian smoothing filter
wts = real(ifft(fft(randn(nw,1)).*fft(gfilt)));  % generate random smooth weights

% Set up nonlinearity
nonlinearityTYPE = 2;  % (1 = softplus p=0.6, 2 = softplus p=2)
switch nonlinearityTYPE

    case 1 % ---- set nonlinearity to softplus p = 0.6 -------
    ptrue = .6;  % power
    fnlin = @(x)(log(1+exp(x)).^ptrue);

    % adjust model-specific params
    bias = 20; % Set bias
    wts = wts./norm(wts)*40; % set weight amplitude

    otherwise % ---- set nonlinearity to softplus p = 2 -------
    ptrue = 2;  % power
    fnlin = @(x)(log(1+exp(x)).^ptrue);

    % adjust model-specific params
    bias = 1.5; % Set bias
    wts = wts./norm(wts)*2; % set weight amplitude
end

% Simulate response (training data)
filteroutputTrain = (Xtrain*wts) + bias; % output of filter + bias
rateTrain = fnlin(filteroutputTrain);  % conditional intensity
spsTrain = poissrnd(rateTrain*dtbin);  % spike train

% Simulate response (test data)
filteroutputTest = (Xtest*wts) + bias; % test output of filter + bias
rateTest = fnlin(filteroutputTest);  % test conditional intensity
spsTest = poissrnd(rateTest*dtbin);  % test spike train


% ======  Make plots showing true params and spike rates =========

subplot(221);  % plot weights
ttw = 1:nw; % time bins for weights
plot(ttw, wts); 
box off; title('true weights'); xlabel('bins');

subplot(223);  % plot nonlinearity
xx = min(filteroutputTrain):.1:max(filteroutputTrain);
plot(xx,fnlin(xx));  box off; 
title('true nonlinearity'); xlabel('filter output'); ylabel('firing rate (sp/s)');

subplot(222);  % plot firing rate
tt = 1:min(nsampsTrain,1/dtbin); % time bins to plot
plot(tt*dtbin,rateTrain(tt)); box off; ylabel('rate (sp/s)'); title('firing rate');

subplot(224);  % plot spike counts
plot(tt*dtbin,spsTrain(tt));  box off;    
ylabel('spike count'); xlabel('time (s)'); title('spike counts');

fprintf('------Simulated dataset-------\n')
fprintf('number of bins: %d\n',nsampsTrain);
fprintf('max spike rate: %.1f sp/s\n',max(rateTrain));
fprintf('total spikes:  %d\n', sum(spsTrain));
fprintf('condition # of design matrix = %.1f\n',cond(Xtrain));

%% === 2. ML fit poisson GLM using true generalized-softplus nonlinearity with p=1 ====================

Xmattrain = [Xtrain, ones(nsampsTrain,1)];  % set design matrix for training set
Xmattest = [Xtest, ones(nsampsTest,1)];  % set design matrix for test set

% -- Set options --- 
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');
prs0 = randn(nw+1,1)*.1;  % initialize weights randomly
powfix = ptrue;  % fixed assumption of p

% -- Make loss function and minimize -----
nlfun = @(x)gnlzsoftplusfun(x,powfix);  % set nonlinearity
lossfun = @(prs)neglogli_PoissGLM(prs,Xmattrain,spsTrain,nlfun,dtbin); % negative log-li fun
prsML = fminunc(lossfun,prs0,opts);

w_ML = prsML(1:nw);  % ML estmate of weights given p=1
b_ML = prsML(nw+1);  % ML estimate of bias b given p=1


%% === 3. MAP fit with graph Laplacian regularization =========================================

Circflag = 1; % flag to indicate circular parameter vector
DCflag = 1; % flag to indicate adding a column and row of zeros, for dc weight
Cmat = mkPrecision_graphLapl_circ(nw,DCflag);  % make graph Laplacian matrix

% Set up grid of lambda values (regularization strength parameters)
lamvals = 2.^(-2:12); % it's common to use a log-spaced set of values
nlam = length(lamvals);

% Allocate space for train and test errors
LLtrain = zeros(nlam,1);  % training log-likelihood
LLtest = zeros(nlam,1);   % test log-likelihood
prs_MAP = zeros(nw+1,nlam); % parameters for each lambda

% Define train and test log-likelihood funcs
negLtrainfun = @(prs)neglogli_PoissGLM(prs,Xmattrain,spsTrain,nlfun,dtbin); 
negLtestfun  = @(prs)neglogli_PoissGLM(prs,Xmattest,spsTest,nlfun,dtbin); 

% Now compute MAP estimate for each ridge parameter
prsHat = prsML; % initialize parameter estimate with ML estimate
subplot(223); cla; plot(ttw,ttw*0,'k--','linewidth',1); hold on; % initialize plot
fprintf('\nSelecting regularization strength using grid of lambda values...\n\n');
for jj = 1:nlam

    % Compute ridge-penalized MAP estimate
    Cinv = lamvals(jj)*Cmat; % set inverse prior covariance
    lossfun = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    prsHat = fminunc(lossfun,prsHat,opts);
    
    % Compute negative logli
    LLtrain(jj) = -negLtrainfun(prsHat); % training loss
    LLtest(jj) = -negLtestfun(prsHat); % test loss
    
    % store the filter
    prs_MAP(:,jj) = prsHat;
    
    % plot it
    subplot(221);
    semilogx(lamvals(1:jj), LLtrain(1:jj)/nsampsTrain, '-o', lamvals(1:jj), LLtest(1:jj)/nsampsTest,'-*');
    box off; xlabel('lambda'); ylabel('log-likelihood per sample'); 
    legend('train', 'test', 'location', 'southwest'); 
    subplot(223);
    plot(ttw,prsHat(1:nw),'linewidth', 2); 
    title(['smoothing estimate: lambda = ', num2str(lamvals(jj))]);
    xlabel('time before spike (s)'); drawnow; 
end
[~,jjmax] = max(LLtest);
lambda = lamvals(jjmax);
prs_map1 = prs_MAP(:,jjmax);
w_map1 = prs_map1(1:nw);
b_map1 = prs_map1(nw+1);

subplot(221); 
hold on;
semilogx(lamvals(jjmax),LLtest(jjmax)/nsampsTest,'kd'); hold off;
title('Graph-Laplace: train and test log-likelihood');
subplot(223); 
plot(ttw,w_map1,'k--', 'linewidth', 3); hold off;
drawnow;

%% === 4. MAP fit with AR1 prior regularization =========================================

Circflag = 1; % flag to indicate circular parameter vector
DCflag = 1; % flag to indicate adding a column and row of zeros, for dc weight
rho = 10;  % marginal variance 

% Set up grid of lambda values (regularization strength parameters)
tauvals = 2.^(0:14); % it's common to use a log-spaced set of values
nlam = length(tauvals);

% Allocate space for train and test errors
LLtrain = zeros(nlam,1);  % training log-likelihood
LLtest = zeros(nlam,1);   % test log-likelihood
prs_MAP = zeros(nw+1,nlam); % parameters for each lambda

% Define train and test log-likelihood funcs
negLtrainfun = @(prs)neglogli_PoissGLM(prs,Xmattrain,spsTrain,nlfun,dtbin); 
negLtestfun  = @(prs)neglogli_PoissGLM(prs,Xmattest,spsTest,nlfun,dtbin); 

% Now compute MAP estimate for each ridge parameter
prsHat = prsML; % initialize parameter estimate with ML estimate
subplot(224); cla; plot(ttw,ttw*0,'k--','linewidth',1); hold on; % initialize plot
fprintf('\nSelecting regularization strength using grid of lambda values...\n\n');
for jj = 1:nlam

    % Compute ridge-penalized MAP estimate
    tau = tauvals(jj); 
    Cinv = mkPrecision_AR1prior_circ(tau,rho,nw,DCflag); % set inverse prior covariance
        lossfun = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    prsHat = fminunc(lossfun,prsHat,opts);
    
    % Compute negative logli
    LLtrain(jj) = -negLtrainfun(prsHat); % training loss
    LLtest(jj) = -negLtestfun(prsHat); % test loss
    
    % store the filter
    prs_MAP(:,jj) = prsHat;
    
    % plot it
    subplot(222);
    semilogx(tauvals(1:jj), LLtrain(1:jj)/nsampsTrain, '-o', tauvals(1:jj), LLtest(1:jj)/nsampsTest,'-*');
    box off; xlabel('lambda'); ylabel('log-likelihood per sample'); 
    legend('train', 'test', 'location', 'southwest'); 
    subplot(224);
    plot(ttw,prsHat(1:nw),'linewidth', 2); 
    title(['smoothing estimate: lambda = ', num2str(tauvals(jj))]);
    xlabel('time before spike (s)'); drawnow; 
end
[~,jjmax] = max(LLtest);
prs_map2 = prs_MAP(:,jjmax);
w_map2 = prs_map2(1:nw);
b_map2 = prs_map2(nw+1);

subplot(222); 
hold on;
semilogx(tauvals(jjmax),LLtest(jjmax)/nsampsTest,'kd'); hold off;
title('AR1: train and test log-likelihood');
subplot(224); 
plot(ttw,w_map2,'k--', 'linewidth', 3); hold off;
drawnow;


%% ==== 5. Make figs and print results ============

% Report result
fprintf('\n------ Fitting Results -------\n')
fprintf('True vs ML:       corr coeff= %.2f\n', corr(wts,w_ML));
fprintf('True vs MAP-GL:   corr coeff= %.2f\n', corr(wts,w_map1));
fprintf('True vs MAP-AR1:  corr coeff= %.2f\n', corr(wts,w_map2));
fprintf('\n    b true: %.2f\n',bias);
fprintf('b ML: %.2f\n',b_ML);
fprintf('b MAP-GL: %.2f\n',b_map1);
fprintf('b MAP-AR1: %.2f\n',b_map2);

% % Make plots
% subplot(224);
% ttw = 1:nw;
% plot(ttw,wts./norm(wts),ttw,w_map1./norm(w_map1), ttw,wMLjoint./norm(wMLjoint));
% axis tight;
% xlabel('time bin'); ylabel('coefficient');
% title('normalized true and inferred weights');
% legend('true','MAP(p=1)','MAP');

% fprintf('\ntrue vs inferred power p:  %.2f, %.2f\n', ptrue,powML);

