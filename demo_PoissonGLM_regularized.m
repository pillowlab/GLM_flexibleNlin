% demo_PoissonGLM_regularized.m
%
% Simulate and recover weights from a Poisson GLM generalized-softplus
% nonlinearity using regularization 

addpath nlfuns
addpath loglifuns
clear;  % clear memory

%% 1. Make simulated dataset =============

dtbin = .01; % bin size for representing time (s)

nw = 20;  % number of weights

% Create stimulus 
nsec = 50;   % stimulus length 
nsampsTrain = round(nsec/dtbin);  % number of bins for training data
nsampsTest = 1e6;  % pick large value for size of test set
Xtrain = randn(nsampsTrain,nw); % training stimulus 
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
    wts = wts./norm(wts)*3; % set weight amplitude
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
title('nonlinearity'); xlabel('filter output'); ylabel('firing rate (sp/s)');

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


%% === 2. ML fit poisson GLM using true generalized-softplus nonlinearity with p=1 ====================

Xmattrain = [Xtrain, ones(nsampsTrain,1)];  % set design matrix for training set
Xmattest = [Xtest, ones(nsampsTest,1)];  % set design matrix for test set

% -- Set options --- 
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');
prs0 = randn(nw+1,1)*.1;  % initialize weights randomly
powfix = 1;  % fixed assumption of p

% -- Make loss function and minimize -----
nlfun = @(x)gnlzsoftplusfun(x,powfix);  % set nonlinearity
lossfun = @(prs)neglogli_PoissGLM(prs,Xmattrain,spsTrain,nlfun,dtbin); % negative log-li fun
prsML = fminunc(lossfun,prs0,opts);

wMLp1 = prsML(1:nw);  % ML estmate of weights given p=1
bMLp1 = prsML(nw+1);  % ML estimate of bias b given p=1

%% === 3. MAP fit with p=1 across diff levels of regularization =========================================

diagvals = ones(nw,1)*[-1 2 -1];  % values needed for graph laplacian matrix
CgraphLapl = spdiags(diagvals, -1:1, nw,nw);
CgraphLapl(1,nw) = -1; % fix corners
CgraphLapl(nw,1) = -1; % fix corners
Cmat = [CgraphLapl, sparse(nw,1); sparse(1,nw+1)];  % add row & column of zeros for bias term

% Set up grid of lambda values (regularization strength parameters)
lamvals = 2.^(-5:10); % it's common to use a log-spaced set of values
nlam = length(lamvals);

% Allocate space for train and test errors
LLtrain = zeros(nlam,1);  % training log-likelihood
LLtest = zeros(nlam,1);   % test log-likelihood
prs_reg = zeros(nw+1,nlam); % parameters for each lambda

% Define train and test log-likelihood funcs
negLtrainfun = @(prs)neglogli_PoissGLM(prs,Xmattrain,spsTrain,nlfun,dtbin); 
negLtestfun  = @(prs)neglogli_PoissGLM(prs,Xmattest,spsTest,nlfun,dtbin); 

% Now compute MAP estimate for each ridge parameter
prsMAP = prsML; % initialize parameter estimate with ML estimate
clf; subplot(212); plot(ttw,ttw*0,'k--','linewidth',1); hold on; % initialize plot
for jj = 1:nlam

    % Compute ridge-penalized MAP estimate
    Cinv = lamvals(jj)*Cmat; % set inverse prior covariance
    lossfun = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    prsMAP = fminunc(lossfun,prsMAP,opts);
    
    % Compute negative logli
    LLtrain(jj) = -negLtrainfun(prsMAP); % training loss
    LLtest(jj) = -negLtestfun(prsMAP); % test loss
    
    % store the filter
    prs_reg(:,jj) = prsMAP;
    
    % plot it
    subplot(211);
    semilogx(lamvals(1:jj), LLtrain(1:jj)/nsampsTrain, '-o', lamvals(1:jj), LLtest(1:jj)/nsampsTest,'-*');
    box off; xlabel('lambda'); ylabel('log-likelihood per sample'); 
    legend('train', 'test', 'location', 'northwest'); 
    subplot(212);
    plot(ttw,prsMAP(1:nw),'linewidth', 2); 
    title(['smoothing estimate: lambda = ', num2str(lamvals(jj))]);
    xlabel('time before spike (s)'); drawnow; 
end
[~,jjmax] = max(LLtest);
subplot(211); 
hold on;
semilogx(lamvals(jjmax),LLtest(jjmax)/nsampsTest,'kd'); hold off;
title('train and test log-likelihood');
subplot(212); 
plot(ttw,prs_reg(1:nw,jjmax),'k--', 'linewidth', 3);


% %% ==== 5. Make figs and print results ============
% 
% % Report result
% fprintf('\n------ Fitting Results -------\n')
% fprintf('True vs softplus (p=1):  corr coeff= %.2f\n', corr(wts,wMLp1));
% %fprintf('   True vs joint:  corr coeff= %.2f\n', corr(wts,wMLjoint));
% fprintf('\n    b true: %.2f\n',bias);
% fprintf('b ML (p=1): %.2f\n',bMLp1);
% %fprintf('   b-joint: %.2f\n',bMLjoint);
% 
% % Make plots
% subplot(224);
% ttw = 1:nw;
% plot(ttw,wts./norm(wts),ttw,wMLp1./norm(wMLp1));
% axis tight;
% xlabel('time bin'); ylabel('coefficient');
% title('normalized true and inferred weights');
% legend('true','ML(p=1)');
% 
% %fprintf('\ntrue vs inferred power p:  %.2f, %.2f\n', ptrue,powML);
% 
