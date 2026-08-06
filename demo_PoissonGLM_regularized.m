% demo_PoissonGLM_regularized.m
%
% Simulate and recover weights from a Poisson GLM generalized-softplus
% nonlinearity using regularization 

addpath nlfuns
addpath loglifuns
addpath utils

clear;  % clear memory

%% 1. Make simulated dataset =============

dtbin = .01; % bin size for representing time (s)

nw = 100;  % number of weights

% Create stimulus 
nsecTrain = 100;   % stimulus length 
nsampsTrain = round(nsecTrain/dtbin);  % number of bins for training data
nsampsTest = 1e6;  % pick large value for size of test set

Xtrain = make1Fstimuli(nw,nsampsTrain,2); % training stimulus 
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
subplot(224); cla; plot(ttw,ttw*0,'k--','linewidth',1); hold on; % initialize plot
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
    subplot(222);
    semilogx(lamvals(1:jj), LLtrain(1:jj)/nsampsTrain, '-o', lamvals(1:jj), LLtest(1:jj)/nsampsTest,'-*');
    box off; xlabel('lambda'); ylabel('log-likelihood per sample'); 
    legend('train', 'test', 'location', 'southwest'); 
    subplot(224);
    plot(ttw,prsHat(1:nw),'linewidth', 2); 
    title(['smoothing estimate: lambda = ', num2str(lamvals(jj))]);
    xlabel('time before spike (s)'); drawnow; 
end
[~,jjmax] = max(LLtest);
lambda = lamvals(jjmax);
prs_map1 = prs_MAP(:,jjmax);
w_map1 = prs_map1(1:nw);
b_map1 = prs_map1(nw+1);

subplot(222); 
hold on;
semilogx(lamvals(jjmax),LLtest(jjmax)/nsampsTest,'kd'); hold off;
title('train and test log-likelihood');
subplot(224); 
plot(ttw,w_map1,'k--', 'linewidth', 3); hold off;
drawnow;

%% === 4. Fit GLM weights for a grid of p values w/ fixed regularization ====================

% Set grid of p values to consider
pgrid = 0.3:.1:4;
npgrid = length(pgrid);

Cinv = lambda*Cmat;  % set regularization matrix

prsHat = zeros(nw+1,npgrid); % store fit at each p value
testLLgivenp = zeros(npgrid,1); % test LL
prs0 = prs_map1; % initialize weights from exp fit
fprintf('----- fitting weights for grid over p ------ \n');
for jj = 1:npgrid

    % -- Make loss function and minimize -----
    pval = pgrid(jj);
    nlfun = @(x)gnlzsoftplusfun(x,pval);  % set nonlinearity

    % Define train and test log-likelihood funcs
    negLtrainfun = @(prs)neglogli_PoissGLM(prs,Xmattrain,spsTrain,nlfun,dtbin);
    negLtestfun  = @(prs)neglogli_PoissGLM(prs,Xmattest,spsTest,nlfun,dtbin);

    % Define negative log posterior 
    lossfun = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    prsMAPgivenp = fminunc(lossfun,prs0,opts);  % find MAP weights for this value of p

    
    prsHat(:,jj) = prsMAPgivenp;
    testLLgivenp(jj) = -negLtestfun(prsMAPgivenp);

    % initialize next fit from current fit
    prs0 = prsMAPgivenp;
    if mod(jj,1)==0
        fprintf('(iter %d, p=%.1f): neglogli = %.2f\n', jj,pval,testLLgivenp(jj));
    end
end

%%

% Select p using the maximum of test log-likelihood
[~,jjmax] = max(testLLgivenp);  % find minimal value of loss
wMLjoint = prsHat(1:nw,jjmax);  % extract weights
bMLjoint = prsHat(nw+1,jjmax);  % extract bias b
powML = pgrid(jjmax);  % extract power p

subplot(222);
plot(pgrid,testLLgivenp,'-o',powML,testLLgivenp(jjmax),'k*'); box off; title('test log-likelihood vs power p');
xlabel('power p'); ylabel('negative log-li');


%% ==== 5. Make figs and print results ============

% Report result
fprintf('\n------ Fitting Results -------\n')
fprintf('True vs softplus (p=1):  corr coeff= %.2f\n', corr(wts,w_map1));
fprintf('   True vs joint:  corr coeff= %.2f\n', corr(wts,wMLjoint));
fprintf('\n    b true: %.2f\n',bias);
fprintf('b ML (p=1): %.2f\n',bMLp1);
fprintf('   b-joint: %.2f\n',bMLjoint);

% Make plots
subplot(224);
ttw = 1:nw;
plot(ttw,wts./norm(wts),ttw,w_map1./norm(w_map1), ttw,wMLjoint./norm(wMLjoint));
axis tight;
xlabel('time bin'); ylabel('coefficient');
title('normalized true and inferred weights');
legend('true','MAP(p=1)','MAP');

fprintf('\ntrue vs inferred power p:  %.2f, %.2f\n', ptrue,powML);

