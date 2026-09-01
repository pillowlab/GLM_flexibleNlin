% demo3_PoissonGLM_regularizeThenFitNonlin.m
%
% Set regularization strength via cross-validation, then estimate nonlinearity parameter p with this
% regularization level 
% 
% Note: illustrates systematic bias toward large p when dataset size is small

setpath;  % add necessary paths
clear;  % clear memory

%% 1. Make simulated dataset =============

dtbin = .01; % bin size for representing time (s)

nw = 100;  % number of weights

% Create stimulus 
nsecTrain = 10;   % stimulus length 
nsampsTrain = round(nsecTrain/dtbin);  % number of bins for training data
nsampsTest = 1e6;  % pick large value for size of test set

IIDTRAIN = 1;  % flag to set training data
if IIDTRAIN
    % Use Gaussian white noise for train
    Xtrain = randn(nsampsTrain,nw); % training stimulus
else
    % Use correlated 1/F stimuli for training
    freqpow = 2;  % exponent for 1/f^p power 
    Xtrain = make1Fstimuli(nw,nsampsTrain,freqpow); % training stimulus
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
powfix = 1;  % fixed assumption of p

% -- Make loss function and minimize -----
nlfun = @(x)gnlzsoftplusfun(x,powfix);  % set nonlinearity
lossfun = @(prs)neglogli_PoissGLM(prs,Xmattrain,spsTrain,nlfun,dtbin); % negative log-li fun
prsML = fminunc(lossfun,prs0,opts);

w_ML = prsML(1:nw);  % ML estmate of weights given p=1
b_ML = prsML(nw+1);  % ML estimate of bias b given p=1


%% === 3. MAP fit with p=1 across diff levels of regularization =========================================

DCflag = 1; % flag to indicate adding a column and row of zeros, for dc weight
L = mkPrecision_graphLapl_circ(nw,DCflag);  % make graph Laplacian matrix

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
fprintf('\nSelecting regularization strength using grid of lambda values...\n\n');
for jj = 1:nlam

    % Compute ridge-penalized MAP estimate
    Cinv = lamvals(jj)*L; % set inverse prior covariance
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
    box off; xlabel('lambda'); ylabel('log-likelihood (per sample)'); 
    legend('train', 'test', 'location', 'southeast'); 
    subplot(224);
    plot(ttw,prsHat(1:nw),'linewidth', 2); 
    title(['smoothing estimate: lambda = ', num2str(lamvals(jj))]);
    xlabel('time before spike (s)'); drawnow; 
end
[~,jjmax] = max(LLtest);
lambda = lamvals(jjmax);
prs_map1 = prs_MAP(:,jjmax);
w_MAP1 = prs_map1(1:nw);
b_MAP1 = prs_map1(nw+1);

subplot(221); 
hold on;
semilogx(lamvals(jjmax),LLtest(jjmax)/nsampsTest,'kd'); hold off;
title('train and test log-likelihood');
subplot(224); 
plot(ttw,w_MAP1,'k--', 'linewidth', 3); hold off;
drawnow;

%% === 4. Fit GLM weights for a grid of p values w/ fixed regularization ====================

% Set grid of p values to consider
pgrid = .5:.1:5;
npgrid = length(pgrid);

Cinv = lambda*L;  % set regularization matrix

prsHat = zeros(nw+1,npgrid); % store fit at each p value
trainLLgivenp = zeros(npgrid,1); % test LL
prs0 = prs_map1; % initialize weights from exp fit
fprintf('Fitting weights for a grid of p values ...\n');
for jj = 1:npgrid

    % -- Make loss function and minimize -----
    pval = pgrid(jj);
    nlfun = @(x)gnlzsoftplusfun(x,pval);  % set nonlinearity

    % Define train and test log-likelihood funcs
    negLtrainfun = @(prs)neglogli_PoissGLM(prs,Xmattrain,spsTrain,nlfun,dtbin);

    % Define negative log posterior 
    lossfun = @(prs)neglogposterior(prs,negLtrainfun,Cinv);
    prsMAPgivenp = fminunc(lossfun,prs0,opts);  % find MAP weights for this value of p

    prsHat(:,jj) = prsMAPgivenp;
    trainLLgivenp(jj) = -lossfun(prsMAPgivenp);

    % initialize next fit from current fit
    prs0 = prsMAPgivenp;
    if mod(jj,10)==0
        fprintf('(iter %d, p=%.1f): neglogli = %.2f\n', jj,pval,trainLLgivenp(jj));
    end
end

%%

% Select p using the maximum of test log-likelihood
[~,jjmax] = max(trainLLgivenp);  % find minimal value of loss
w_MAPfittedp = prsHat(1:nw,jjmax);  % extract weights
b_MAPfittedp = prsHat(nw+1,jjmax);  % extract bias b
powML = pgrid(jjmax);  % extract power p

subplot(222);
plot(pgrid,trainLLgivenp,'-o',powML,trainLLgivenp(jjmax),'k*'); box off; title('(training) log-likelihood vs power p');
xlabel('power p'); ylabel('log-likelihood');


%% ==== 5. Make figs and print results ============

% Report result
fprintf('\n------ Fitting Results -------\n')
fprintf('True vs ML  (softplus p=1): corr coeff= %.2f\n', corr(wts,w_ML));
fprintf('True vs MAP (softplus p=1): corr coeff= %.2f\n', corr(wts,w_MAP1));
fprintf('True vs MAP (fitted p):     corr coeff= %.2f\n', corr(wts,w_MAPfittedp));
fprintf('\n    b true: %.2f\n',bias);
fprintf('b ML (p=1): %.2f\n',b_ML);
fprintf('   b-joint: %.2f\n',b_MAPfittedp);

% Make plots
subplot(224);
ttw = 1:nw;
plot(ttw,wts./norm(wts),ttw,w_ML/norm(w_ML), ttw,w_MAP1./norm(w_MAP1), ttw,w_MAPfittedp./norm(w_MAPfittedp));
axis tight;
xlabel('time bin'); ylabel('coefficient');
title('normalized true and inferred weights');
legend('true','ML', 'MAP(p=1)','MAP (fitted p)');

fprintf('\ntrue vs inferred power p:  %.2f, %.2f\n', ptrue,powML);

