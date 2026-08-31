% demo_PoissonGLM.m
%
% Simulate and recover weights from a Poisson GLM under an exponential or
% generalized-softplus nonlinearity 

addpath nlfuns
addpath loglifuns
clear;  % clear memory

%% 1. Make simulated dataset =============

dtbin = .01; % bin size for representing time (s)

nw = 100;  % number of weights

% Create stimulus 
nsec = 1000;   % stimulus length 
nsamps = round(nsec/dtbin);  % number of bins
X = randn(nsamps,nw); % stimulus 

% Generate weights
sigsmooth = 2.5;  % sigma for smoothing true weights
gfilt = normpdf((1:nw)',nw/2,sigsmooth);  % Gaussian smoothing filter
wts = real(ifft(fft(randn(nw,1)).*fft(gfilt)));  % generate random smooth weights
wts = wts./norm(wts); % normalize weights to be unit vector

% Set up nonlinearity
nonlinearityTYPE = 2;  % (0 = exp, 1 = softplus p=0.6, 2 = softplus p=2)
switch nonlinearityTYPE
    case 0   % ---- set nonlinearity to exponential ---------
        fnlin = @exp; % set nonlinearity to exp

        % adjust model-specific params
        bias = 1.3;  % Set bias
        ptrue = 1; % not used, but set this to default so we don't get an error

    case 1 % ---- set nonlinearity to softplus p = 0.6 -------
    ptrue = .6;  % power
    fnlin = @(x)(log(1+exp(x)).^ptrue);

    % adjust model-specific params
    bias = 20; % Set bias
    wts = wts./norm(wts)*40; % set weights to be larger in amplitude

    otherwise % ---- set nonlinearity to softplus p = 2 -------
    ptrue = 2;  % power
    fnlin = @(x)(log(1+exp(x)).^ptrue);

    % adjust model-specific params
    bias = 2.5; % Set bias
    wts = wts./norm(wts)*2; % set weights to be larger in amplitude
end

% Simulate response
filteroutput = (X*wts) + bias; % output of filter + bias
rate = fnlin(filteroutput);  % conditional intensity
sps = poissrnd(rate*dtbin);  % spike train

% ======  Make plots showing true params and spike rates =========

subplot(221);  % plot weights
plot(wts); 
box off; title('true weights'); xlabel('bins');

subplot(223);  % plot nonlinearity
xx = min(filteroutput):.1:max(filteroutput);
plot(xx,fnlin(xx));  box off; 
title('nonlinearity'); xlabel('filter output'); ylabel('firing rate (sp/s)');

subplot(222);  % plot firing rate
tt = 1:min(nsamps,1/dtbin); % time bins to plot
plot(tt*dtbin,rate(tt)); box off; ylabel('rate (sp/s)'); title('firing rate');
subplot(224);  % plot spike counts
plot(tt*dtbin,sps(tt));  box off;    
ylabel('spike count'); xlabel('time (s)'); title('spike counts');

fprintf('------ Simulated dataset -------\n')
fprintf('number of bins: %d\n',nsamps);
fprintf('max spike rate: %.1f sp/s\n',max(rate));
fprintf('total spikes:  %d\n\n', sum(sps));


%% === 2. Fit poisson GLM using exponential nonlinearity ====================

Xmat = [X, ones(nsamps,1)];  % design matrix

% -- Set options --- 
opts = optimoptions('fminunc','algorithm','trust-region','SpecifyObjectiveGradient',true,'HessianFcn','objective','display','off');

% -- Make loss function and minimize -----
lossfunexp = @(w)neglogli_poissGLM_expnlin(w,Xmat,sps,dtbin); % set negative log-likelihood as loss func
prs0 = randn(nw+1,1)*.1;  % initialize weights randomly
tic; prsMLexp = fminunc(lossfunexp,prs0,opts); toc;

wMLexp = prsMLexp(1:nw);  % ML estimate of weights under exp nonlinearity
bMLexp = prsMLexp(nw+1);  % ML estimate of bias b under exp nonlinearity

%% === 3. Fit poisson GLM using true generalized-softplus nonlinearity ====================

% -- Make loss function and minimize -----
nlfun = @(x)gnlzsoftplusfun(x,ptrue);  % set nonlinearity
lossfunsp = @(w)neglogli_PoissGLM(w,Xmat, sps,nlfun,dtbin); % negative log-li fun
prs0 = prsMLexp; % initialize weights from exp fit
%prs0 = [wts;bias]; % initialize weights from ground truth
tic; prsMLsp = fminunc(lossfunsp,prs0,opts); toc;

wMLsp = prsMLsp(1:nw); % ML estimate of weights under (true) softplus nonlinearity
bMLsp = prsMLsp(nw+1); % ML estimate of bias under (true) softplus nonlinearity

%% === 4. Fit poisson GLM using generalized-softplus nonlinearity for a grid of p values ====================

% Set grid of values for power p
pgrid = 0.3:.1:4;
npgrid = length(pgrid);

prsHat = zeros(nw+1,npgrid);  % estimated params for each p
testLLgrid = zeros(npgrid,1);  % test log-likelihood 
prs0 = prsMLexp; % initialize weights from exp fit
fprintf('\n----- fitting weights for grid over p ------ \n');
for jj = 1:npgrid

    % -- Make loss function and minimize -----
    pval = pgrid(jj);
    nlfun = @(x)gnlzsoftplusfun(x,pval);  % set nonlinearity
    lossfunsp = @(w)neglogli_PoissGLM(w,Xmat, sps,nlfun,dtbin); % negative log-li fun
    prsML = fminunc(lossfunsp,prs0,opts);
    prsHat(:,jj) = prsML;
    testLLgrid(jj) = -lossfunsp(prsML);

    % initialize next fit from current fit
    prs0 = prsML;
    if mod(jj,5)==0
        fprintf('(iter %d, p=%.1f): neglogli = %.2f\n', jj,pval,testLLgrid(jj));
    end
end
  

% Select p using the minimum of negative log-likelihood
[~,jjmax] = max(testLLgrid);  % find minimal value of loss
wMLjoint = prsHat(1:nw,jjmax);  % extract weights
bMLjoint = prsHat(nw+1,jjmax);  % extract bias b
powML = pgrid(jjmax);  % extract power p

%% ==== 5. Make figs and print results ============

% Report result
fprintf('\n------ Fitting Results -------\n')
fprintf('     True vs exp:  corr coeff= %.2f\n', corr(wts,wMLexp));
fprintf('True vs softplus:  corr coeff= %.2f\n', corr(wts,wMLsp));
fprintf('   True vs joint:  corr coeff= %.2f\n', corr(wts,wMLjoint));
fprintf('\n    b true: %.2f\n',bias);
fprintf('     b-exp: %.2f\n',bMLexp);
fprintf('b-softplus: %.2f\n',bMLsp);
fprintf('   b-joint: %.2f\n',bMLjoint);

subplot(222);
plot(pgrid,testLLgrid,'-o',pgrid(jjmax),testLLgrid(jjmax),'k*'); box off; title('log-likelihood vs power p');
xlabel('power p'); ylabel('log-li');


% Make plots
subplot(224);
ttw = 1:nw;
plot(ttw,wts./norm(wts),ttw,wMLexp./norm(wMLexp),ttw,wMLsp./norm(wMLsp));
axis tight;
xlabel('time bin'); ylabel('coefficient');
title('normalized true and inferred weights');
legend('true','ML-exp','ML-sp');

fprintf('\ntrue vs inferred power p:  %.2f, %.2f\n', ptrue,powML);

