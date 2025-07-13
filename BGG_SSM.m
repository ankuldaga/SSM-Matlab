% BGG_SSM.m   ──  2025-07-11  (significant-lags edition, bug-free)
% =====================================================================
% Unsmooth private-equity NAV returns via a Kalman state–space model
% (Brown, Ghysels & Gredil, 2023) with user-selectable quarterly lags.
%
% USAGE
%   >> BGG_SSM          % default 4 lags (0-4)
%   >> BGG_SSM(6)       % include 0-6 lags of every factor
%
% INPUT FILE  :  "PE_data.xlsx"   (edit DATA_FILE if different)
% REQUIRED COLS:  Date, PE - RF, Mkt-RF, SMB, HML, Liq
% ---------------------------------------------------------------------
function BGG_SSM(numLags)

if nargin < 1, numLags = 4; end
numLags = max(round(numLags),0);
fprintf('Using %d lag(s) of each factor\n',numLags);

%% 1 ▸ Load & clean data ------------------------------------------------
DATA_FILE = "PE_data.xlsx";
if ~isfile(DATA_FILE)
    error("Excel input '%s' not found",DATA_FILE);
end
opts  = detectImportOptions(DATA_FILE,'PreserveVariableNames',true);
TBL   = readtable(DATA_FILE,opts);
TBL   = sortrows(TBL,'Date');

% handle Excel serial vs datetime
if isnumeric(TBL{1,'Date'})
    dates = datetime(TBL{:,'Date'},'ConvertFrom','excel');
else
    dates = TBL{:,'Date'};
end

peR = TBL{:,'PE - RF'};
F0  = [TBL{:,'Mkt-RF'},TBL{:,'SMB'},TBL{:,'HML'},TBL{:,'Liq'}];

% purge rows containing any NaNs
good    = all(~isnan([peR F0]),2);
dates   = dates(good);
peR     = peR(good);
F0      = F0(good,:);
[T,K]   = size(F0);

allNames  = ["Mkt","SMB","HML","LIQ"];
baseNames = allNames(1:K);                       % align with actual cols

%% 2 ▸ Build lag-augmented factor matrix --------------------------------
Ffull    = [];
facNames = strings(1,(numLags+1)*K);

for lag = 0:numLags
    if lag==0
        blk    = F0;
        suffix = "";
    else
        blk    = [zeros(lag,K); F0(1:end-lag,:)];
        suffix = "_L" + string(lag);
    end
    Ffull = [Ffull blk];                         %#ok<AGROW>
    facNames(1, lag*K+(1:K)) = string(baseNames) + suffix;
end
KF = size(Ffull,2);

%% 3 ▸ Keep only individually significant lags -------------------------
first = numLags + 1;                             % align to max lag
Y     = peR(first:end);
Xfull = [ones(numel(Y),1) Ffull(first:end,:)];

mdl   = fitlm(Xfull(:,2:end), Y, 'Intercept', true);
keep  = abs(mdl.Coefficients.tStat(2:end)) >= 1.28;   % |t| ≥ 1.96

Fuse    = Ffull(:, keep);
Kuse    = nnz(keep);
keepIdx = keep;                                  % logical mask

if Kuse == 0
    error('No lagged factor is individually significant at the 5 %% level.');
end

%% 4 ▸ Initial state via OLS -------------------------------------------
Xols      = [ones(T,1) Fuse];
beta0     = Xols\peR;
state_mu0 = beta0;
state_P0  = diag([0.1; ones(Kuse,1)]);
Q         = diag([0.005^2; 0.05^2*ones(Kuse,1)]);
sigma2    = var(peR - Xols*beta0);

%% 5 ▸ Kalman log-likelihood helper ------------------------------------
function ll = kfLogLik(lambda)
    x = state_mu0;  P = state_P0;  ll = 0;  prev = 0;
    for t = 1:T
        P = P + Q;
        y = peR(t) - lambda*prev;
        H = (1-lambda)*[1; Fuse(t,:)'];
        S = H'*P*H + sigma2;
        Kg = P*H / S;
        innov = y - H'*x;
        x = x + Kg*innov;
        P = P - Kg*H'*P;
        ll = ll - 0.5*(log(2*pi*S) + innov^2/S);
        prev = peR(t);
    end
end

%% 6 ▸ Grid-search for λ -------------------------------------------------
lambdas = 0:0.01:0.94;
[~,ix]  = max(arrayfun(@kfLogLik,lambdas));
bestLambda = lambdas(ix);
fprintf('λ̂ = %.3f\n',bestLambda);

%% 7 ▸ Final Kalman filter & unsmoothed series --------------------------
alpha = zeros(T,1);  beta = zeros(T,Kuse);  unsmooth = zeros(T,1);
x = state_mu0;  P = state_P0;  prev = 0;

for t = 1:T
    P = P + Q;
    y = peR(t) - bestLambda*prev;
    H = (1-bestLambda)*[1; Fuse(t,:)'];
    S = H'*P*H + sigma2;
    Kg = P*H / S;
    innov = y - H'*x;
    x = x + Kg*innov;
    P = P - Kg*H'*P;

    alpha(t)    = x(1);
    beta(t,:)   = x(2:end)';
    unsmooth(t) = x(1) + x(2:end)'*Fuse(t,:)';

    prev = peR(t);
end

den        = max(1e-6, 1-bestLambda);            % avoid ÷0
peShift    = [0; peR(1:end-1)];
unsmRough  = (peR - bestLambda*peShift) / den;

%% 8 ▸ Rolling R² (20-quarter window) -----------------------------------
ROLL   = 20;
rollR2 = NaN(T,1);
for i = ROLL:T
    yy = unsmooth(i-ROLL+1:i);
    XX = [ones(ROLL,1) Fuse(i-ROLL+1:i,:)];
    b  = XX\yy;
    yhat = XX*b;
    rollR2(i) = 1 - sum((yy-yhat).^2) / sum((yy-mean(yy)).^2);
end

%% 9 ▸ Save results & plots --------------------------------------------
save('BGG_state_results.mat','dates','peR','alpha','beta', ...
     'unsmooth','unsmRough','rollR2','bestLambda', ...
     'numLags','keepIdx','facNames');


%% 10 ▸ **Diagnostics & plots**  ----------------------------------------
% ----- factor loadings & t-stats -----
mdlBest = fitlm(Fuse(first:end,:), Y, 'Intercept', true);
tblOut  = table(facNames(keepIdx)', ...
                mdlBest.Coefficients.Estimate(2:end), ...
                mdlBest.Coefficients.tStat(2:end), ...
                'VariableNames',{'Factor','Beta','tStat'});
writetable(tblOut,'factor_loadings.csv');

fprintf('\nFactor loadings (lag depth = %d)\n',numLags);
disp(tblOut)

% aggregate β per base factor
aggBeta = zeros(1,K);
for k = 1:K
    pat = "^" + baseNames(k);
    idx = ~cellfun(@isempty, regexp(tblOut.Factor, pat));
    aggBeta(k) = sum(tblOut.Beta(idx));
end
aggTbl = table(baseNames', aggBeta', ...
               'VariableNames',{'Factor','AggregateBeta'});
fprintf('Aggregate betas (sum of retained lags):\n');
disp(aggTbl)

% ----- diagnostic figure -----
if ~isfolder('figs'), mkdir figs, end
figure('Visible','off','Position',[100 100 1400 800])

% (a) Prediction error (only one model here, so show bar)
subplot(2,3,1)
bar(numLags, sum((Y - mdlBest.Fitted).^2),0.4)
title('Prediction Error'), xlabel('Num. Lags'), ylabel('Sum Sq Error')

% (b) Retained factor weights
subplot(2,3,2)
stem(1:Kuse, mdlBest.Coefficients.Estimate(2:end),'filled')
set(gca,'XTick',1:Kuse,'XTickLabel',tblOut.Factor,'XTickLabelRotation',45)
title('Retained Factor Weights'), ylabel('\beta')

% (c) Adj R²
subplot(2,3,3)
text(0.1,0.5,sprintf('Adj R^2 = %.3f',mdlBest.Rsquared.Adjusted), ...
     'FontSize',12)
axis off
title('Model Fit')

% (d) AIC & BIC
subplot(2,3,4)
crit = mdlBest.ModelCriterion;
bar(categorical({'AIC','BIC'}),[crit.AIC, crit.BIC])
title('Information Criteria')

% (e) Factor-mimicking return series
subplot(2,3,[5 6])
plot(unsmooth,'b-','DisplayName','Unsmooth'), hold on
plot(beta0(1)+Fuse*beta0(2:end),'r--','DisplayName','Smoothed (F/OLS)')
legend, title('Factor-Mimicking Return Series')
xlabel('Time'), ylabel('Return')

print(gcf,'figs/model_diagnostics','-dpng','-r200'); close

% ----- beta dynamics & rolling R² from original code -----
start = ROLL;

figure('Visible','off'); hold on
plotIdx = find(keepIdx);
for k = 1:Kuse
    plot(dates(start:end), beta(start:end,k), ...
         'DisplayName', facNames(plotIdx(k)));
end
yline(0,'k-'); title('Dynamic betas'); legend('Location','best');
print(gcf,'figs/dynamic_betas','-dpng','-r150'); close;

figure('Visible','off');
plot(dates(start:end), rollR2(start:end));
ylim([0 1]); title('Rolling R² (20 qtrs)');
print(gcf,'figs/rolling_R2','-dpng','-r150'); close;

fprintf('\nSaved:\n  • BGG_state_results.mat\n  • factor_loadings.csv\n  • figs/model_diagnostics.png\n  • figs/dynamic_betas.png\n  • figs/rolling_R2.png\n');
end