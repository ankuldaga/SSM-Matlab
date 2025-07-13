% =========================================================================
% BGG_SSM.m  —  2025‑07‑12  (diagnostics‑plus **annotated** edition)
% =========================================================================
% PURPOSE
%   Implements the Brown‑Ghysels‑Gredil (2023) **unsmoothing** procedure for
%   private‑equity NAV returns via a Kalman state‑space model **in one file**.
%   The script deliberately stays self‑contained so students or colleagues can
%   walk through each transformation without chasing dependencies.
%
%   Compared with a vanilla BGG implementation, this version adds:
%     • **Flexible sample window** via `startDate` and `endDate`
%     • **User‑selectable factor‑lag depth** (`numLags`)
%     • A **profile log‑likelihood** search for the persistence parameter λ
%     • **Information criteria** (AIC/BIC)
%     • **One‑step‑ahead RMSE**, split 80 % in‑sample / 20 % OOS
%     • A **Rolling R²** diagnostic (20‑quarter window)
%     • Extensive **comments** so you can extend / debug quickly.
%
%   Output artefacts (MAT‑file, PNG figs, CSV) are written to ./results/.
%
% INPUT FILE
%   "PE_data.xlsx" — must contain at least these columns:
%       Date  |  PE - RF  |  Mkt-RF  |  SMB  |  HML  |  Liq
%   Modify `DATA_FILE` or the column list below if your naming is different.
%
% FUNCTION SIGNATURE / EXAMPLES
%   BGG_SSM()                               % <- full sample, 0‑4 lags
%   BGG_SSM(6)                              % <- full sample, 0‑6 lags
%   BGG_SSM([], sd, ed)                     % <- default 4 lags, custom dates
%   BGG_SSM(8, datetime(2000,1,1), datetime(2024,12,31))
%
% AUTHOR  :  SSM (ChatGPT)
% UPDATED :  2025‑07‑12
% =========================================================================
function BGG_SSM(numLags, startDate, endDate)

%% ---------------------------------------------------------------------
%% 0. ARGUMENT PARSING & SANITY CHECKS                                   
%% ---------------------------------------------------------------------
% If the caller omits arguments, fall back to sensible defaults
if nargin < 1 || isempty(numLags),  numLags   = 4;                end
if nargin < 2 || isempty(startDate),startDate = datetime(-Inf,1,1);end % earliest
if nargin < 3 || isempty(endDate),  endDate   = datetime( Inf,1,1);end % latest

numLags = max(round(numLags),0);          % enforce non‑negative integer
assert(startDate <= endDate, "startDate must not exceed endDate");

fprintf('[BGG_SSM] lags = %d | window = %s → %s\n', numLags, ...
        string(startDate), string(endDate));

%% ---------------------------------------------------------------------
%% 1. LOAD EXCEL DATA & BASIC CLEANING                                   
%% ---------------------------------------------------------------------
DATA_FILE = "PE_data.xlsx";              % <‑‑ change here if file differs

% Import with variable names untouched (so spaces/dashes survive)
opts = detectImportOptions(DATA_FILE,'PreserveVariableNames',true);
TBL  = readtable(DATA_FILE, opts);
TBL  = sortrows(TBL, 'Date');             % ensure chronological order

% ---- Handle Excel serial dates vs proper datetimes -------------------
if isnumeric(TBL{1,'Date'})                      % Excel serial numbers?
    dates = datetime(TBL{:,'Date'}, 'ConvertFrom','excel');
else                                            % already datetime/text
    dates = TBL{:,'Date'};
end

% ---- Pull target return and raw factor matrix ------------------------
peR = TBL{:,'PE - RF'};                          % PE excess return
F0  = [ TBL{:,'Mkt-RF'}, ...
        TBL{:,'SMB'}, TBL{:,'HML'}, TBL{:,'Liq'} ];

% ---- Apply user‑defined date window *once* ---------------------------
mask = (dates >= startDate) & (dates <= endDate);
[dates, peR, F0] = deal(dates(mask), peR(mask), F0(mask,:));

% ---- Drop rows with any NaNs to keep matrices tidy -------------------
good = all(~isnan([peR F0]), 2);
[dates, peR, F0] = deal(dates(good), peR(good), F0(good,:));

[T, K] = size(F0);                           % sample length, # base factors
allNames  = ["Mkt","SMB","HML","LIQ"];
baseNames = allNames(1:K);                       % align with actual cols

%% ---------------------------------------------------------------------
%% 2. BUILD LAG‑AUGMENTED FACTOR MATRIX                                  
%% ---------------------------------------------------------------------
% We stack contemporaneous and lagged factors:  [F_t, F_{t‑1}, …, F_{t‑L}].
% Lag 0 block is F0 ; lag>0 blocks are zero‑padded to maintain size T.
Ffull    = [];                                % big design matrix
facNames = strings(1, (numLags+1)*K);         % names for later plots/CSV

for lag = 0:numLags
    if lag == 0
        blk    = F0;                          % contemporaneous values
        suffix = "";
    else
        blk    = [zeros(lag, K); F0(1:end-lag,:)]; % shift down by `lag`
        suffix = "_L" + string(lag);          % e.g. "_L1", "_L2", …
    end
    Ffull = [Ffull blk];                      %#ok<AGROW> concatenate
    facNames(1, lag*K + (1:K)) = string(baseNames) + suffix;
end

%% ---------------------------------------------------------------------
%% 3. SCREEN LAGS BY INDIVIDUAL t‑STATISTIC                              
%% ---------------------------------------------------------------------
% Brown‑Ghysels‑Gredil keep only factor‑lags with |t| ≥ 1.96 in a simple
% OLS regression of the PE return on *all* lags. We replicate that.
first   = numLags + 1;               % first row with full lag history
Y_sig   = peR(first:end);
X_sig   = Ffull(first:end, :);
mdlSig  = fitlm(X_sig, Y_sig, 'Intercept', true);
keep    = abs(mdlSig.Coefficients.tStat(2:end)) >= 1.96; % logical mask
Fuse    = Ffull(:, keep);            % reduced matrix (only significant lags)
Kuse    = nnz(keep);                 % # retained predictors
if Kuse == 0
    error('No lagged factor is individually significant at the 5%% level.');
end

%% ---------------------------------------------------------------------
%% 4. INITIAL STATE (OLS SEED) & SYSTEM NOISE                            
%% ---------------------------------------------------------------------
X_ols       = [ones(T,1) Fuse];      % intercept + factors
beta0       = X_ols \ peR;           % OLS coefficients  (\ = pinv)
state_mu0   = beta0;                 % initial state mean  [α ; β]
state_P0    = diag([0.1 ; ones(Kuse,1)]); % diffuse but finite covariance
Q           = diag([0.005^2 ; 0.05^2 * ones(Kuse,1)]); % state noise var
sigma2      = var(peR - X_ols * beta0);    % measurement noise var

%% ---------------------------------------------------------------------
%% 5. GRID SEARCH FOR λ VIA PROFILE LOG‑LIKELIHOOD                       
%% ---------------------------------------------------------------------
%   y_t = λ * y_{t‑1} + (1‑λ) * (α_t + β_t' Fuse_t) + ε_t
%   α_t, β_t follow a random walk. We treat λ as fixed and choose it by
%   maximising the Kalman log‑likelihood over a dense grid.

% --- Helper: compute log‑likelihood for a given λ ---------------------
function ll = profileLL(lambda)
    x = state_mu0;  P = state_P0;  ll = 0;  prevObs = 0;      % initialise
    for t = 1:T
        % --- Time update ---------------------------------------------
        P = P + Q;                      % propagate state covariance

        % --- Innovation ---------------------------------------------
        y     = peR(t) - lambda * prevObs;      % demeaned measurement
        H     = (1 - lambda) * [1 ; Fuse(t,:)'];% design vector (α & β)
        S     = H' * P * H + sigma2;            % innovation variance

        % --- Kalman gain & state update -----------------------------
        K_g   = P * H / S;
        innov = y - H' * x;
        x     = x + K_g * innov;
        P     = P - K_g * H' * P;               % Joseph update skipped

        % --- Accumulate log‑likelihood ------------------------------
        ll    = ll - 0.5 * ( log(2*pi*S) + innov^2 / S );
        prevObs = peR(t);                       % store y_{t} for next step
    end
end

% --- Evaluate grid ----------------------------------------------------
lambdas = 0 : 0.01 : 0.94;                     % empirical range from BGG
logLVec = arrayfun(@profileLL, lambdas);
[~, ix] = max(logLVec);                        % locate maximum
bestLambda = lambdas(ix);
fprintf('λ̂ = %.3f (via profile LL)\n', bestLambda);

%% ---------------------------------------------------------------------
%% 6. FINAL KALMAN FILTER / SMOOTHER & PREDICTION RESIDUALS              
%% ---------------------------------------------------------------------
alpha  = zeros(T,1);           % time‑varying intercept α_t
beta   = zeros(T,Kuse);        % time‑varying factor loadings β_t
unsmooth = zeros(T,1);         % latent “true” PE excess return
resid  = zeros(T,1);           % one‑step‑ahead prediction error ε̂_t

x = state_mu0; P = state_P0; prevObs = 0;
for t = 1:T
    % --- Time update (random walk) -----------------------------------
    P = P + Q;

    % --- Innovation --------------------------------------------------
    y  = peR(t) - bestLambda * prevObs;
    H  = (1 - bestLambda) * [1 ; Fuse(t,:)'];
    S  = H' * P * H + sigma2;
    Kg = P * H / S;
    innov = y - H' * x;

    % --- Measurement update -----------------------------------------
    x = x + Kg * innov;            % filtered state mean
    P = P - Kg * H' * P;           % filtered covariance

    % --- Store outputs ----------------------------------------------
    alpha(t)    = x(1);
    beta(t,:)   = x(2:end)';
    unsmooth(t) = x(1) + x(2:end)' * Fuse(t,:)';

    % One‑step‑ahead fitted value and residual
    yhat     = bestLambda * prevObs + (1 - bestLambda) * unsmooth(t);
    resid(t) = peR(t) - yhat;

    prevObs  = peR(t);              % shift actual observation forward
end

%% ---------------------------------------------------------------------
%% 7. DIAGNOSTICS: RMSE, ROLLING R², IC                                  
%% ---------------------------------------------------------------------
% --- RMSE split (80 % in‑sample) --------------------------------------
split      = floor(0.8 * T);
rmse_in    = sqrt(mean(resid(1:split).^2));
rmse_oos   = sqrt(mean(resid(split+1:end).^2));

% --- Rolling R²: 20‑quarter (~5‑year) window --------------------------
ROLL  = 20;                      %# windows must be ≥ 20 obs
rollR2 = NaN(T,1);
for i = ROLL:T
    yy = unsmooth(i-ROLL+1:i);
    XX = [ones(ROLL,1) Fuse(i-ROLL+1:i,:)];
    b  = XX \ yy;                        % OLS fit inside window
    yhat = XX * b;
    rollR2(i) = 1 - sum( (yy - yhat).^2 ) / sum( (yy - mean(yy)).^2 );
end

% --- Information criteria for final model ----------------------------
Kparam = 1 + 1 + Kuse + 1;     % [α0, λ, β’s, σ²] -> simplistic count
AIC    = -2 * logLVec(ix) + 2 * Kparam;
BIC    = -2 * logLVec(ix) + Kparam * log(T);

%% ---------------------------------------------------------------------
%% 8. SAVE NUMBERS TO DISK                                               
%% ---------------------------------------------------------------------
% ---- timestamped Results folder ----
runStamp = datestr(now,'yyyymmdd_HHMMSS');   % e.g. 20250713_141530
OUT      = sprintf('results_%s', runStamp);
if ~isfolder(OUT), mkdir(OUT); end

FIG = fullfile(OUT,'figs');                  % figures sub-dir
if ~isfolder(FIG), mkdir(FIG); end

save(fullfile(OUT,'BGG_state_results.mat'), ...
     'dates','peR','unsmooth','bestLambda','rollR2', ...
     'rmse_in','rmse_oos','AIC','BIC','logLVec','lambdas', ...
     'numLags','startDate','endDate','keep','facNames');

%% ---------------------------------------------------------------------
%% 9. QUICK PLOTS                                                        
%% ---------------------------------------------------------------------
% 9.1  Profile log‑likelihood over λ -----------------------------------
fig1 = figure('Visible','off');
plot(lambdas, logLVec, '-o'); hold on
plot(bestLambda, logLVec(ix), 'rx', 'MarkerSize',10,'LineWidth',2);
xlabel('\lambda'); ylabel('Log‑likelihood'); grid on
	title('Profile Log‑Likelihood for \lambda');
print(fig1, fullfile(FIG,'profile_LL'), '-dpng','-r150'); close(fig1)

% 9.2  Rolling R² time‑series ------------------------------------------
fig2 = figure('Visible','off');
plot(dates, rollR2, 'LineWidth',1); ylim([0 1]); grid on
xlabel('Date'); ylabel('R^{2}');
	title(sprintf('Rolling R^{2} (%d quarters)', ROLL));
print(fig2, fullfile(FIG,'rolling_R2'), '-dpng','-r150'); close(fig2)

%% ---------------------------------------------------------------------
%% 10. CONSOLE SUMMARY                                                   
%% ---------------------------------------------------------------------
fprintf('\n=== SUMMARY =====================================================\n');
fprintf(' λ̂             : %.3f\n', bestLambda);
fprintf(' AIC / BIC      : %.1f / %.1f\n', AIC, BIC);
fprintf(' RMSE (in / OOS): %.4f | %.4f\n', rmse_in, rmse_oos);
fprintf(' Saved → %s\n',OUT);

function printTstats(tt,lags)
    % Text + CSV ------------------------------------------------------
    fprintf('\nT-statistics for all lags (L=%d)\n', lags);
    disp(tt)
    csvOut = sprintf(OUT, 'tstats_L%d.csv', lags);
    writetable(tt, csvOut);
    fprintf('Saved → %s\n', csvOut);

    % Figure ----------------------------------------------------------
    %figDir = 'results/figs';                    % static folder; always exists
    if ~isfolder(FIG), mkdir(FIG); end

    f = figure('Visible','off');
    bar(tt.tStat, 'FaceColor', [0.2 0.6 0.8]); hold on
    yline([-1.96 1.96], 'r--', 'LineWidth', 1);
    set(gca,'XTick',1:height(tt),'XTickLabel',tt.Factor, ...
            'XTickLabelRotation',45);
    ylabel('t-Statistic');
    title(sprintf('t-stats (L=%d)', lags));
    grid on

    print(f, fullfile(FIG, sprintf('tstats_L%d.png', lags)), ...
          '-dpng','-r150');
    close(f)
end

function [res, tTbl] = runOneLag(peR, F0, baseNames, numLags)
[T, K] = size(F0);
Ffull = []; facNames = strings(1,(numLags+1)*K);
for g = 0:numLags
    blk = (g==0).*F0 + (g>0).*[zeros(g,K);F0(1:end-g,:)];
    Ffull = [Ffull blk];
    facNames(1,g*K+(1:K)) = string(baseNames) + (g==0).*"" + (g>0).*( "_L" + string(g));
end

first = numLags + 1;
ySig = peR(first:end);
XSig = Ffull(first:end,:);
mdlSig = fitlm(XSig,ySig,'Intercept',true);
tStats = mdlSig.Coefficients.tStat(2:end);
tTbl = table(facNames', tStats, 'VariableNames',{'Factor','tStat'});

keep = abs(tStats) >= 1.96;
Fuse = Ffull(:, keep); Kuse = nnz(keep);
if Kuse == 0, error('No significant factor at lag %d', numLags); end

Xols = [ones(T,1) Fuse]; beta0 = Xols \ peR;
state_mu0 = beta0; state_P0 = diag([0.1; ones(Kuse,1)]);
Q = diag([0.005^2 ; 0.05^2 * ones(Kuse,1)]);
sigma2 = var(peR - Xols * beta0);

function ll = LL(lambda)
    x=state_mu0; P=state_P0; ll=0; prev=0;
    for t = 1:T
        P = P + Q; y = peR(t) - lambda*prev; H = (1 - lambda)*[1; Fuse(t,:)'];
        S = H'*P*H + sigma2; K = P*H/S; v = y - H'*x;
        x = x + K*v; P = P - K*H'*P; ll = ll - 0.5*(log(2*pi*S) + v^2/S);
        prev = peR(t);
    end
end

lambdas = 0:0.01:0.94; llvec = arrayfun(@LL, lambdas);
[~, ix] = max(llvec); lambda = lambdas(ix);

x = state_mu0; P = state_P0; resid = zeros(T,1); prev = 0;
for t = 1:T
    P = P + Q; y = peR(t) - lambda*prev; H = (1 - lambda)*[1; Fuse(t,:)'];
    S = H'*P*H + sigma2; K = P*H/S; v = y - H'*x;
    x = x + K*v; P = P - K*H'*P;
    yhat = lambda*prev + (1 - lambda)*(x(1) + x(2:end)' * Fuse(t,:)');
    resid(t) = peR(t) - yhat; prev = peR(t);
end

split = floor(0.8 * T);
rmse_in = sqrt(mean(resid(1:split).^2));
rmse_oos = sqrt(mean(resid(split+1:end).^2));
Kparam = 1 + 1 + Kuse + 1;
AIC = -2*llvec(ix) + 2*Kparam;
BIC = -2*llvec(ix) + Kparam*log(T);

res = struct('lambda', lambda, 'rmse_in', rmse_in, ...
             'rmse_oos', rmse_oos, 'AIC', AIC, 'BIC', BIC);
end
%[resultMain, tStatTable] = runOneLag(numLags);
end