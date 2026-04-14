function [recParams, simParams, rec_nll, rec_pR2, rec_aic, rec_bic] = paramRecovery(simParamSet, fitParamSet, n_reps, use_crossval)

if nargin < 4
    use_crossval = 0;
end

if ~isfield(fitParamSet,'linear_env')
    fitParamSet.linear_env = 1; % Set default value for linear_env if not provided
end

if ~isfield(fitParamSet,'rnd_flag')
    fitParamSet.rnd_flag = 0; % Set default value for rnd_flag if not provided
end

if ~isfield(simParamSet,'log_params')
    simParamSet.logspace = zeros(size(simParamSet.param_bnds(1,:))); % Set default value for logspace if not provided
end

n_arms = size(simParamSet.rprobset,2);
n_params = size(simParamSet.param_bnds,2);
log_params = simParamSet.logspace == 1;

simParams = zeros(n_reps,n_params);
recParams = zeros(n_reps,n_params);
rec_nll = zeros(n_reps,1);
rec_pR2 = zeros(n_reps,1);
rec_aic = zeros(n_reps,1);
rec_bic = zeros(n_reps,1);

% loop over reps
for rp = 1 : n_reps
    if rp/10 == fix(rp/10); disp(rp); end

    % Pick random params from uniform distribution within sim param bnds (either linear or log-spaced)
    % linear range
    if any(~log_params)
        simParams(rp,~log_params) = rand(1,sum(~log_params)) ...
            .* (simParamSet.param_bnds(2,~log_params)-simParamSet.param_bnds(1,~log_params)) ...
            +  simParamSet.param_bnds(1,~log_params);
    end
    % log range
    if any(log_params)        
        simParams(rp,log_params) = exp(rand(1,sum(log_params)) ...
            .* log(simParamSet.param_bnds(2,log_params)./simParamSet.param_bnds(1,log_params)) ...
            +  log(simParamSet.param_bnds(1,log_params)));
    end

    % simulate choices and rewards
    [choices, rewards] = simAlgos(simParamSet.algo, simParams(rp,:), simParamSet.n_rounds, simParamSet.n_trials, simParamSet.rprobset);
    
    if use_crossval == 0
        % get best fit params (no cross-val)
        [rec_nll(rp), rec_pR2(rp), recParams(rp,:), rec_aic(rp), rec_bic(rp)] = modelFit(fitParamSet.algo, fitParamSet.iters, ...
            fitParamSet.param_bnds, choices, rewards, n_arms, fitParamSet.optim, fitParamSet.linear_env, fitParamSet.rnd_flag);
    else
        % get best fit params (with cross-val)
        [rec_nll(rp), rec_pR2(rp), recParams(rp,:)] = crossValFit(fitParamSet.algo, fitParamSet.kfolds, fitParamSet.iters, ...
            fitParamSet.param_bnds, choices, rewards, n_arms, fitParamSet.optim, fitParamSet.linear_env, fitParamSet.rnd_flag);
    end
    disp(['Rep ' num2str(rp) ' of ' num2str(n_reps)]);
end