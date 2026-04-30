function [recParams, simParams, rec_nll, rec_pR2, rec_aic, rec_bic] = paramRecovery(simParamSet, fitParamSet, n_reps)

if ~isfield(fitParamSet,'linear_env')
    fitParamSet.linear_env = 1; % Set default value for linear_env if not provided
end

if ~isfield(simParamSet,'linear_env')
    simParamSet.linear_env = 1; % Set default value for linear_env if not provided
end

if ~isfield(fitParamSet,'rnd_flag')
    fitParamSet.rnd_flag = 0; % Set default value for rnd_flag if not provided
end

if ~isfield(fitParamSet,'crossval')
    fitParamSet.crossval = 'kfold'; % Set default value for crossval if not provided
end

if ~isfield(simParamSet,'logspace')
    simParamSet.logspace = zeros(size(simParamSet.param_bnds(1,:))); % Set default value for logspace if not provided
end

if ~isfield(simParamSet,'rnd_seed')
    simParamSet.rnd_seed = 1; % Set default value for rnd_seed if not provided
end

n_arms = size(simParamSet.rprobset,2);
n_sim_params = size(simParamSet.param_bnds,2);
log_params = simParamSet.logspace == 1;


%% Sim loop

simParams = zeros(n_reps,n_sim_params);

rng(simParamSet.rnd_seed, 'twister');
rnd_nums = rand(n_reps,n_sim_params);

choices = zeros(simParamSet.n_rounds, simParamSet.n_trials, n_reps);
rewards = zeros(simParamSet.n_rounds, simParamSet.n_trials, n_reps);

% simulate choices and rewards over all reps
for rp = 1 : n_reps
    
    % Pick random params from uniform distribution within sim param bnds (either linear or log-spaced)
    % linear range
    if any(~log_params)
        simParams(rp,~log_params) = rnd_nums(rp,~log_params) ...
            .* (simParamSet.param_bnds(2,~log_params)-simParamSet.param_bnds(1,~log_params)) ...
            +  simParamSet.param_bnds(1,~log_params);
    end
    % log range
    if any(log_params)        
        simParams(rp,log_params) = exp(rnd_nums(rp,log_params) ...
            .* log(simParamSet.param_bnds(2,log_params)./simParamSet.param_bnds(1,log_params)) ...
            +  log(simParamSet.param_bnds(1,log_params)));
    end

    % simulate choices and rewards
    [choices(:,:,rp), rewards(:,:,rp)] = simAlgos(simParamSet.algo, simParams(rp,:), simParamSet.n_rounds, simParamSet.n_trials, simParamSet.rprobset, simParamSet.linear_env);
    
end

%% Fitting loop

n_rec_params = size(fitParamSet.param_bnds,2);
recParams = zeros(n_reps,n_rec_params);
rec_nll = zeros(n_reps,1);
rec_pR2 = zeros(n_reps,1);
rec_aic = zeros(n_reps,1);
rec_bic = zeros(n_reps,1);

% fitting loop over reps
for rp = 1 : n_reps
    disp(['Rep ' num2str(rp)]);
    [rec_nll(rp), rec_pR2(rp), recParams(rp,:), rec_bic(rp), rec_aic(rp)] = fitModel(fitParamSet, choices(:,:,rp), rewards(:,:,rp), n_arms);
end