function [recParams, rec_nll, rec_pR2] = paramRecovery(simParams, fitParams)


if ~isfield(fitParams,'linear_env')
    fitParams.linear_env = 1; % Set default value for linear_env if not provided
end

if ~isfield(fitParams,'rnd_flag')
    fitParams.rnd_flag = 0; % Set default value for rnd_flag if not provided
end


% simulate choices and rewards
[choices, rewards] = simAlgos(simParams.algo, simParams.params, simParams.n_rounds, simParams.n_trials, simParams.rprobset);

narms = size(simParams.rprobset,2);


% get best fit params
[rec_nll, rec_pR2, recParams] = crossValFit(fitParams.algo, fitParams.kfolds, fitParams.iters, fitParams.param_bnds, ...
                choices, rewards, narms, fitParams.optim, fitParams.linear_env, fitParams.rnd_flag);