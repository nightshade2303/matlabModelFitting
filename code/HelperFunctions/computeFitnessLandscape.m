function [fitnessLandscape] = computeFitnessLandscape(simParamSet,reps)

if isfield(simParamSet,'rnd_seed')
    rng(simParamSet.rnd_seed, 'twister'); % set seed only if provided
end
if ~isfield(simParamSet,'linear_env')
    simParamSet.linear_env = 0; % Set default value for linear_env if not provided
end

param_bnds = simParamSet.param_bnds;
n_sim_params = size(simParamSet.param_bnds,2);

if isscalar(reps)
    n_reps = reps;
    % generate sim params
    rnd_nums = rand(n_reps,n_sim_params);
    simParams = zeros(n_reps,size(param_bnds,2));
    for rp = 1 : n_reps
        simParams(rp,:) = rnd_nums(rp,:).*(param_bnds(2,:)-param_bnds(1,:))+param_bnds(1,:);
    end
else
    % sim params have been provided
    simParams = reps;
    n_reps = size(reps,1);
end

% compute performance for each rep of simParams
avg_reward = zeros(n_reps,simParamSet.n_rounds);
% avg_perfindex = zeros(n_reps,simParamSet.n_rounds);
for rp = 1 : n_reps
    % simulate choices and rewards
    [choices, rewards, rprobs, ~] = simAlgosBlocked(simParamSet.algo, simParams(rp,:), ...
        simParamSet.n_rounds, simParamSet.n_trials, simParamSet.rprobset, simParamSet.linear_env);
    avg_reward(rp,:) = mean(rewards,2);
    % perfindex = permute(calcPerfIndex(permute(choices,[3,1,2]),permute(rprobs,[3,1,2])),[2,3,1]);
    % avg_perfindex(rp,:) = mean(perfindex,2);
end

fitnessLandscape = struct;
fitnessLandscape.param_bnds = param_bnds;
fitnessLandscape.free_params = find((param_bnds(2,:)-param_bnds(1,:)) > 0);
fitnessLandscape.simParams = simParams;
fitnessLandscape.n_samps = n_reps;
fitnessLandscape.n_reps = n_reps;
fitnessLandscape.reward = avg_reward;
% fitnessLandscape.perfindex = avg_perfindex;