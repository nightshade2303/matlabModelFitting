function fL = computeFitnessLandscape(simParamSet, reps)
%COMPUTEFITNESSLANDSCAPE  Simulate a model at many parameter sets in one fixed
%   environment, keeping every choice and reward.
%
%   fL = computeFitnessLandscape(simParamSet, reps)
%
% Inputs
%   simParamSet  struct with fields
%     algo                algorithm name passed to simAlgosBlocked ('Qlearning', ...)
%     param_bnds          2 x n_params [lb; ub] (from modelParamBounds). Params
%                         with lb == ub are fixed at that value.
%     rprobset            n_rounds x n_trials x n_arms reward probabilities
%                         (from generateRProbSet). Every parameter set is
%                         simulated on this same environment.
%     n_rounds            number of sessions (must equal size(rprobset,1))
%     n_trials_per_block  trials per block
%     n_blocks_per_round  blocks per session
%     linear_env          (optional, default 1) 1 = arms on a line, 0 = circular
%     sampling            (optional, default 'rand') how free params are drawn:
%                         'rand' (uniform random), 'sobol' (scrambled Sobol),
%                         'lhs' (Latin hypercube)
%     param_seed          (optional, default 1) seed for drawing param sets
%     sim_seed            (optional, default 2) seed for the simulations. The
%                         generator is reset to this seed before every
%                         parameter set, so all sets see the same random
%                         numbers (common random numbers). Differences in
%                         reward between sets then come from the parameters,
%                         not from simulation noise.
%     use_parfor          (optional, default true) run the sets in parallel
%     param_transform     (optional, default all "none") 1 x n_params string
%                         array, 'none' | 'log' | 'logit' (see transformParams).
%                         Each free param is drawn uniformly in this space
%                         between its transformed bounds, e.g. 'log' gives
%                         log-uniform draws.
%     transform_offset    (optional, default 1e-3) c in transformParams
%   reps         either a scalar (number of parameter sets to draw) or an
%                n_sets x n_params matrix of parameter sets to simulate
%
% Output: struct fL with fields
%   simParams    n_sets x n_params parameter sets
%   free_params  indices of free (sampled) params
%   choices      n_sets x n_rounds x n_trials int8, chosen arm (1..n_arms)
%   rewards      n_sets x n_rounds x n_trials logical
%   sess_reward  n_sets x n_rounds mean reward in each session
%   mean_reward  n_sets x 1 mean reward over sessions
%   sem_reward   n_sets x 1 SEM of the session rewards
%   max_reward   n_rounds x 1 expected reward of always picking the best arm
%   chance_reward n_rounds x 1 expected reward of picking arms at random
%   plus the settings used (param_bnds, sampling, seeds, ...)

%% settings and defaults
defaults = struct('linear_env', 1, 'sampling', 'rand', 'param_seed', 1, ...
                  'sim_seed', 2, 'use_parfor', true, 'transform_offset', 1e-3);
for f = fieldnames(defaults)'
    if ~isfield(simParamSet, f{1})
        simParamSet.(f{1}) = defaults.(f{1});
    end
end

param_bnds = simParamSet.param_bnds;
n_params   = size(param_bnds, 2);
if ~isfield(simParamSet, 'param_transform')
    simParamSet.param_transform = repmat("none", 1, n_params);
end
param_transform  = string(simParamSet.param_transform);
transform_offset = simParamSet.transform_offset;
free_params = find(param_bnds(2,:) > param_bnds(1,:));

n_rounds  = simParamSet.n_rounds;
n_tpb     = simParamSet.n_trials_per_block;
n_blocks  = simParamSet.n_blocks_per_round;
n_trials  = n_tpb * n_blocks;
rprobset  = simParamSet.rprobset;

assert(size(rprobset,1) == n_rounds, ...
    'rprobset has %d sessions but n_rounds = %d', size(rprobset,1), n_rounds);
assert(size(rprobset,2) == n_trials, ...
    'rprobset has %d trials but n_trials_per_block*n_blocks_per_round = %d', ...
    size(rprobset,2), n_trials);

%% parameter sets
if isscalar(reps)
    n_sets = reps;
    rng(simParamSet.param_seed, 'twister');
    u = sampleUnitCube(n_sets, numel(free_params), simParamSet.sampling);
    simParams = repmat(param_bnds(1,:), n_sets, 1); % fixed params at their value
    for k = 1 : numel(free_params)
        p = free_params(k);
        % draw uniformly between the bounds in the param's transformed space
        t_bnds = transformParams(param_bnds(:,p), param_transform(p), transform_offset, 'forward');
        y = t_bnds(1) + u(:,k) * (t_bnds(2) - t_bnds(1));
        x = transformParams(y, param_transform(p), transform_offset, 'inverse');
        simParams(:, p) = min(max(x, param_bnds(1,p)), param_bnds(2,p)); % guard rounding at the bounds
    end
else
    simParams = reps;
    n_sets = size(simParams, 1);
    assert(size(simParams,2) == n_params, 'reps must have %d columns', n_params);
end

%% simulate
% copy struct fields to plain variables so parfor broadcasts only what it needs
algo       = simParamSet.algo;
linear_env = simParamSet.linear_env;
sim_seed   = simParamSet.sim_seed;

choices = zeros(n_sets, n_rounds, n_trials, 'int8');
rewards = false(n_sets, n_rounds, n_trials);

if simParamSet.use_parfor
    n_workers = Inf; % use the pool (starts one if none is open)
else
    n_workers = 0;   % run in serial on this worker
end

parfor (rp = 1 : n_sets, n_workers)
    rng(sim_seed, 'twister'); % common random numbers for every param set
    [c, r] = simAlgosBlocked(algo, simParams(rp,:), n_rounds, n_tpb, ...
                             rprobset, linear_env, n_blocks);
    choices(rp,:,:) = reshape(int8(c), 1, n_rounds, n_trials);
    rewards(rp,:,:) = reshape(r > 0, 1, n_rounds, n_trials);
end

%% summaries
sess_reward = mean(rewards, 3);                         % n_sets x n_rounds
fL = struct;
fL.algo          = algo;
fL.param_bnds    = param_bnds;
fL.free_params   = free_params;
fL.sampling      = simParamSet.sampling;
fL.param_transform  = param_transform;
fL.transform_offset = transform_offset;
fL.param_seed    = simParamSet.param_seed;
fL.sim_seed      = sim_seed;
fL.linear_env    = linear_env;
fL.n_sets        = n_sets;
fL.n_rounds      = n_rounds;
fL.n_trials      = n_trials;
fL.simParams     = simParams;
fL.choices       = choices;
fL.rewards       = rewards;
fL.sess_reward   = sess_reward;
fL.mean_reward   = mean(sess_reward, 2);
fL.sem_reward    = std(sess_reward, 0, 2) / sqrt(n_rounds);
fL.max_reward    = mean(max(rprobset, [], 3), 2);
fL.chance_reward = mean(mean(rprobset, 3), 2);

end

function u = sampleUnitCube(n, d, method)
% n points in the d-dimensional unit cube
switch lower(method)
    case 'rand'
        u = rand(n, d);
    case 'sobol'
        p = scramble(sobolset(d), 'MatousekAffineOwen');
        u = net(p, n);
    case 'lhs'
        u = lhsdesign(n, d);
    otherwise
        error('Unknown sampling method "%s" (use rand, sobol or lhs)', method);
end
end
