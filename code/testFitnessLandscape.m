% testFitnessLandscape.m
% Stage 1 of the fitness-landscape analysis: draw many parameter sets for one
% model, simulate each on the same set of sessions, and save every choice and
% reward to output/sim/. Run the sections in order.

%% paths

codedir = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/code/';
savedir = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/output/sim/';
addpath([codedir 'HelperFunctions/']);

%% settings

simModel = "Qdghl";

% environment
env       = 'unstr';  % 'unstr' (unstructured) or 'str' (structured)
pure      = false;    % false = environment as run in the task (see generateRProbSet)
n_arms    = 4;
n_rounds  = 100;      % sessions per parameter set
n_trials_per_block = 150;
n_blocks_per_round = 2;
env_seed  = 3;        % seed for generating the sessions

% parameter sets
n_sets     = 10000;
sampling   = 'rand';  % 'rand', 'sobol' or 'lhs'
param_seed = 1;       % seed for drawing parameter sets
sim_seed   = 2;       % seed for the simulations (same for every set)

simParamSet = struct;
simParamSet.model      = simModel;
simParamSet.env        = env;
simParamSet.pure       = pure;
simParamSet.linear_env = 1;
simParamSet.n_rounds   = n_rounds;
simParamSet.n_trials_per_block = n_trials_per_block;
simParamSet.n_blocks_per_round = n_blocks_per_round;
simParamSet.n_trials   = n_trials_per_block*n_blocks_per_round;
simParamSet.sampling   = sampling;
simParamSet.param_seed = param_seed;
simParamSet.sim_seed   = sim_seed;
simParamSet.env_seed   = env_seed;
simParamSet.use_parfor = true;

[simParamSet.algo, simParamSet.param_bnds, param_names, free_params] = modelParamBounds(char(simModel));
simParamSet.param_names = param_names;
free_params = find(free_params);

% sessions: every parameter set is simulated on these same sessions
rng(env_seed, 'twister');
[simParamSet.rprobset, simParamSet.is_unstr] = generateRProbSet(n_rounds, ...
    n_trials_per_block, n_arms, strcmp(env, 'unstr'), n_blocks_per_round, pure);

%% timing test: simulate a few sets and extrapolate

n_test = 20;
testSet = simParamSet;
testSet.use_parfor = false;
tic;
fL_test = computeFitnessLandscape(testSet, n_test);
t_test = toc;
fprintf('%d sets: %.1f s (%.2f s/set) -> ~%.1f CPU-min for %d sets\n', ...
    n_test, t_test, t_test/n_test, t_test/n_test*n_sets/60, n_sets);

%% run and save

savefolder = sprintf('%s_landscape_%s', char(datetime("now","Format","yyyyMMdd")), env);
savepath = fullfile(savedir, savefolder);
if exist(savepath, 'dir') ~= 7
    mkdir(savepath);
end
savefile = fullfile(savepath, sprintf('%s_%s_n%d.mat', simModel, sampling, n_sets));
if exist(savefile, 'file') == 2
    error('%s already exists; change the settings or delete it first', savefile);
end

% under Slurm: start a pool sized to the allocation (otherwise parfor opens
% the default pool, which may use more cores than were allocated)
n_workers = str2double(getenv('SLURM_CPUS_PER_TASK'));
if simParamSet.use_parfor && ~isnan(n_workers) && n_workers > 1 && isempty(gcp('nocreate'))
    pc = parcluster('Processes');
    % per-job storage dir so concurrent jobs don't clash in ~/.matlab
    job_storage = fullfile(tempdir, ['matlab_pool_' getenv('SLURM_JOB_ID')]);
    if ~exist(job_storage, 'dir'); mkdir(job_storage); end
    pc.JobStorageLocation = job_storage;
    pc.NumWorkers = n_workers;
    parpool(pc, n_workers);
end

tic;
fL = computeFitnessLandscape(simParamSet, n_sets);
fprintf('Simulated %d sets in %.1f min\n', n_sets, toc/60);

save(savefile, 'fL', 'simParamSet', '-v7.3');
fprintf('Saved %s\n', savefile);

%% quick look: top parameter sets vs all sets

% load(savefile, 'fL', 'simParamSet'); % to start from a saved run
top_pct = 1; % percent of best-performing sets to keep

[~, order] = sort(fL.mean_reward, 'descend');
n_top = max(1, round(fL.n_sets * top_pct/100));
top_idx = order(1:n_top);

fprintf('mean reward: best %.3f, top %g%% cutoff %.3f, median %.3f (max possible %.3f, chance %.3f)\n', ...
    fL.mean_reward(order(1)), top_pct, fL.mean_reward(top_idx(end)), median(fL.mean_reward), ...
    mean(fL.max_reward), mean(fL.chance_reward));
fprintf('typical SEM of a set''s mean reward: %.4f\n', median(fL.sem_reward));

h_rew = figure;
histogram(fL.mean_reward, 100); hold on;
xline(fL.mean_reward(top_idx(end)), 'r', sprintf('top %g%%', top_pct));
xlabel('mean reward'); ylabel('# parameter sets'); title(simModel);

h_par = figure;
tiledlayout('flow', 'TileSpacing', 'compact');
for p = free_params
    nexttile;
    edges = linspace(fL.param_bnds(1,p), fL.param_bnds(2,p), 21);
    histogram(fL.simParams(:,p), edges, 'Normalization', 'pdf'); hold on;
    histogram(fL.simParams(top_idx,p), edges, 'Normalization', 'pdf');
    xlabel(param_names(p));
end
legend({'all', sprintf('top %g%%', top_pct)});
sgtitle(sprintf('%s: params of top %g%% vs all sets', simModel, top_pct));

% in batch jobs (Slurm) nobody sees the figures, so save them next to the results
if batchStartupOptionUsed
    [~, fname] = fileparts(savefile);
    exportgraphics(h_rew, fullfile(savepath, [fname '_reward.png']));
    exportgraphics(h_par, fullfile(savepath, sprintf('%s_top%gpct_params.png', fname, top_pct)));
end
