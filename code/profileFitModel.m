function profileFitModel(simRecPath, rep, outDir)
%profileFitModel Time the fit of a single param-recovery rep, broken down per BADS restart.
%   Simulates data exactly as paramRecovery does, then runs the same BADS calls as
%   fitModel ('none' crossval branch), recording per-restart wall time, function-evaluation
%   count, and how much of that time is spent inside calcNLL vs. BADS's own overhead.
%   Thread count is taken from SLURM_CPUS_PER_TASK (maxNumCompThreads).

addpath('./HelperFunctions/');
addpath(genpath('./bads-master/'));

n_threads = str2double(getenv('SLURM_CPUS_PER_TASK'));
if isnan(n_threads); n_threads = 1; end
maxNumCompThreads(n_threads);

load([simRecPath '.mat'], 'simParamSet', 'fitParamSet');
if ~isfield(simParamSet,'linear_env'); simParamSet.linear_env = 1; end
if ~isfield(fitParamSet,'linear_env'); fitParamSet.linear_env = 1; end
if ~isfield(simParamSet,'logspace'); simParamSet.logspace = zeros(size(simParamSet.param_bnds(1,:))); end
if ~isfield(simParamSet,'rnd_seed'); simParamSet.rnd_seed = 1; end

%% simulate reps 1..rep exactly as in paramRecovery (same global rng sequence)
n_arms = size(simParamSet.rprobset, ndims(simParamSet.rprobset)); % arms are the last dim
n_sim_params = size(simParamSet.param_bnds,2);
log_params = simParamSet.logspace == 1;
S = load([simRecPath '.mat'], 'n_reps');
rng(simParamSet.rnd_seed, 'twister');
rnd_nums = rand(S.n_reps, n_sim_params);
for rp = 1 : rep
    sp = zeros(1, n_sim_params);
    if any(~log_params)
        sp(~log_params) = rnd_nums(rp,~log_params) .* (simParamSet.param_bnds(2,~log_params)-simParamSet.param_bnds(1,~log_params)) + simParamSet.param_bnds(1,~log_params);
    end
    if any(log_params)
        sp(log_params) = exp(rnd_nums(rp,log_params) .* log(simParamSet.param_bnds(2,log_params)./simParamSet.param_bnds(1,log_params)) + log(simParamSet.param_bnds(1,log_params)));
    end
    [choices, rewards] = simAlgosBlocked(simParamSet.algo, sp, simParamSet.n_rounds, simParamSet.n_trials_per_block, simParamSet.rprobset, simParamSet.linear_env, simParamSet.n_blocks_per_round);
end
simParams = sp;

%% fit setup, mirroring fitModel
algo = fitParamSet.algo;
linear_env = fitParamSet.linear_env;
pb = fitParamSet.param_bnds;
lb = pb(1,:); ub = pb(2,:); plb = pb(3,:); pub = pb(4,:);
iters = fitParamSet.iters;
options = bads('defaults');
options.Display = 'off';

% calcNLL micro-benchmark at random plausible points
n_bench = 50;
bench_t = zeros(n_bench,1);
rng(12345, 'twister');
calcNLL(plb + (pub-plb).*rand(1,numel(plb)), algo, choices, rewards, n_arms, linear_env); % warm-up
for b = 1 : n_bench
    p = plb + (pub-plb).*rand(1,numel(plb));
    t0 = tic; calcNLL(p, algo, choices, rewards, n_arms, linear_env); bench_t(b) = toc(t0);
end

% timed NLL wrapper: accumulates time spent inside calcNLL
nll_time = 0; nll_calls = 0;
    function f = timedNLL(params)
        t1 = tic;
        f = calcNLL(params, algo, choices, rewards, n_arms, linear_env);
        nll_time = nll_time + toc(t1);
        nll_calls = nll_calls + 1;
    end

rng(simParamSet.rnd_seed + rep, 'twister'); % same per-rep seed as parfor paramRecovery
iter_wall = zeros(iters,1); iter_nll_time = zeros(iters,1); iter_calls = zeros(iters,1);
iter_funccount = zeros(iters,1); iter_fval = zeros(iters,1); iter_params = zeros(iters, numel(lb));
t_rep = tic;
for i = 1 : iters
    st = plb + (pub-plb).*rand(1,numel(plb));
    nll_time = 0; nll_calls = 0;
    t_it = tic;
    [x, fval, ~, out] = bads(@timedNLL, st, lb, ub, plb, pub, [], options);
    iter_wall(i) = toc(t_it);
    iter_nll_time(i) = nll_time; iter_calls(i) = nll_calls;
    iter_funccount(i) = out.funccount; iter_fval(i) = fval; iter_params(i,:) = x;
    fprintf('rep %d | iter %d/%d | %.1f s | %d evals | calcNLL %.1f s (%.0f%%) | nll %.2f\n', ...
        rep, i, iters, iter_wall(i), nll_calls, nll_time, 100*nll_time/iter_wall(i), fval);
end
rep_wall = toc(t_rep);

fprintf('\n=== rep %d, %d threads ===\n', rep, n_threads);
fprintf('total fit time: %.1f min\n', rep_wall/60);
fprintf('per restart: min %.1f / median %.1f / max %.1f s\n', min(iter_wall), median(iter_wall), max(iter_wall));
fprintf('evals per restart: min %d / median %d / max %d\n', min(iter_calls), median(iter_calls), max(iter_calls));
fprintf('calcNLL per call (bench): median %.2f ms, range %.2f-%.2f ms\n', 1e3*median(bench_t), 1e3*min(bench_t), 1e3*max(bench_t));
fprintf('share of time in calcNLL: %.0f%%  (rest = BADS overhead)\n', 100*sum(iter_nll_time)/sum(iter_wall));

if ~exist(outDir,'dir'); mkdir(outDir); end
save(fullfile(outDir, sprintf('profile_rep%d_thr%d.mat', rep, n_threads)), ...
    'rep','n_threads','simParams','rep_wall','iter_wall','iter_nll_time','iter_calls', ...
    'iter_funccount','iter_fval','iter_params','bench_t');
end
