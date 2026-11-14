% testFitnessLandscape.m
% Stage 1 of the fitness-landscape analysis: draw many parameter sets for one
% model, simulate each on the same set of sessions, and save every choice and
% reward to output/sim/. Run the sections in order.

%% paths

codedir = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/code/';
savedir = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/output/sim/';
fitroot = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/output/r_fits/';
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

% parameter bounds: 'model' = full modelParamBounds range,
%                   'fits'  = from Gaussian fits to the rats' fitted params:
%                             per param, the first of raw/log/logit under which
%                             the rats look normal (Anderson-Darling; raw if
%                             none), a Gaussian fit in that space, then
%                             mu +/- n_sd SD (see fitParamBounds)
bnds_source = 'fits';
fit_dirs = {[fitroot '20261003_allblock_unstr_rats/20261003_B_Qdghl/'], ... % ub rats
            [fitroot '20261003_allblock_str_rats/20261022_B_Qdghl/']};      % sb rats (pooled with ub)
cohort_names = {'ub', 'sb'};
fit_opts = struct;
fit_opts.max_nll_per_trial = log(n_arms); % exclude fits with nll <= 0 / NaN (nothing fit) or nll per trial >= this (chance)
bnd_opts = struct;
bnd_opts.alpha  = 0.05;           % normality test significance level
bnd_opts.n_sd   = 2;              % range = mu +/- n_sd SD
bnd_opts.offset = 1e-3;           % keeps log/logit finite for rats at 0 (or 1)
bnd_opts.fit_method = 'mle';      % Gaussian fit that sets the bounds: 'mle' (fitdist) or 'curve' (least squares on the histogram)
bnd_opts.n_bins = 8;              % histogram bins for the curve fit (and the plot)

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
simParamSet.model_bnds   = simParamSet.param_bnds;
simParamSet.bnds_source  = bnds_source;

switch bnds_source
    case 'model'
        % keep modelParamBounds
    case 'fits'
        % free params: from the pooled rats' fits; fixed params unchanged
        [fit_params, cohort, fit_info] = loadFitParams(fit_dirs, fit_opts);
        bad = fit_info(fit_info.excluded, :);
        for k = 1 : height(bad)
            fprintf('excluded: %s (%s, %s): %s\n', bad.rat(k), cohort_names{bad.cohort(k)}, bad.file(k), bad.reason(k));
        end
        simParamSet.fit_info = fit_info;
        model_bnds = simParamSet.model_bnds;
        [simParamSet.param_bnds, simParamSet.param_transform, bnds_info] = ...
            fitParamBounds(fit_params, model_bnds, free_params, bnd_opts);
        simParamSet.transform_offset = bnd_opts.offset;

        simParamSet.fit_dirs     = fit_dirs;
        simParamSet.cohort_names = cohort_names;
        simParamSet.fit_params   = fit_params;
        simParamSet.cohort       = cohort;
        simParamSet.bnds_info    = bnds_info;

        fprintf(['Gaussian fits to %d rats (%s). Space = first of raw/log/logit passing Anderson-Darling (alpha %g), raw if none.\n' ...
            'Bounds = mu +/- %g SD of the %s fit, in that space, clipped to the model bounds.\n'], ...
            size(fit_params,1), strjoin(compose('%s %d', string(cohort_names(:)), histcounts(cohort, 0.5:1:numel(fit_dirs)+0.5)'), ', '), ...
            bnd_opts.alpha, bnd_opts.n_sd, upper(bnd_opts.fit_method));
        fprintf('  %-11s %-6s %-11s %-22s %-36s %-32s %-5s %s\n', 'param', 'space', 'normal?', 'A-D p (none/log/logit)', ...
            'MLE mu [95% CI]   sd [95% CI]', 'curve mu [95% CI]   sd', 'R2', 'new bounds (raw)');
        for p = free_params
            pv = sprintf('%.3f ', bnds_info.pvals{p});
            if bnds_info.normal(p), nrm = 'yes'; else, nrm = 'NOT NORMAL'; end
            fprintf('  %-11s %-6s %-11s %-22s %6.3f [%6.3f %6.3f] %5.3f [%5.3f %5.3f]  %6.3f [%6.3f %6.3f] %5.3f  %5.2f [%7.3f, %7.3f]\n', ...
                param_names(p), bnds_info.space(p), nrm, pv, ...
                bnds_info.mle_mu(p), bnds_info.mle_mu_ci(:,p), bnds_info.mle_sd(p), bnds_info.mle_sd_ci(:,p), ...
                bnds_info.curve_mu(p), bnds_info.curve_mu_ci(:,p), bnds_info.curve_sd(p), bnds_info.curve_r2(p), ...
                simParamSet.param_bnds(1,p), simParamSet.param_bnds(2,p));
        end
        flagged = free_params(~isnan(bnds_info.curve_mu(free_params)) & ~bnds_info.curve_ok(free_params));
        if ~isempty(flagged)
            msg = sprintf('Curve fit ended at its limits (center at the data edge or width at max) for: %s.', ...
                strjoin(param_names(flagged), ', '));
            if strcmp(bnd_opts.fit_method, 'curve')
                msg = [msg ' Their bounds use the MLE fit instead.'];
            end
            warning(msg); %#ok<SPWRN>
        end
        if ~all(bnds_info.normal(free_params))
            warning('No transform looks normal for: %s. Using a Gaussian fit in raw space for these.', ...
                strjoin(param_names(free_params(~bnds_info.normal(free_params))), ', '));
        end
        collapsed = free_params(simParamSet.param_bnds(1, free_params) == simParamSet.param_bnds(2, free_params));
        if ~isempty(collapsed)
            warning('Bounds collapse to one value for: %s. These params will be fixed.', ...
                strjoin(param_names(collapsed), ', '));
        end

        % diagnostic plot per param: the histogram the curve was fit to (stacked
        % by cohort), the MLE and curve-fit Gaussians, and the simulation range
        % (skipped in batch jobs: no display)
        if ~batchStartupOptionUsed
            h_fits = figure('Position', [100 100 1200 800]);
            tiledlayout('flow', 'TileSpacing', 'compact');
            for p = free_params
                nexttile; hold on;
                t = bnds_info.space(p);
                y = transformParams(fit_params(:,p), t, bnd_opts.offset, 'forward');
                edges = bnds_info.hist_edges{p};
                if isempty(edges), edges = y(1) + [-0.5 0.5]; end % all rats at one value
                bw = edges(2) - edges(1);
                centers = (edges(1:end-1) + edges(2:end)) / 2;
                counts_by_cohort = zeros(numel(centers), numel(fit_dirs));
                for k = 1 : numel(fit_dirs)
                    counts_by_cohort(:, k) = histcounts(y(cohort == k), edges);
                end
                hb = bar(centers, counts_by_cohort, 1, 'stacked', 'FaceAlpha', 0.6);
                for k = 1 : numel(fit_dirs), hb(k).DisplayName = cohort_names{k}; end

                y_bnds = transformParams(simParamSet.param_bnds(:,p), t, bnd_opts.offset, 'forward');
                yy = linspace(min([edges(:); y_bnds]), max([edges(:); y_bnds]), 300);
                if ~isnan(bnds_info.mle_mu(p))
                    plot(yy, normpdf(yy, bnds_info.mle_mu(p), bnds_info.mle_sd(p)) * numel(y) * bw, ...
                        'k-', 'LineWidth', 1.5, 'DisplayName', 'MLE');
                end
                if ~isnan(bnds_info.curve_mu(p))
                    % the fitted curve itself (its own height a, not rescaled)
                    plot(yy, bnds_info.curve_a(p) * exp(-((yy - bnds_info.curve_mu(p)) / (sqrt(2)*bnds_info.curve_sd(p))).^2), ...
                        'r--', 'LineWidth', 1.5, 'DisplayName', 'curve fit');
                end
                xline(y_bnds, 'k:', 'LineWidth', 1, 'HandleVisibility', 'off');

                if t == "none", xlabel(param_names(p)); else, xlabel(sprintf('%s(%s)', t, param_names(p))); end
                ylabel('# rats');
                p_space = bnds_info.pvals{p}(bnds_info.tried{p} == t); % A-D p in the plotted space
                ttl = sprintf('A-D p = %.3f, curve R^2 = %.2f', p_space, bnds_info.curve_r2(p));
                if ~isnan(bnds_info.curve_mu(p)) && ~bnds_info.curve_ok(p)
                    ttl = [ttl ' (curve at limit)'];
                end
                if ~bnds_info.normal(p)
                    title({'NOT NORMAL (raw)', ttl}, 'Color', [0.8 0 0]);
                else
                    title(ttl);
                end
            end
            lgd = legend;
            lgd.Layout.Tile = 'east'; % outside the panels
            sgtitle(sprintf('%s: Gaussian fits to %d rats; dotted = \\pm%g SD range (%s)', ...
                simModel, size(fit_params,1), bnd_opts.n_sd, upper(bnd_opts.fit_method)));
        end
    otherwise
        error('bnds_source must be ''model'' or ''fits''');
end

% sessions: every parameter set is simulated on these same sessions
rng(env_seed, 'twister');
[simParamSet.rprobset, simParamSet.is_unstr] = generateRProbSet(n_rounds, ...
    n_trials_per_block, n_arms, strcmp(env, 'unstr'), n_blocks_per_round, pure);

%% timing test: simulate a few sets and extrapolate

% n_test = 20;
% testSet = simParamSet;
% testSet.use_parfor = false;
% tic;
% fL_test = computeFitnessLandscape(testSet, n_test);
% t_test = toc;
% fprintf('%d sets: %.1f s (%.2f s/set) -> ~%.1f CPU-min for %d sets\n', ...
%     n_test, t_test, t_test/n_test, t_test/n_test*n_sets/60, n_sets);

%% run and save

% Slurm jobs (rLandscape_submit.sh) set the folder name, so results go
% where the job's logs are; otherwise build it from today's date
savefolder = getenv('LANDSCAPE_SAVEFOLDER');
if isempty(savefolder)
    savefolder = sprintf('%s_landscape_%s', char(datetime("now","Format","yyyyMMdd")), env);
end
savepath = fullfile(savedir, savefolder);
if exist(savepath, 'dir') ~= 7
    mkdir(savepath);
end
savefile = fullfile(savepath, sprintf('%s_%s_n%d_%sbnds.mat', simModel, sampling, n_sets, bnds_source));
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

% % load(savefile, 'fL', 'simParamSet'); % to start from a saved run
% top_pct = 1; % percent of best-performing sets to keep

% [~, order] = sort(fL.mean_reward, 'descend');
% n_top = max(1, round(fL.n_sets * top_pct/100));
% top_idx = order(1:n_top);

% fprintf('mean reward: best %.3f, top %g%% cutoff %.3f, median %.3f (max possible %.3f, chance %.3f)\n', ...
%     fL.mean_reward(order(1)), top_pct, fL.mean_reward(top_idx(end)), median(fL.mean_reward), ...
%     mean(fL.max_reward), mean(fL.chance_reward));
% fprintf('typical SEM of a set''s mean reward: %.4f\n', median(fL.sem_reward));

% h_rew = figure;
% histogram(fL.mean_reward, 100); hold on;
% xline(fL.mean_reward(top_idx(end)), 'r', sprintf('top %g%%', top_pct));
% xlabel('mean reward'); ylabel('# parameter sets'); title(simModel);

% h_par = figure;
% tiledlayout('flow', 'TileSpacing', 'compact');
% for p = free_params
%     nexttile;
%     edges = linspace(fL.param_bnds(1,p), fL.param_bnds(2,p), 21);
%     histogram(fL.simParams(:,p), edges, 'Normalization', 'pdf'); hold on;
%     histogram(fL.simParams(top_idx,p), edges, 'Normalization', 'pdf');
%     xlabel(param_names(p));
% end
% legend({'all', sprintf('top %g%%', top_pct)});
% sgtitle(sprintf('%s: params of top %g%% vs all sets', simModel, top_pct));

% % in batch jobs (Slurm) nobody sees the figures, so save them next to the results
% if batchStartupOptionUsed
%     [~, fname] = fileparts(savefile);
%     exportgraphics(h_rew, fullfile(savepath, [fname '_reward.png']));
%     exportgraphics(h_par, fullfile(savepath, sprintf('%s_top%gpct_params.png', fname, top_pct)));
% end
