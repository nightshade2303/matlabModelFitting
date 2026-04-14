% addpath('./HelperFunctions/');
% addpath(genpath('./bads-master/'));

% specify param bounds and algo
% algo = 'Qlearning';
% params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
% param_bnds = [0,     0,       0,       0,      1e-2,   0,     0;    % lb
%               1,     0,       0,       0,      2e2,    10,    0;    % ub
%               1e-2,  0,       0,       0,      1e-2,   0,     0;    % plb
%               1,     0,       0,       0,      2e2,    10,    0;    % pub
%               0.3,   0,       0,       0,       3,   0.8,   0.0]; % st


% specify reps
% reps = 100;

% specify inputs for sim and fit
% simParams.algo, simParams.params, simParams.n_rounds, simParams.n_trials, simParams.rprobset
% simParams = struct;
% simParams.algo = algo;
% simParams.n_rounds = 200;
% simParams.n_trials = 100;
% simParams.rprobset = generateRProbSet(simParams.n_rounds, 4, true);

% fitParams.algo, fitParams.kfolds, fitParams.iters, fitParams.param_bnds, ...
%    choices, rewards, narms, fitParams.optim, fitParams.linear_env, fitParams.rnd_flag
fitParams = struct;
fitParams.algo = algo;
fitParams.kfolds = 4;
fitParams.iters = 1;
fitParams.param_bnds = param_bnds;
fitParams.optim = 'bads';
fitParams.linear_env = 1;
fitParams.rnd_flag = 0;

% uniformly distributed numbers between lower and upper bounds for sim
% sim_bnds = param_bnds;
% sim_bnds(1, 5) = 1e-1;
% sim_bnds(2, 5) = 50;
% rand_params = unifrnd(repmat(sim_bnds(1, :), reps, 1), repmat(sim_bnds(2, :), reps, 1), [reps size(sim_bnds, 2)]);

% run repetitively 
% recParams = zeros(reps, size(param_bnds, 2));
% rec_nll = zeros(reps, 1);
% rec_pR2 = zeros(reps, 1);
% for i = 1:reps
%     simParams.params = rand_params(i, :);
%     [recParams(i, :), rec_nll(i), rec_pR2(i)] = paramRecovery(simParams, fitParams);
%     % disp(['Rep ' num2str(i) ' of ' num2str(reps)]);
% end

% saving the params
% t = datetime('now', 'Format', 'yyyyMMdd');
% filename = sprintf('%s_ql_vanilla.m', datetime('now', 'Format', 'yyyyMMdd'));
% save(fullfile("~/behavioralAnalysis/matlabModelFitting/output/sim/", filename), "rand_params", "recParams")

% with recParams and rand_params, plot actual v. fit values
% for j = 1:size(rand_params, 2)
%     subplot(2,4,j);
%     scatter(recParams(:,j), rand_params(:, j),'xkcd:pink','filled');
%     hold on;
%     grid on;
%     plot(xlim, ylim, 'k-', 'LineWidth', 0.7) % Draws 1:1 line
%     hold off;
% end

%% ashesh's version
%% Set paths
addpath('./HelperFunctions/');
addpath(genpath('./bads-master/'));
addpath('./fminsearchbnd/');

%% Load datasets
% load("./Data/hMABdat.mat","hMABenvs");
% load("./Data/hMABdat.mat","hMABall");


%% Qgen (str)
n_reps = 100;
simParamSet = struct;
simParamSet.rprobset = hMABenvs.Str1D;
simParamSet.n_rounds = 34;
simParamSet.n_trials = 60;
simParamSet.algo = 'Qlearning';
%            params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
simParamSet.param_bnds = [1e-2, 0.5,    0,       0,      1e-2,   0,     0; 
                           1,    4,      1,       1,      2e1,    10,    1];
simParamSet.logspace =    [0,    0,     0,       0,       1,     0,     0];

[recParams, simParams, rec_nll, rec_pR2] = paramRecovery(simParamSet, fitParamSet, n_reps);