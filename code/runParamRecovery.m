% with recParams and rand_params, plot actual v. fit values
% for j = 1:size(rand_params, 2)
%     subplot(2,4,j);
%     scatter(recParams(:,j), rand_params(:, j),'xkcd:pink','filled');
%     hold on;
%     grid on;
%     plot(xlim, ylim, 'k-', 'LineWidth', 0.7) % Draws 1:1 line
%     hold off;
% end

%% mixed with ashesh's version
%% Set paths
addpath('./HelperFunctions/');
addpath(genpath('./bads-master/'));
addpath('./fminsearchbnd/');

%% Load datasets
% load("./Data/hMABdat.mat","hMABenvs");
% load("./Data/hMABdat.mat","hMABall");


%% specify algo
algo = 'Qlearning';

%% Qgen (unstr)
n_reps = 100;
n_rounds = 200;
simParamSet = struct;
% simParamSet.rprobset = hMABenvs.Str1D;
simParamsSet.rprobset = generateRProbSet(n_rounds, 4, true);
simParamSet.n_rounds = n_rounds;
simParamSet.n_trials = 100;
simParamSet.algo = 'Qlearning';
%            params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
simParamSet.param_bnds = [1e-2, 0.5,    0,       0,      1e-2,   0,     0; 
                           1,    4,      1,       1,      2e1,    10,    1];
simParamSet.logspace =    [0,    0,     0,       0,       1,     0,     0];



%% specify param bounds
%% params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
param_bnds = [0,     0,       0,       0,      1e-2,   0,     0;    % lb
              1,     0,       0,       0,      2e2,    10,    0;    % ub
              1e-2,  0,       0,       0,      1e-2,   0,     0;    % plb
              1,     0,       0,       0,      2e2,    10,    0;    % pub
              0.3,   0,       0,       0,       3,   0.8,   0.0]; % st

%% fitParams.algo, fitParams.kfolds, fitParams.iters, fitParams.param_bnds, ...
%%    choices, rewards, narms, fitParams.optim, fitParams.linear_env, fitParams.rnd_flag
fitParamsSet = struct;
fitParamsSet.algo = algo;
fitParamsSet.kfolds = 0;
fitParamsSet.iters = 10;
fitParamsSet.param_bnds = param_bnds;
fitParamsSet.optim = 'bads';
fitParamsSet.linear_env = 1;
fitParamsSet.rnd_flag = 0;

[recParams, simParams, rec_nll, rec_pR2] = paramRecovery(simParamSet, fitParamSet, n_reps);

%% saving the params
t = datetime('now', 'Format', 'yyyyMMdd');
filename = sprintf('%s_%s.m', datetime('now', 'Format', 'yyyyMMdd'), algo);
save(fullfile("~/behavioralAnalysis/matlabModelFitting/output/sim/", filename), "simParams", "recParams")