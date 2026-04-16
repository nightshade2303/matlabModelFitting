%% mixed with ashesh's version
%% Set paths
addpath('./HelperFunctions/');
addpath(genpath('./bads-master/'));
addpath('./fminsearchbnd/');

%% Load datasets
% load("./Data/hMABdat.mat","hMABenvs");
% load("./Data/hMABdat.mat","hMABall");

%% specify algo
fit_type = "BB";

%% simulation settings
n_reps = 100;
n_rounds = 200;
simParamSet = struct;
simParamSet.rprobset = generateRProbSet(n_rounds, 4, true);
simParamSet.n_rounds = n_rounds;
simParamSet.n_trials = 100;

switch char(fit_type)
    case {'BB', 'BBgen'}
        simParamSet.algo = 'BetaBayes';
                   % params    lr_max  lambda, decay, ucb2_w, rnd_w,  sty_w,  a0,  b0
        simParamSet.param_bnds = [0,    0,      0,       0,      1e-2,   0,     1,   1;    % lb
                                  1,    0,      0,       0,      2e1,    10,    1,   1];    % ub
        simParamSet.logspace =   [0,    0,     0,       0,       1,     0,     0,    0];

                                
    case {'Ql', 'Qgen'}
        simParamSet.algo = 'Qlearning';
                    % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
        simParamSet.param_bnds = [0,     0,    0,       0,      1e-2,   0,     0; 
                                   1,    0,      0,       0,     2e1,    10,    1];
        simParamSet.logspace =    [0,    0,     0,       0,       1,     0,     0];

    case {'GPCs'}
        simParamSet.algo = 'GPspace';
                      % params    lambda  ucb_w, rnd_w, sty_w, time_scale
        simParamSet.param_bnds = [1e-2,   0,   1e-2,   0,     100;    % lb
                                    2,     0,   2,    10,    100];  % ub
        simParamSet.logspace =    [0,     0,      1,     0,     0];
end
%% recovery fit settings
% fitParams.algo, fitParams.kfolds, fitParams.iters, fitParams.param_bnds, ...
%    choices, rewards, narms, fitParams.optim, fitParams.linear_env, fitParams.rnd_flag
fitParamSet = struct;
fitParamSet.kfolds = 0;
fitParamSet.iters = 1;
fitParamSet.optim = 'bads';
fitParamSet.linear_env = 1;
fitParamSet.rnd_flag = 0;
fitParamSet.crossval = 'none';


%% specify param bounds for recovery
switch char(fit_type)
    case 'Qgen'
        % Q-learning with generalisation (discrete)
        fitParamSet.algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
        param_bnds = [0,     1e-2,       0,       0,      1e-2,   0,     0;    % lb
                     1,     3,       0,       0,      2e2,    10,    1;    % ub
                     1e-2,  1e-2,       0,       0,      1e-2,   0,     0;    % plb
                      1,     3,       0,       0,      2e2,    10,    1;    % pub
                     0.3,   1,       0,       0,       3,   0.8,     0.0]; % st
        
    case 'Ql'
        % % Qlearning (discrete)
        fitParamSet.algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
        param_bnds = [0,     0,       0,       0,      1e-2,   0,     0;    % lb
                      1,    0,       0,       0,      2e2,    10,    1;    % ub
                      1e-2,  0,       0,       0,      1e-2,   0,     0;    % plb
                      1,     0,       0,       0,      2e2,    10,    1;    % pub
                      0.3,   0,       0,      0,      3,      0.8,      0]; % st
    
    case 'GPCst'
        % % GPC spacetime (discrete)
        fitParamSet.algo = 'GPspacetime_linear';
        % params    lambda  ucb_w, rnd_w, sty_w, time_scale
        param_bnds = [1e-2,   0,   1e-3,   0,     5;    % lb
                        5,     10,   20,    15,    100;  % ub
                        1e-1,   0,   1e-1,   0,     5;    % plb
                        4,      5,   15,     5,    100;   % pub
                        0.3,    0.4, 12.2,   2.3,  62];   % start  
    case 'GPCs'
        % % GPC space (discrete)
        fitParamSet.algo = 'GPspace_linear';
        % params    lambda  ucb_w, rnd_w, sty_w, time_scale
        param_bnds = [1e-2,   0,   1e-3,   0,     100;    % lb
                        5,     10,   20,    15,    100;  % ub
                        1e-1,   0,   1e-1,   0,     100;    % plb
                        4,      5,   15,     5,    100;   % pub
                        0.3,    0.4, 12.2,   2.3,  100];   % start  
    
    case 'BB'
        % BetaBayes (discrete)
        fitParamSet.algo = 'BetaBayes';
        % params    lr_max  lambda, decay, ucb2_w, rnd_w,  sty_w,  a0,  b0
        param_bnds = [0,    0,      0,       0,      1e-3,   0,     1,   1;    % lb
                        1,    0,      0,       0,      20,    10,    1,   1;    % ub
                        0,    0,      0,       0,      1e-2,   0,     1,   1;    % plb
                        1,    0,      0,       0,      20,    10,    1,   1;    % pub
                        0.7,  0,      0,     0,       2,     2.5,   1,   1];   % st
    
    case 'BBgen'
        % % BetaBayes with generalisation (discrete)
        fitParamSet.algo = 'BetaBayes';
        % params    lr_max  lambda, decay, ucb2_w, rnd_w,  sty_w,  a0,  b0
        param_bnds = [0,    1e-3,   0,       0,      1e-3,   0,     1,   1;    % lb
                        1,    10,     1,       1,      5e2,    10,    20,   20;    % ub
                        0,    0.5,    0,       0,      1e-2,   0,     1,   1;    % plb
                        1,    4,      1,       1,      5e2,    10,    5,   5;    % pub
                        0.7,  0.2,    0.2,     0.09,   38,     2.5,   1,   1];   % st

end
fitParamSet.param_bnds = param_bnds;

tic()
[recParams, simParams, rec_nll, rec_pR2] = paramRecovery(simParamSet, fitParamSet, n_reps);

%% saving the params
t = datetime('now', 'Format', 'yyyyMMdd_HHmmss');
filename = sprintf('%s_%s_%s_%s.mat', t, fit_type, fitParamSet.optim, num2str(fitParamSet.iters));
save(fullfile("~/behavioralAnalysis/matlabModelFitting/output/sim/", filename), "simParams", "recParams")
toc()

% with recParams and simParams, plot actual v. fit values
% param_names = {'lr', 'ls', 'decay', 'ucb', 'invtemp', 'sticky', 'q0'};

% for j = 1:size(simParams, 2)
%     subplot(2,4,j);
%     scatter(simParams(:,j), recParams(:, j),'filled');
%     if strcmp(param_names{j}, 'invtemp')==1
%         set(gca, 'xscale', 'log', 'yscale', 'log');
%     else
%         hold on;
%         plot(xlim, ylim, 'k-', 'LineWidth', 0.7) % Draws 1:1 line
%         hold off;
%     end
%     title(param_names{j});
%     xlabel('actual');
%     ylabel('fit');
% end