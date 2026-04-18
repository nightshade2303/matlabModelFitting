%% mixed with ashesh's version
%% Set paths
addpath('./HelperFunctions/');
addpath(genpath('./bads-master/'));
addpath('./fminsearchbnd/');

%% Load datasets
% load("./Data/hMABdat.mat","hMABenvs");
% load("./Data/hMABdat.mat","hMABall");

%% specify algo
fit_type = "BBgen";

%% simulation settings
n_reps = 10;
n_rounds = 200;
simParamSet = struct;
simParamSet.rprobset = generateRProbSet(n_rounds, 4, true);
simParamSet.n_rounds = n_rounds;
simParamSet.n_trials = 100;
simParamSet.linear_env = 1;

switch char(fit_type)
    case {'BB', 'BBgen'}
        param_names = {'lr', 'ls', 'decay','a0', 'b0', 'ucb', 'invtemp', 'sticky', 'localpol'};
        simParamSet.algo = 'BayesUCB';
                   % params    lr_max  lambda, decay, a0,  b0,  ucb2_w, rnd_w, sty_w,  localpol
        simParamSet.param_bnds = [1,    1e-2,    0,   1,   1,    0,      1e-2,   0,     1e-2;    % lb
                                  1,    2,       0,   1,   1,    1,      2e1,    10,    2];    % ub
        simParamSet.logspace =   [0,    0,       0,   0,   0,    0,       0,     0,     0];

                                
    case {'Ql', 'Qgen'}
        param_names = {'lr', 'lenscale', 'decay', 'q0', 'ucb', 'invtemp', 'sticky', 'localpol'};

        simParamSet.algo = 'Qlearning';
                    % params     lr_max  lambda, q_decay, q0, ucb1_w, rnd_w, sty_w, localpol
        simParamSet.param_bnds = [0,     0,       0,     0,    0,      1e-2,   0,     0; 
                                   1,    0,       0,     1,     0,     2e1,    10,    1];
        simParamSet.logspace =    [0,    0,       0,     0,     0,       0,     0,     0];

    case {'GPCs'}
        % untested
        param_names = {'lenscale', 'timescale', 'ucb', 'invtemp', 'sticky', 'localpol'};
        simParamSet.algo = 'GPspace';
                      % params    lambda  time, ucb_w, rnd_w, sty_w, localpol
        simParamSet.param_bnds = [1e-2,  5,      0,      1e-3,   0,     1e-2;    % lb
                                    5,   100,    10,     20,    15,     2 ];      % ub
        simParamSet.logspace =    [0,     0,     0,      0,       0,      0];
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
    case 'Ql'
        % % Qlearning (discrete)
        fitParamSet.algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, q0, ucb1_w, rnd_w, sty_w, localpol
        param_bnds = [0,     0,        0,     0,   0,     1e-2,   0,     0;    % lb
                      1,     0,         0,     0,   0,      2e2,   10,    0;    % ub
                     1e-2,   0,         0,     0,   0,     1e-2,   0,     0;    % plb
                      1,      0,        0,     0,   0,      2e2,   10,    0;    % pub
                     0.3,     0,        0,     0,   0,       3,    0.8,   0]; % st

    case 'Qgen'
        % Q-learning with generalisation (discrete)
        fitParamSet.algo = 'Qlearning';
        % params     lr_max  lambda, q_decay, q0, ucb1_w, rnd_w, sty_w, localpol
        param_bnds = [0,     1e-2,      0,     0,   0,     1e-2,   0,     0;    % lb
                      1,     3,         0,     0,   0,      2e2,   10,    0;    % ub
                     1e-2,   1e-2,      0,     0,   0,     1e-2,   0,     0;    % plb
                      1,      3,        0,     0,   0,      2e2,   10,    0;    % pub
                     0.3,     1,        0,     0,   0,       3,    0.8,   0]; % st
    
    case 'GPCst'
        % % GPC spacetime (discrete)
        fitParamSet.algo = 'GPspacetime';
        % params    lambda  time, ucb_w, rnd_w, sty_w, localpol
        param_bnds = [1e-2,  5,    0,   1e-3,   0,     1e-2;    % lb
                       5,   100,  10,   20,    15,     2;        % ub
                      1e-1,  5,    0,   1e-1,   0,     1e-1;    % plb
                       4,   100,   5,   15,     5,      1;      % pub
                      0.3,   62,  0.4,  12.2,   2.3,  0.5];    % start  
    case 'GPCs'
        % % GPC space (discrete)
        fitParamSet.algo = 'GPspace';
        % params    lambda  time, ucb_w, rnd_w, sty_w, localpol
        param_bnds = [1e-2,  5,    0,   1e-3,   0,     1e-2;    % lb
                       5,   100,  10,   20,    15,     2;        % ub
                      1e-1,  5,    0,   1e-1,   0,     1e-1;    % plb
                       4,   100,   5,   15,     5,      1;      % pub
                      0.3,   62,  0.4,  12.2,   2.3,  0.5];    % start 
    
    case 'BB'
        % BetaBayes (discrete)
        fitParamSet.algo = 'BayesUCB';
        % params    lr_max  lambda, decay, a0,  b0, ucb2_w, rnd_w,  sty_w, localpol 
        param_bnds = [0,    0,      0,     1,   1,    0,     1e-3,   0,    0;    % lb
                      1,    0,      0,     1,   1,    1,     20,    10,    0;    % ub
                      0,    0,      0,     1,   1,    0,     1e-2,   0,    0;    % plb
                      1,    0,      0,     1,   1,    1,     20,    10,    0;    % pub
                      0.7,  0,      0,     1,   1,    0.1,   2,     2.5,   0];   % st
    
    case 'BBgen'
        % % BetaBayes with generalisation (discrete)
        fitParamSet.algo = 'BayesUCB';
        % params    lr_max  lambda, decay, a0,  b0, ucb2_w, rnd_w,  sty_w, localpol
        param_bnds = [0,    1e-2,   0,     1,   1,    0,     1e-3,   0,    0;    % lb
                      1,    3,      0,     1,   1,    1,     20,    10,    0;    % ub
                      0,    1e-1,   0,     1,   1,    0,     1e-2,   0,    0;    % plb
                      1,    3,      0,     1,   1,    1,     20,    10,    0;    % pub
                      0.7,  0.5,    0,     1,   1,    0.1,   2,     2.5,   0];   % st

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
% addpath('./ParamParseUtils')
% myPlotSettings();
% figure;
% for j = 1:size(simParams, 2)
%     subplot(2,5,j);
%     scatter(simParams(:,j), recParams(:, j),'filled');
%     if strcmp(param_names{j}, 'invtemp')==1
%         set(gca, 'xscale', 'log', 'yscale', 'log');
    
%     else 
%         hold on;
%         % plot(xlim, ylim, 'k-', 'LineWidth', 0.7) % Draws 1:1 line
%         myRefLine = refline(1, 0);
%         myRefLine.Color = 'k';
%         myRefLine.LineStyle = '-';
%         myRefLine.LineWidth = 0.7;
%         hold off;
%     end
%     title(param_names{j});
%     xlabel('actual');
%     ylabel('fit');
% end


% figure();
% subplot(1,2,1);
% imagesc(corrcoef(simParams));
% colorbar;
% subplot(1,2,2);
% imagesc(corrcoef(recParams));
% colorbar;
