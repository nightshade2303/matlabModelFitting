% generateHumanFitParams.m

%% Load dataset

if exist("rMABpre","var") ~= 1
    load("../../data/rMABdat_20260212.mat", "rMABpre");
end

% if exist("rMABstr","var") ~= 1
%     load("../../data/rMABstr_20260324.mat", "rMABstr");
% end

%% Initialise fit param struct and savedir

rFitParams = struct;

savedir = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/output/r_fits/';

slurm_file = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/code/rMAB_submit.sh';

%% Task, sess and round structure

fitName = 'unstr_all_rats_block1';
rFitParams.task = '4ABunstr';
% rFitParams.rounds = {1:100};
% trials and sessions are defined individually for each animal in fitRatData.m

% if fitting structured or lesion data - uncomment below:
% fitName = 'dls';
% rFitParams.task = '4ABunstr';

% fitName = 'str';
% rFitParams.task = '4ABstr';

% fitName = 'str_s1_rds';
% hFitParams.task = '1D_Str';
% hFitParams.sess = 1;
% hFitParams.rounds = {1:5, 6:10, 11:15, 16:20, 21:25, 26:30, 31:34}; % binned rounds
% 
% fitName = 'ustr_s1_rds';
% hFitParams.task = '1D_Unstr';
% hFitParams.sess = 1;
% hFitParams.rounds = {1:5, 6:10, 11:15, 16:20, 21:25, 26:30, 31:34}; % binned rounds
% 
% fitName = 'str_s2_rds';
% hFitParams.task = '1D_Str';
% hFitParams.sess = 2;
% hFitParams.rounds = {1:5, 6:10, 11:15, 16:20, 21:25, 26:30, 31:34}; % binned rounds
% 
% fitName = 'ustr_s2_rds';
% hFitParams.task = '1D_Unstr';
% hFitParams.sess = 2;
% hFitParams.rounds = {1:5, 6:10, 11:15, 16:20, 21:25, 26:30, 31:34}; % binned rounds
% 
% fitName = 'ustr_s2';
% hFitParams.task = '1D_Unstr';
% hFitParams.sess = 2;
% hFitParams.rounds = {1:34};                                       % all rounds
% 
% fitName = 'str_s2';
% hFitParams.task = '1D_Str';
% hFitParams.sess = 2;
% hFitParams.rounds = {1:34};                                       % all rounds
% 
% fitName = 'inter';
% hFitParams.rounds = {1:10, 11:25};
% hFitParams.task = '1D_Inter';
% hFitParams.sess = 1;

% fitName = 'block';
% hFitParams.rounds = {1:5, 6:10, 11:15, 18:22, 23:27, 28:32};
% hFitParams.sess = 1;
% hFitParams.task = '1D_Block';


%% Algo and fit details

% for algoName = ["Qgen","Ql", "GPCst","GPCs", "BB", "BBgen"]
for algoName = ["Qgen_vanilla", "Qgen_q0", "Qgen_decay", "Qgen_sticky"]
% for algoName = ["GPCs", "GPCst"]
    switch char(algoName)
        case 'Qgen'
            % Q-learning with generalisation (discrete)
            rFitParams.algo = 'Qlearning';
            % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
            param_bnds = [0,    1e-3,   0,       0,      1e-3,   0,     0;    % lb
                          1,    10,     1,       1,      2e2,    10,    1;    % ub
                          1e-2, 0.5,    0,       0,      1e-2,   0,     0;    % plb
                          1,    4,      1,       1,      2e2,    10,    1;    % pub
                          0.3,  0.6,    0.2,     1/150,  0.07,   0.8,   0.1]; % st
            
        case 'Qgen_vanilla'
            % Q-learning with generalisation (discrete)
            rFitParams.algo = 'Qlearning';
            % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
            param_bnds = [0,    1e-3,   0,       0,      1e-3,   0,     0;    % lb
                          1,    10,     0,       0,      2e2,    0,    0;    % ub
                          1e-2, 0.5,    0,       0,      1e-2,   0,     0;    % plb
                          1,    4,      0,       0,      2e2,    0,     0;    % pub
                          0.3,  0.5,    0,       0,      0.07,   0,    0]; % st

        case 'Qgen_q0'
            % Q-learning with generalisation (discrete)
            rFitParams.algo = 'Qlearning';
            % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
            param_bnds = [0,    1e-3,   0,       0,      1e-3,   0,     0;    % lb
                          1,    10,     0,       0,      2e2,    0,    1;    % ub
                          1e-2, 0.5,    0,       0,      1e-2,   0,     0;    % plb
                          1,    4,      0,       0,      2e2,    0,     1;    % pub
                          0.3,  0.5,    0,       0,      0.07,   0,    0.1]; % st

        case 'Qgen_decay'
            % Q-learning with generalisation (discrete)
            rFitParams.algo = 'Qlearning';
            % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
            param_bnds = [0,    1e-3,   0,       0,      1e-3,   0,     0;    % lb
                          1,    10,     1,       0,      2e2,    0,    0;    % ub
                          1e-2, 0.5,    0,       0,      1e-2,   0,     0;    % plb
                          1,    4,      1,       0,      2e2,    0,     0;    % pub
                          0.3,  0.5,    0.1,       0,      0.07,   0,    0]; % st

        case 'Qgen_sticky'
            % Q-learning with generalisation (discrete)
            rFitParams.algo = 'Qlearning';
            % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
            param_bnds = [0,    1e-3,   0,       0,      1e-3,   0,     0;    % lb
                          1,    10,     0,       0,      2e2,    10,    1;    % ub
                          1e-2, 0.5,    0,       0,      1e-2,   0,     0;    % plb
                          1,    4,      0,       0,      2e2,    10,     1;    % pub
                          0.3,  0.5,    0,       0,      0.07,   0.8,    0.1]; % st

        case 'Ql'
            % % Qlearning (discrete)
            rFitParams.algo = 'Qlearning';
            % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
            param_bnds = [0,     0,       0,       0,      1e-3,   0,     0;    % lb
                          1,     0,       1,       1,      2e2,    10,    1;    % ub
                          1e-2,  0,       0,       0,      1e-2,   0,     0;    % plb
                          1,     0,       1,       1,      2e2,    10,    1;    % pub
                          0.3,   0,       0.2,     1/150,  0.07,   0.8,   0.1]; % st
        
        
        
        case 'GPCst'
            % % GPC spacetime (discrete)
            rFitParams.algo = 'GPspacetime_linear';
            % params    lambda  ucb_w, rnd_w, sty_w, time_scale
            param_bnds = [1e-2,   0,   1e-3,   0,     5;    % lb
                          5,     10,   20,    15,    100;  % ub
                          1e-1,   0,   1e-1,   0,     5;    % plb
                          4,      5,   15,     5,    100;   % pub
                          0.3,    0.4, 12.2,   2.3,  62];   % start  
        case 'GPCs'
            % % GPC space (discrete)
            rFitParams.algo = 'GPspace_linear';
            % params    lambda  ucb_w, rnd_w, sty_w, time_scale
            param_bnds = [1e-2,   0,   1e-3,   0,     100;    % lb
                          5,     10,   20,    15,    100;  % ub
                          1e-1,   0,   1e-1,   0,     100;    % plb
                          4,      5,   15,     5,    100;   % pub
                          0.3,    0.4, 12.2,   2.3,  100];   % start  
        
        
        case 'BB'
            % BetaBayes (discrete)
            rFitParams.algo = 'BetaBayes';
            % params    lr_max  lambda, decay, ucb2_w, rnd_w,  sty_w,  a0,  b0
            param_bnds = [0,    0,      0,       0,      1e-3,   0,     1,   1;    % lb
                          1,    0,      1,       1,      5e2,    10,    20,   20;    % ub
                          0,    0,      0,       0,      1e-2,   0,     1,   1;    % plb
                          1,    0,      1,       1,      5e2,    10,    5,   5;    % pub
                          0.7,  0,      0.2,     0.09,   38,     2.5,   1,   1];   % st
        
        
        case 'BBgen'
            % % BetaBayes with generalisation (discrete)
            rFitParams.algo = 'BetaBayes';
            % params    lr_max  lambda, decay, ucb2_w, rnd_w,  sty_w,  a0,  b0
            param_bnds = [0,    1e-3,   0,       0,      1e-3,   0,     1,   1;    % lb
                          1,    10,     1,       1,      5e2,    10,    20,   20;    % ub
                          0,    0.5,    0,       0,      1e-2,   0,     1,   1;    % plb
                          1,    4,      1,       1,      5e2,    10,    5,   5;    % pub
                          0.7,  0.2,    0.2,     0.09,   38,     2.5,   1,   1];   % st
    
    end
    
    rFitParams.param_bnds = param_bnds;
    rFitParams.kfolds = 4;
    rFitParams.iters = 5;
    rFitParams.linear_env = 1;
    rFitParams.optim = 'bads'; % 'bads' or 'fmin'
    rFitParams.rnd_flag = 0;
    
    %% Find subj/sess list
    
    % switch hFitParams.task
    %     case '1D_Inter'
    %         subjIndices = find(contains({hMABall.task}, hFitParams.task) ...
    %         & cellfun(@(x) sum(x)/numel(x), {hMABall.structured}) == 0.5 ...
    %         & [hMABall.session] == hFitParams.sess);
    %     otherwise
    %         subjIndices = find(contains({hMABall.task}, hFitParams.task) ...
    %             & [hMABall.session] == hFitParams.sess);
    % end

    % specify how to get subject indices - if 4ABunstr in task list then select those animals
    subjIndices = find(contains({rMABpre.task}, rFitParams.task));
    % subjIndices = find(contains({rMABstr.task}, rFitParams.task));

    rFitParams.subjIndices = subjIndices;
    
    
    %% save fit params
    
    savefolder = [char(datetime("now","Format","yyyyMMdd")), '_',  fitName, '_', char(algoName), '/'];
    savepath = [savedir, savefolder];
    
    if exist(savepath, 'dir') ~= 7
        mkdir(savepath);    
        mkdir([savepath 'Jobs']);
    elseif numel(dir(savepath)) > 3
        disp('Directory is not empty...');
        return;
    end
    
    disp([savefolder(1:end-1), ' : ', num2str(length(subjIndices)), ' subjects']);
    disp(rFitParams);
    save([savepath,'rFitParams.mat'], 'rFitParams');

    copyfile(slurm_file, savepath);

end