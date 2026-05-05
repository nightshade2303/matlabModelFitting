% generateRatFitParams.m

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

% info about fitting
rFitParams.crossval = 'kfold';
rFitParams.optim = 'bads'; % 'bads' or 'fmin'
rFitParams.kfolds = 4;
rFitParams.iters = 5;
rFitParams.linear_env = 1;
rFitParams.rnd_flag = 0;

%% Find subj/sess list

% specify subject indices - if 4ABunstr in task list then select those animals
subjIndices = find(contains({rMABpre.task}, rFitParams.task));
% subjIndices = find(contains({rMABstr.task}, rFitParams.task));

rFitParams.subjIndices = subjIndices;


%% Algo and fit details

% for algoName = ["Qgen","Ql", "GPCst","GPCs", "BB", "BBgen"]
for algoName = ["Ql", "Qgen", "BB", "BBgen", "GPCs"]
    switch char(algoName)
        case 'Qgen'
            % Q-learning with generalisation (discrete)
            rFitParams.algo = 'Qlearning';
            % params     lr_max  lambda, q_decay, q0, ucb_w, rnd_w, sty_w, local
            param_bnds = [0,    1e-2,     0,       0.5,    0,     1e-2,   0,     0;    % lb
                          1,    5,        0,       0.5,    0,      2e1,    10,    0;    % ub
                          1e-2, 1e-2,     0,       0.5,    0,      1e-2,   0,     0;    % plb
                          1,    5,        0,       0.5,    0,      2e1,    10,    0;    % pub
                          0.2,  1,        0,       0.5,    0,        1,      2,     0]; % st
            

        case 'Ql'
            % % Qlearning (discrete)
            rFitParams.algo = 'Qlearning';
            % params     lr_max  lambda, q_decay, q0, ucb_w, rnd_w, sty_w, local
            param_bnds = [0,    0,       0,       0.5,    0,     1e-2,   0,     0;    % lb
                          1,    0,       0,       0.5,    0,      2e1,    10,    0;    % ub
                          1e-2, 0,       0,       0.5,    0,      1e-2,   0,     0;    % plb
                          1,    0,       0,       0.5,    0,      2e1,    10,    0;    % pub
                          0.2,  0,       0,       0.5,    0,        1,      2,     0]; % st
        
        
        case 'GPCs'
            % % GPC space (discrete)
            rFitParams.algo = 'GPspace';
            % params    lambda  time_scale, ucb_w, rnd_w, sty_w, localpol
            param_bnds = [1e-2,  100,       0,      1e-2,     0,       0;    % lb
                          5,     100,      10,      2e1,     10,       0;  % ub
                          1e-2,   100,       0,     1e-2,     0,        0;    % plb
                          5,     100,       10,     2e1,      10,       0;   % pub
                          1,     100,      2.2,        1,      2,       0];   % start  
        
        
        case 'BB'
            % BetaBayes (discrete)
            rFitParams.algo = 'BayesUCB';
            % params    lr_max  lambda, decay, a0, b0, ucb2_w, rnd_w,  sty_w,  local
            param_bnds = [1,    0,      0,     1,   1,   0,     1e-2,     0,   0;    % lb
                          1,    0,      0,     1,   1,   10,    2e1,     10,   0;    % ub
                          1,    0,      0,     1,   1,    0,    1e-2,     0,   0;    % plb
                          1,    0,      0,     1,   1,   10,    2e1,     10,   0;    % pub
                          1,    0,      0,     1,   1,   2.2,     1,      2,   0];   % st
        
        
        case 'BBgen'
            % % BetaBayes with generalisation (discrete)
            rFitParams.algo = 'BayesUCB';
            % params    lr_max  lambda, decay, a0, b0, ucb2_w, rnd_w,  sty_w,  local
            param_bnds = [1,    1e-2,     0,   1,   1,   0,     1e-2,     0,   0;    % lb
                          1,    5,        0,   1,   1,   10,    2e1,     10,   0;    % ub
                          1,    1e-2,     0,   1,   1,    0,    1e-2,     0,   0;    % plb
                          1,    5,        0,   1,   1,   10,    2e1,     10,   0;    % pub
                          1,    1,        0,   1,   1,   2.2,     1,      2,   0];   % st
    
    end
    
    rFitParams.param_bnds = param_bnds;
    
    
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