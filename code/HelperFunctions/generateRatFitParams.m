% generateRatFitParams.m

%% Load dataset

% unstructured dataset
if exist("rMABpre","var") ~= 1
    load("../../data/rMABdat_20260212.mat", "rMABpre");
end

% structured dataset
% if exist("rMABstr","var") ~= 1
%     load("../../data/rMABpre_20260513.mat", "rMABpre");
% end

%% Initialise fit param struct and savedir

rFitParams = struct;

savedir = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/output/r_fits/20260519_unstr_b_rats/';

slurm_file = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/code/rMAB_submit.sh';  % change this maybe?

%% Task, sess and round structure

fitName = 'unstr_b';
rFitParams.task = '4ABunstr';
% rFitParams.rounds = {1:100};
% trials and sessions are defined individually for each animal in fitRatData.m

% if fitting structured or lesion data - uncomment below:
% fitName = 'dls';
% rFitParams.task = '4ABunstr';

% fitName = 'str';
% rFitParams.task = '4ABstr';

% info about fitting
rFitParams.crossval = 'none'; % no more kfolding
rFitParams.optim = 'bads'; % 'bads' or 'fmin'
rFitParams.kfolds = 4;
rFitParams.iters = 5;
rFitParams.linear_env = 1;
rFitParams.rnd_flag = 0;

%% Find subj/sess list
unblocked_rats = {'Ivysaur', 'Jigglypuff', 'Vanillish', 'Whismur', 'Zorua'};
blocked_rats = {'Inkay', 'Kirlia', 'Mesprit', 'Quilava', 'Raltz'};
str_rats = {'HSB_F_008', 'HSB_F_009', 'HSB_F_011', 'HSB_F_013', 'HSB_F_014'};

% specify subject indices - if 4ABunstr in task list then select those animals
subjIndices = find(contains({rMABpre.task}, rFitParams.task));
% subjIndices = find(contains({rMABstr.task}, rFitParams.task));
% subjIndices = find(contains({rMABpre.ratName}, blocked_rats));
% subjIndices = find(contains({rMABpre.ratName}, str_rats));

rFitParams.subjIndices = subjIndices;


%% Algo and fit details
modelList = ["Q","Qs","Qsl","Qd","Qds","Qdsl","Qg","Qgs","Qgsl","Qdg","Qdgs","Qdgsl",...
             "B","Bs","Bsl","Bd","Bds","Bdsl","Bu","Bus","Busl","Bdu","Bdus","Bdusl"];
            %   "G","Gs","Gsl","Gd","Gds","Gdsl","Gu","Gus","Gusl","Gdu","Gdus","Gdusl"];
fitData = 'B';

for fitModel = modelList
    % rFitParams = myFitParamSet; % Copy default fit parameters - find out what this does exactly (not much?)
    rFitParams.model = fitModel;
    [rFitParams.algo, fit_param_bnds] = modelParamBounds(char(fitModel));
    rFitParams.param_bnds = [fit_param_bnds; fit_param_bnds; mean(fit_param_bnds,1)];
    
    
    %% save fit params
    fitName = [char(fitData) '_' char(fitModel)];
    
    savefolder = [char(datetime("now","Format","yyyyMMdd")), '_',  char(fitData), '_', char(fitModel), '/'];
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
    
for fitModel = modelList
    fitName = [char(fitData) '_' char(fitModel)];
    savefolder = [char(datetime("now","Format","yyyyMMdd")), '_',  char(fitData), '_', char(fitModel), '/'];
    savepath = [savedir, savefolder];
    %% format sbatch command
    cmd = sprintf('cd %s/Jobs/', savepath(1:end - 1));
    disp(cmd);
    cmd = sprintf('sbatch -J %s_%s -t 3-00:00 --array=1-5 %s %s/%s', ...
        char(fitData),char(fitModel),char(slurm_file),'20260513_str_b_rats',savefolder(1:end-1));
    disp(cmd);
end


% for algoName = ["Qgen","Ql", "GPCst","GPCs", "BB", "BBgen"]
% for algoName = ["Ql", "Qgen","Qlocal", "ALR", "BB", "BBlocal", "GPCs"]
%     switch char(algoName)
%         case 'Qgen'
%             % Q-learning with generalisation (discrete)
%             rFitParams.algo = 'Qlearning';
%             % params     lr_max  lambda, q_decay, q0, ucb_w, rnd_w, sty_w, local
%             param_bnds = [0,    0,       0,       0.5,    0,     1e-2,   0,     0;    % lb
%                           1,    3,        0.5,     0.5,    0,      2e1,    4,    0;    % ub
%                           1e-2, 0,        0,       0.5,    0,      1e-2,   0,     0;    % plb
%                           1,    3,        0.5,     0.5,    0,      2e1,    4,    0;    % pub
%                           0.2,  1,        0,       0.5,    0,        1,      2,     0]; % st
            

%         case 'Ql'
%             % % Qlearning (discrete)
%             rFitParams.algo = 'Qlearning';
%             % params     lr_max  lambda, q_decay, q0, ucb_w, rnd_w, sty_w, local
%             param_bnds = [0,    0,       0,       0.5,    0,     1e-2,   0,     0;    % lb
%                           1,    0,       0.5,     0.5,    0,      2e1,    4,    0;    % ub
%                           1e-2, 0,       0,       0.5,    0,      1e-2,   0,     0;    % plb
%                           1,    0,       0.5,     0.5,    0,      2e1,    4,    0;    % pub
%                           0.2,  0,       0,       0.5,    0,        1,      2,     0]; % st

%         case 'Qlocal'
%             % % Qlearning (discrete)
%             rFitParams.algo = 'Qlearning';
%             % params     lr_max  lambda, q_decay, q0, ucb_w, rnd_w, sty_w, local
%             param_bnds = [0,    0,       0,        0.5,    0,     1e-2,   0,     0;    % lb
%                           1,    0,       0.5,      0.5,    0,      2e1,    4,    3;    % ub
%                           1e-2, 0,       0,        0.5,    0,      1e-2,   0,     0;    % plb
%                           1,    0,       0.5,      0.5,    0,      2e1,    4,    3;    % pub
%                           0.2,  0,       0,        0.5,    0,        1,      2,     0]; % st
        
        
%         case 'GPCs'
%             % % GPC space (discrete)
%             rFitParams.algo = 'GPspace';
%             % params    lambda  time_scale, ucb_w, rnd_w, sty_w, localpol
%             param_bnds = [1e-2,  100,       0,      1e-2,     0,       0;    % lb
%                           5,     100,      1,      2e1,     10,       0;  % ub
%                           1e-2,   100,      0,     1e-2,     0,        0;    % plb
%                           5,     100,       1,     2e1,      10,       0;   % pub
%                           1,     100,      0.2,        1,      2,       0];   % start  
        
%         case 'ALR'
%             % BetaBayes (discrete)
%             rFitParams.algo = 'BayesUCB';
%             % params    lr_max  lambda, decay, a0, b0, ucb2_w, rnd_w,  sty_w,  local
%             param_bnds = [1,    0,      0,     1,   1,   0,     1e-2,     0,   0;    % lb
%                           1,    0,      0,     1,   1,   0,    2e1,     10,   0;    % ub
%                           1,    0,      0,     1,   1,   0,    1e-2,     0,   0;    % plb
%                           1,    0,      0,     1,   1,   0,    2e1,     10,   0;    % pub
%                           1,    0,      0,     1,   1,   0,     1,      2,   0];   % st
        
%         case 'BB'
%             % BetaBayes (discrete)
%             rFitParams.algo = 'BayesUCB';
%             % params    lr_max  lambda, decay, a0, b0, ucb2_w, rnd_w,  sty_w,  local
%             param_bnds = [1,    0,      0,     1,   1,   0,     1e-2,     0,   0;    % lb
%                           1,    0,      0,     1,   1,   1,    2e1,     10,   0;    % ub
%                           1,    0,      0,     1,   1,    0,    1e-2,     0,   0;    % plb
%                           1,    0,      0,     1,   1,   1,    2e1,     10,   0;    % pub
%                           1,    0,      0,     1,   1,   0.2,     1,      2,   0];   % st
        
        
%         case 'BBlocal'
%             % % BetaBayes with generalisation (discrete)
%             rFitParams.algo = 'BayesUCB';
%             % params    lr_max  lambda, decay, a0, b0, ucb2_w, rnd_w,  sty_w,  local
%             param_bnds = [1,    0,        0,   1,   1,   0,     1e-2,     0,   0;    % lb
%                           1,    0,        0,   1,   1,   1,    2e1,     10,   3;    % ub
%                           1,    0,        0,   1,   1,    0,    1e-2,     0,   0;    % plb
%                           1,    0,        0,   1,   1,   1,    2e1,     10,   3;    % pub
%                           1,    0,        0,   1,   1,   0.2,     1,      2,   1];   % st
    
%     end
    
