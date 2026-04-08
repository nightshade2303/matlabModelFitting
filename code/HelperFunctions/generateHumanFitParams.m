% generateHumanFitParams.m

%% Load dataset

if exist("hMABall","var") ~= 1
    load("hMABdat.mat", "hMABall");
end

%% Initialise fit param struct and savedir

hFitParams = struct;

savedir = '.\h_fits\';

slurm_file = '.\hMAB_submit.sh';

%% Task, sess and round structure

hFitParams.trials = 1:60;

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
fitName = 'inter';
hFitParams.rounds = {1:10, 11:25};
hFitParams.task = '1D_Inter';
hFitParams.sess = 1;

% fitName = 'block';
% hFitParams.rounds = {1:5, 6:10, 11:15, 18:22, 23:27, 28:32};
% hFitParams.sess = 1;
% hFitParams.task = '1D_Block';


%% Algo and fit details

for algoName = ["Qgen","Ql", "GPCst", "BB", "BBgen"]

    switch char(algoName)
        case 'Qgen'
            % Q-learning with generalisation (discrete)
            hFitParams.algo = 'Qlearning';
            % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
            param_bnds = [0,    1e-3,   0,       0,      1e-3,   0,     0;    % lb
                          1,    10,     1,       1,      2e2,    10,    1;    % ub
                          1e-2, 0.5,    0,       0,      1e-2,   0,     0;    % plb
                          1,    4,      1,       1,      2e2,    10,    1;    % pub
                          0.3,  0.6,    0.2,     1/150,  0.07,   0.8,   0.1]; % st
            
        case 'Ql'
            % % Qlearning (discrete)
            hFitParams.algo = 'Qlearning';
            % params     lr_max  lambda, q_decay, ucb1_w, rnd_w, sty_w, q0
            param_bnds = [0,     0,       0,       0,      1e-3,   0,     0;    % lb
                          1,     0,       1,       1,      2e2,    10,    1;    % ub
                          1e-2,  0,       0,       0,      1e-2,   0,     0;    % plb
                          1,     0,       1,       1,      2e2,    10,    1;    % pub
                          0.3,   0,       0.2,     1/150,  0.07,   0.8,   0.1]; % st
        
        
        
        case'GPCst'
            % % GPC spacetime (discrete)
            hFitParams.algo = 'GPspacetime';
            % params    lambda  ucb_w, rnd_w, sty_w, time_scale
            param_bnds = [1e-1,   0,   1e-3,   0,     1;    % lb
                          10,     10,   20,    15,    200;  % ub
                          1e-1,   0,   1e-1,   0,     1;    % plb
                          4,      5,   15,     5,    200;   % pub
                          0.3,    0.4, 12.2,   2.3,  62];   % start  
        
        
        case 'BB'
            % BetaBayes (discrete)
            hFitParams.algo = 'BetaBayes';
            % params    lr_max  lambda, decay, ucb2_w, rnd_w,  sty_w,  a0,  b0
            param_bnds = [0,    0,      0,       0,      1e-3,   0,     1,   1;    % lb
                          1,    0,      1,       1,      5e2,    10,    1,   1;    % ub
                          0,    0,      0,       0,      1e-2,   0,     1,   1;    % plb
                          1,    0,      1,       1,      5e2,    10,    1,   1;    % pub
                          0.7,  0,      0.2,     0.09,   38,     2.5,   1,   1];   % st
        
        
        case 'BBgen'
            % % BetaBayes with generalisation (discrete)
            hFitParams.algo = 'BetaBayes';
            % params    lr_max  lambda, decay, ucb2_w, rnd_w,  sty_w,  a0,  b0
            param_bnds = [0,    1e-3,   0,       0,      1e-3,   0,     1,   1;    % lb
                          1,    10,     1,       1,      5e2,    10,    1,   1;    % ub
                          0,    0.5,    0,       0,      1e-2,   0,     1,   1;    % plb
                          1,    4,      1,       1,      5e2,    10,    1,   1;    % pub
                          0.7,  0.2,    0.2,     0.09,   38,     2.5,   1,   1];   % st
    
    end
    
    hFitParams.param_bnds = param_bnds;
    hFitParams.kfolds = 4;
    hFitParams.iters = 1;
    hFitParams.linear_env = 0;
    hFitParams.optim = 'bads'; % 'bads' or 'fmin'
    hFitParams.rnd_flag = 0;
    
    %% Find subj/sess list
    
    switch hFitParams.task
        case '1D_Inter'
            subjIndices = find(contains({hMABall.task}, hFitParams.task) ...
            & cellfun(@(x) sum(x)/numel(x), {hMABall.structured}) == 0.5 ...
            & [hMABall.session] == hFitParams.sess);
        otherwise
            subjIndices = find(contains({hMABall.task}, hFitParams.task) ...
                & [hMABall.session] == hFitParams.sess);
    end
    hFitParams.subjIndices = subjIndices;
    
    
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
    
    save([savepath 'hFitParams'], 'hFitParams');

    copyfile(slurm_file, savepath);

end

