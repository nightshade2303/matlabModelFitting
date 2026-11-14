% generateRecoveryParamSets.m

%% save directory

savedir = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/output/r_recovery/';

%% Load envs dataset
% what is this?
% if exist("rMABpre","var") ~= 1
%     load(".\Data\rMABpre.mat", "rMABenvs");
% end

%% Set paths

addpath('./HelperFunctions/');

%% make save folder

savefolder = [char(datetime("now","Format","yyyyMMdd")), '_',  'modelRec',  '/'];
savepath = [savedir, savefolder];
    
if exist(savepath, 'dir') ~= 7
    mkdir(savepath);    
    mkdir([savepath 'Jobs']);
elseif numel(dir(savepath)) > 3
    disp('Directory is not empty...');
    return;
end


%% Define simulation and recovery model sets

n_reps = 100;
rounds = [15];
n_blocks_per_sess = 2;
n_trials_per_block = 150;

% Sim param set
mySimParamSet = struct;
mySimParamSet.rnd_seed = 1;
mySimParamSet.linear_env = 1;
mySimParamSet.n_rounds = 15;
% need all of the following for paramrec
mySimParamSet.n_trials = n_blocks_per_sess*n_trials_per_block;
mySimParamSet.n_trials_per_block = n_trials_per_block;
mySimParamSet.n_blocks_per_round = n_blocks_per_sess;
% environment type; rprobset is generated (seeded) per n_rounds inside the loop below
% task = 'str'; % structured
task = 'unstr'; % unstructured

% true  = param recovery only (fit each sim model with itself)
% false = model recovery (fit each sim model with every model in modelList)
paramRecOnly = false;

% Fit params set
myFitParamSet = struct;
myFitParamSet.crossval = 'none';
myFitParamSet.optim = 'bads';
myFitParamSet.kfolds = 1;
myFitParamSet.iters = 5;
myFitParamSet.linear_env = 1;
myFitParamSet.rnd_flag = 0;

modelList = ["Qdghl", "Qdhl", "Qghl", "Qdgh", "Qdg", "Q", "Qd", "Qdh", "Qh", "Bduhl", "Gduhl", "Ighl"];
% modelList = ["Qdghl"];
cmds = {}; % sbatch commands, written to <savepath>/submit_jobs_rec.sh at the end
for simModel = modelList

    simParamSet = mySimParamSet; % Copy simulation parameters
    simParamSet.model = simModel;
    [simParamSet.algo, simParamSet.param_bnds] = modelParamBounds(char(simModel));

    if paramRecOnly
        fitModels = simModel;  % sim all, fit same (only param recovery)
    else
        fitModels = modelList; % sim all, fit all (model + param recovery)
    end

    for recModel = fitModels
        fitParamSet = myFitParamSet; % Copy fit parameters
        fitParamSet.model = recModel;
        [fitParamSet.algo, fit_param_bnds] = modelParamBounds(char(recModel));
        fitParamSet.param_bnds = [fit_param_bnds; fit_param_bnds; mean(fit_param_bnds,1)];

        rounds_str = sprintf('%u,',rounds); rounds_str = rounds_str(1:end-1);
        recName = ['sim-', char(simModel), '_fit-' char(recModel)];

        % format sbatch command
        cmd = sprintf('sbatch -J %s_%s -t 3-00:00 --array=%s ../../rParamRec_submit.sh %s %s', ...
            char(simModel),char(recModel),rounds_str,savefolder(1:end-1),recName);
        disp(cmd);
        cmds{end+1} = cmd; %#ok<SAGROW>

        for n_rounds = rounds
            simParamSet.n_rounds = n_rounds;
            % seeded environment with exactly n_rounds sessions: every fit model of
            % this sim model (and every rerun) gets the identical reward schedule
            rng(simParamSet.rnd_seed, 'twister');
            simParamSet.rprobset = generateRProbSet(n_rounds, n_trials_per_block, 4, strcmp(task,'unstr'), n_blocks_per_sess);
            % save param sets
            recName = ['sim-', char(simModel), '_fit-' char(recModel) '_' num2str(n_rounds)];
            save([savepath, recName], 'simParamSet', 'fitParamSet', 'n_reps');
        end


    end
end

%% write submit script for this folder

fid = fopen([savepath 'submit_jobs_rec.sh'], 'w');
fprintf(fid, '#!/bin/bash\n');
fprintf(fid, 'cd %sJobs/\n', savepath);
fprintf(fid, '%s\n', cmds{:});
fclose(fid);
fprintf('Submit with: bash %ssubmit_jobs_rec.sh\n', savepath);



