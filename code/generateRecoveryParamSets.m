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
rounds = [10];

% Sim param set
mySimParamSet = struct;
mySimParamSet.rnd_seed = 1;
mySimParamSet.linear_env = 1;
mySimParamSet.n_rounds = 10;
mySimParamSet.n_trials = 100;
% mySimParamSet.rprobset = generateRProbSet(mySimParamSet.n_rounds, 4, false); task = 'str'; % structured
mySimParamSet.rprobset = generateRProbSet(mySimParamSet.n_rounds, 4, true); task = 'unstr'; % unstructured

% Fit params set
myFitParamSet = struct;
myFitParamSet.crossval = 'none';
myFitParamSet.optim = 'bads';
myFitParamSet.kfolds = 4;
myFitParamSet.iters = 5;
myFitParamSet.linear_env = 1;
myFitParamSet.rnd_flag = 0;

modelList = ["Q","Qs","Qgs","Qsl", "Qgsl", "Qdgsl", "B", "Bs","Bsl", "Bus","Busl", "Bdusl", "G", "Gs", "Gsl", "Gus", "Gusl", "Gdus", "Gdusl"];
for simModel = modelList    
    
    simParamSet = mySimParamSet; % Copy simulation parameters
    simParamSet.model = simModel;    
    [simParamSet.algo, simParamSet.param_bnds] = modelParamBounds(char(simModel));

    for recModel = modelList % sim all, fit all (model + param recovery)
    % for recModel = simModel % sim all, fit same (only param recovery)
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
        
        for n_rounds = rounds
            simParamSet.n_rounds = n_rounds;
            % save param sets
            recName = ['sim-', char(simModel), '_fit-' char(recModel) '_' num2str(n_rounds)];
            save([savepath, recName], 'simParamSet', 'fitParamSet', 'n_reps');
        end
        
        
    end
end



