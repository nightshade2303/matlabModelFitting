% generateRatFitParams.m

%% Load dataset

% unstructured dataset
if exist("rMABpre","var") ~= 1
    load("../../data/rMABdat_20260212.mat", "rMABpre");
end

% structured dataset
% if exist("rMABpre","var") ~= 1
%     load("../../data/rMABstr_20260529.mat", "rMABpre");
% end

% lesioned dataset
% if exist("rMABlesions","var") ~= 1
%     load("../../data/rMABdat_20260212.mat", "rMABlesions");
% end

%% Initialise fit param struct and savedir

rFitParams = struct;

savedir = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/output/r_fits/20260815_allblock_unstr_rats/';

slurm_file = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/code/rMAB_submit.sh';  % change this maybe?

%% Task, sess and round structure

fitName = 'ub_all';
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
blocked_rats = {'Ivysaur', 'Jigglypuff', 'Vanillish', 'Whismur', 'Zorua', 'Emolga', 'Giratina', 'Lugia', 'Quaxly', 'Uxie'};
unblocked_rats = {'Inkay', 'Kirlia', 'Mesprit', 'Quilava', 'Raltz', 'Hoppip', 'Oddish', 'Phione', 'Goldeen', 'Jirachi'};
str_rats = {'HSB_F_006','HSB_F_007','HSB_F_008', 'HSB_F_009', 'HSB_F_010','HSB_F_011','HSB_F_012', 'HSB_F_013', 'HSB_F_014'};
dls_ub_rats = {'Torchic', 'Uxie', 'Vanillish', 'Yamper', 'Zorua'};
dls_uu_rats = {'Blissey', 'Darkrai', 'Inkay', 'Mesprit', 'Quilava', 'Raltz', 'Zacian', 'Alakazam'};
dls_rats = {'Torchic', 'Uxie', 'Vanillish', 'Yamper', 'Zorua', 'Blissey', 'Darkrai', 'Inkay', 'Mesprit', 'Quilava', 'Raltz', 'Zacian', 'Alakazam'};
dms_rats = {'Giratina', 'Hoppip', 'Ivysaur', 'Kirlia', 'Lugia', 'Nidorina', 'Quaxly', 'Togepi', 'Whismur', 'Xatu', 'Yanma'};
sham_rats = {'Emolga','Goldeen','Haxorus','Jigglypuff','Jirachi','Phione','Umbreon','Vulpix','test05022023'};
% specify subject indices - if 4ABunstr in task list then select those animals
% subjIndices = find(contains({rMABpre.task}, rFitParams.task));
% subjIndices = find(contains({rMABstr.task}, rFitParams.task));
subjIndices = find(contains({rMABpre.ratName}, blocked_rats));
% subjIndices = find(contains({rMABlesions.ratName}, dls_rats));

rFitParams.subjIndices = subjIndices;


%% Algo and fit details
% modelList = ["Qdsl", "Qdgsl"];
modelList = ["4dsl", "4dgsl", "4dgs"];
% modelList = ["Q","Qs","Qsl","Qd","Qds","Qdsl","Qg","Qgs","Qgsl","Qdg","Qdgs","Qdgsl",...
%              "B","Bs","Bsl","Bd","Bds","Bdsl","Bu","Bus","Busl","Bdu","Bdus","Bdusl"];
% modelList =               ["G","Gs","Gsl","Gd","Gds","Gdsl","Gu","Gus","Gusl","Gdu","Gdus","Gdusl"];
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
    cmd = sprintf('sbatch -J %s_%s -t 3-00:00 --array=1-10 %s %s/%s', ...
        char(fitData),char(fitModel),char(slurm_file),'20260815_allblock_unstr_rats',savefolder(1:end-1));
    disp(cmd);
end