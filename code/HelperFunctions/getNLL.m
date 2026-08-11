% getNLL.m
% get NLL for a specific model/ bunch of params
addpath(genpath('./bads-master'));
addpath('./HelperFunctions');
dataPath = '/mnt/pve/Homes/rishika/behavioralAnalysis/matlabModelFitting/data/';
load([dataPath,'rMABdat_20260212.mat'], 'rMABpre');
params = [0.75, 15, 2, 10, 2, 0.75];
algo = 'GPspacetime';
linear_env = 1;
num_choices = 4;
blocked_rats = {'Ivysaur', 'Jigglypuff', 'Vanillish', 'Whismur', 'Zorua', 'Emolga', 'Giratina', 'Lugia', 'Quaxly', 'Uxie'};
unblocked_rats = {'Inkay', 'Kirlia', 'Mesprit', 'Quilava', 'Raltz', 'Hoppip', 'Oddish', 'Phione', 'Goldeen', 'Jirachi'};
str_rats = {'HSB_F_006','HSB_F_007','HSB_F_008', 'HSB_F_009', 'HSB_F_010','HSB_F_011','HSB_F_012', 'HSB_F_013', 'HSB_F_014'};

% specify subject indices - if 4ABunstr in task list then select those animals
% subjIndices = find(contains({rMABpre.task}, rFitParams.task));
% subjIndices = find(contains({rMABstr.task}, rFitParams.task));
subjIndices = find(contains({rMABpre.ratName}, blocked_rats));
% subjIndices = find(contains({rMABpre.ratName}, str_rats));

rng(1, 'twister');
% n_subjects = numel(rFitParams.subjIndices);
weeks = 4:8;

subj_num = 1;
nll = NaN(length(subjIndices),1);
for subjID = subjIndices
    
    atleast_trials = 300; 
    trials = 1:300;
    sess_idx = find(ismember(rMABpre(subjID).weekOfTraining, weeks) & rMABpre(subjID).trialNums>=atleast_trials);
    if length(sess_idx)>30
        sess_idx = randsample(sess_idx, 30);
    else
        disp('Not enough sessions');
        sess_idx = [];
    end
    rounds = {sess_idx};

    choices = rMABpre(subjID).choices(rounds{1}, trials);
    rewards = rMABpre(subjID).rewards(rounds{1}, trials);
    [nll(subj_num, 1), ~, ~] = calcNLL(params, algo, choices, rewards, num_choices, linear_env);

    subj_num = subj_num+1
end