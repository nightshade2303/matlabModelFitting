function [recParams, simParams, rec_nll, rec_pR2] = paramRecoveryRCwrapper(simRecPath)
%paramRecoveryRCwrapper load sim and params file and save fits

addpath('./HelperFunctions/');

% load sim and fitParamSets
load([simRecPath '.mat'], 'simParamSet','fitParamSet','n_reps');

switch fitParamSet.optim
    case 'bads'
        addpath(genpath('./bads-master/'));
    case 'fmin'
        addpath('./fminsearchbnd/');
end

% start a local parallel pool sized to the Slurm allocation (after addpath so workers inherit the path)
n_workers = str2double(getenv('SLURM_CPUS_PER_TASK'));
if isnan(n_workers) || n_workers < 1
    n_workers = feature('numcores');
end
if isempty(gcp('nocreate')) && n_workers > 1
    pc = parcluster('Processes');
    % per-job storage dir so concurrent array tasks don't clash in ~/.matlab
    job_storage = fullfile(tempdir, ['matlab_pool_' getenv('SLURM_JOB_ID') '_' getenv('SLURM_ARRAY_TASK_ID')]);
    if ~exist(job_storage, 'dir'); mkdir(job_storage); end
    pc.JobStorageLocation = job_storage;
    pc.NumWorkers = n_workers;
    parpool(pc, n_workers);
end

tic;
% rec_nll is the in-sample NLL (crossval 'none') or the summed held-out NLL (crossval);
% rec_info{rep} holds n_obs and, when cross-validating, fold-wise NLL / trial counts / params
[recParams, simParams, rec_nll, rec_pR2, rec_aic, rec_bic, rec_info] = paramRecovery(simParamSet, fitParamSet, n_reps);
% save fit results to disk
save([simRecPath '_result.mat'],'recParams','simParams','rec_nll','rec_pR2','rec_aic','rec_bic','rec_info','n_reps');
disp('Param recovery completed.');
toc;
end