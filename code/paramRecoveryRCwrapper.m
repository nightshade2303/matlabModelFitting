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

tic;
switch fitParamSet.crossval
    case 'none'
        [recParams, simParams, rec_nll, rec_pR2, rec_aic, rec_bic] = paramRecovery(simParamSet, fitParamSet, n_reps);
        % save fit results to disk
        save([simRecPath '_result.mat'],'recParams','simParams','rec_nll','rec_pR2','rec_aic','rec_bic','n_reps');
    otherwise 
        [recParams, simParams, rec_nll, rec_pR2] = paramRecovery(simParamSet, fitParamSet, n_reps);
        % save fit results to disk
        save([simRecPath '_result.mat'],'recParams','simParams','rec_nll','rec_pR2','n_reps');
end
disp('Param recovery completed.');
toc;
end