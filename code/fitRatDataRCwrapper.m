function rFitResult = fitRatDataRCwrapper(dataPath,fitPath,subj_id)
%fitHumanDataRCwrapper load params file, data file and save fits

addpath('./HelperFunctions/');

% load fitParams
load([fitPath,'rFitParams.mat'], 'rFitParams');

switch rFitParams.optim
    case 'bads'
        addpath(genpath('./bads-master/'));
    case 'fmin'
        addpath('./fminsearchbnd/');
end

% load dataset
% load([dataPath,'rMABstr_20260529.mat'], 'rMABpre');
load([dataPath,'rMABdat_20260212.mat'], 'rMABpre');
% load([dataPath,'rMABdat_20260212.mat'], 'rMABlesions');

% fit session
% rFitResult = fitRatData(rMABstr, rFitParams, subj_id);
% rFitResult = fitRatData(rMABlesions, rFitParams, subj_id);
rFitResult = fitRatData(rMABpre, rFitParams, subj_id);

% save result
save([fitPath,'rFit_' num2str(subj_id)], 'rFitResult');

end