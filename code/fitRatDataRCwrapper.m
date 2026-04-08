function rFitResult = fitRatDataRCwrapper(dataPath,fitPath,subj_id)
%fitHumanDataRCwrapper load params file, data file and save fits

addpath('./HelperFunctions/');
addpath(genpath('./bads-master/'));

% load fitParams
load([fitPath,'rFitParams.mat'], 'rFitParams');

% load dataset
% load([dataPath,'rMABstr_20260324.mat'], 'rMABstr');
load([dataPath,'rMABdat_20260212.mat'], 'rMABpre');

% fit session
% rFitResult = fitRatData(rMABstr, rFitParams, subj_id);
rFitResult = fitRatData(rMABpre, rFitParams, subj_id);

% save result
save([fitPath,'rFit_' num2str(subj_id)], 'rFitResult');

end