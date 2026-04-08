function hFitResult = fitHumanDataRCwrapper(dataPath,fitPath,subj_id)
%fitHumanDataRCwrapper load params file, data file and save fits

addpath('./HelperFunctions/');
addpath(genpath('./bads-master/'));

% load fitParams
load([fitPath,'hFitParams'], 'hFitParams');

% load dataset
load([dataPath,'hMABdat.mat'], 'hMABall');

% fit session
hFitResult = fitHumanData(hMABall, hFitParams, subj_id);

% save result
save([fitPath,'hFit_' num2str(subj_id)], 'hFitResult');

end