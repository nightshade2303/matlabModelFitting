function hFitResult = fitHumanDataRCwrapper(dataPath, fitPath)
%fitHumanDataRCwrapper load params file, data file and save fits

addpath('./HelperFunctions/');

% load fitParams
load([fitPath,'.mat'], 'hFitParamSet');

c

% load dataset
load([dataPath,'hMABdat.mat'], 'hMABall');

tic;
% fit session
hFitResult = fitHumanData(hMABall, hFitParamSet);
disp('Fits completed.');
toc;

% save result
save([fitPath,'_result.mat'], 'hFitResult');

end