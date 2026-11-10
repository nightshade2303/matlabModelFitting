%%
load("output/r_recovery/20261110_modelRec/sim-Qdghl_fit-Qdghl_17.mat","simParamSet");
simParamSet.n_rounds=100;
tic;fL_U = computeFitnessLandscape(simParamSet,10);toc;

x = fL_U.simParams(:,2);
y = fL_U.simParams(:,8);
r_U = mean(fL_U.perfindex,2);
r_S = mean(fL_S.perfindex,2);
r_I = mean(fL_I.perfindex,2);

[xi,yi] = meshgrid(linspace(min(x),max(x),100),linspace(min(y),max(y),100));
xq = repmat(max(simParamSet.param_bnds),numel(xi(:)),1);
xq(:,2) = xi(:);
xq(:,end) = yi(:);
tic; ri_U = griddatan(fL_U.simParams(:,[1,2,3,6,7,8]), r_U, xq(:,[1,2,3,6,7,8])); toc;
tic; ri_S = griddatan(fL_S.simParams(:,[1,2,3,6,7,8]), r_S, xq(:,[1,2,3,6,7,8])); toc;
tic; ri_I = griddatan(fL_I.simParams(:,[1,2,3,6,7,8]), r_I, xq(:,[1,2,3,6,7,8])); toc;

figure; imagesc(reshape(ri_U,100,100))
figure; imagesc(reshape(ri_S,100,100))
figure; imagesc(reshape(ri_I,100,100))
figure; imagesc(reshape(ri_S-ri_U,100,100)); colormap(redblueu);
figure; imagesc(reshape(ri_S-ri_I,100,100)); colormap(redblueu);
figure; imagesc(reshape(ri_U-ri_I,100,100)); colormap(redblueu);

%% add paths and folders

addpath('.\HelperFunctions\');
addpath(genpath('.\bads-master\'));

load('.\Data\hMABdat.mat','hMABenvs','hMABall');

datadir = '\\10.36.17.152\ashesh\Results\MAB\hfits\';

% line colors
Scol = [.29,.86,.25]; % correlated 
Ucol = [.96,.46,.16]; % independent
Icol = [.75,.36,.98]; % interleaved 


%% 

simModel = "Qdgsl";
n_rounds = 100;
n_reps = 10000;

simParamSet = struct;
simParamSet.linear_env = 0;
simParamSet.rnd_seed = 1;
simParamSet.n_rounds = n_rounds;
simParamSet.n_trials = 60;
simParamSet.model = simModel;

[simParamSet.algo,simParamSet.param_bnds,param_names,free_params] = modelParamBounds(simModel);
free_params = find(free_params);
param_bnds = simParamSet.param_bnds;

simParamSet_S = simParamSet;
simParamSet_S.rprobset = hMABenvs.Str1D;
fL_S = computeFitnessLandscape(simParamSet_S,n_reps);

simParamSet_U = simParamSet;
simParamSet_U.rprobset = hMABenvs.Unstr1D;
fL_U = computeFitnessLandscape(simParamSet_U,n_reps);

simParamSet_I = simParamSet;
simParamSet_I.rprobset = cat(1,hMABenvs.Str1D,hMABenvs.Unstr1D);
fL_I = computeFitnessLandscape(simParamSet_I,n_reps);

simParams = fL_I.simParams;

%% load best fit params over rounds for the same model

datafolder = '20260508_hfits'; % no CV
datapath = [datadir, datafolder, '\'];

% S + U
Sfname = sprintf('Ss1r5_%s_result.mat',simModel);
load([datapath,Sfname], 'hFitResult');
Ss1r5_params = hFitResult.params(:,:,free_params);
Sfname = sprintf('Ss2r5_%s_result.mat',simModel);
load([datapath,Sfname], 'hFitResult');
Ss2r5_params = hFitResult.params(:,:,free_params);
Ufname = sprintf('Us1r5_%s_result.mat',simModel);
load([datapath,Ufname], 'hFitResult');
Us1r5_params = hFitResult.params(:,:,free_params);
Ufname = sprintf('Us2r5_%s_result.mat',simModel);
load([datapath,Ufname], 'hFitResult');
Us2r5_params = hFitResult.params(:,:,free_params);

SU_rd_bins = 1:7;

% Inter
Iset = "I30s1r5";
I_rd_bins = 1:6;
% Iset = "Is1r5";
% I_rd_bins = 1:5;
Ifname = sprintf('%s_%s_result.mat',Iset,simModel);
load([datapath,Ifname], 'hFitResult');

Is1r5_params = hFitResult.params(:,:,free_params,3); 

% Block
Bfname = sprintf('Bs1r5_%s_result.mat',simModel);
load([datapath,Bfname], 'hFitResult');
Bs1r5_params = hFitResult.params(:,:,free_params);   
SUind = find(cellfun(@(x) x(1)==1, {hMABall(hFitResult.subjIndices).structured}));
USind = find(cellfun(@(x) x(1)==0, {hMABall(hFitResult.subjIndices).structured}));

B1_rd_bins = 1:3;
B2_rd_bins = 4:6;

%% loop through param combinations

n_grid_pts = 50;
rew_metric = "reward";

all_params = cat(1,Us1r5_params,Ss1r5_params);
m_fit_params = squeeze(mean(mean(all_params,2),1)); % mean fit params

for p1 = 1 : (length(free_params)-1)
    p1_bins = linspace(param_bnds(1,free_params(p1)),param_bnds(2,free_params(p1)),n_grid_pts+2);
    p1_bins = p1_bins(2:end-1);
    for p2 = p1+1 : length(free_params)
        p2_bins = linspace(param_bnds(1,free_params(p2)),param_bnds(2,free_params(p2)),n_grid_pts+2);
        p2_bins = p2_bins(2:end-1);
        [p1_grid,p2_grid] = meshgrid(p1_bins,p2_bins);
        grid_params = repmat(m_fit_params',n_grid_pts^2,1);
        grid_params(:,p1) = p1_grid(:);
        grid_params(:,p2) = p2_grid(:);

        r_U = griddatan(simParams(:,free_params),mean(fL_U.(rew_metric),2),grid_params);
        r_grid_U = reshape(r_U,n_grid_pts,n_grid_pts);
        r_S = griddatan(simParams(:,free_params),mean(fL_S.(rew_metric),2),grid_params);
        r_grid_S = reshape(r_S,n_grid_pts,n_grid_pts);
        r_I = griddatan(simParams(:,free_params),mean(fL_I.(rew_metric),2),grid_params);
        r_grid_I = reshape(r_I,n_grid_pts,n_grid_pts);


        h = figure;
        tiledlayout(1,3,"TileSpacing","compact","Padding","compact");
        nexttile;
        imagesc(p1_bins,p2_bins,r_grid_U);
        xlabel(param_names(free_params(p1)));
        ylabel(param_names(free_params(p2)));
        title('U');
        hold on;
        plot(mean(Us1r5_params(:,:,p1)),mean(Us1r5_params(:,:,p2)),'ko-','LineWidth',2,'MarkerFaceColor',Ucol);
        plot(mean(Us2r5_params(:,:,p1)),mean(Us2r5_params(:,:,p2)),'ko-','LineWidth',2,'MarkerFaceColor',Ucol);

        nexttile;
        imagesc(p1_bins,p2_bins,r_grid_S);
        xlabel(param_names(free_params(p1)));
        ylabel(param_names(free_params(p2)));
        title('S');
        hold on;
        plot(mean(Ss1r5_params(:,:,p1)),mean(Ss1r5_params(:,:,p2)),'ko-','LineWidth',2,'MarkerFaceColor',Scol);
        plot(mean(Ss2r5_params(:,:,p1)),mean(Ss2r5_params(:,:,p2)),'ko-','LineWidth',2,'MarkerFaceColor',Scol);
        
        nexttile;
        imagesc(p1_bins,p2_bins,r_grid_I);
        xlabel(param_names(free_params(p1)));
        ylabel(param_names(free_params(p2)));
        title('I');
        hold on;
        plot(mean(Is1r5_params(:,:,p1)),mean(Is1r5_params(:,:,p2)),'ko-','LineWidth',2,'MarkerFaceColor',Icol);
    
        sgtitle(simModel);
    end
end
