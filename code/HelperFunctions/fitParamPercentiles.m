function [prc_bnds, fit_params] = fitParamPercentiles(fitDir, prct)
%FITPARAMPERCENTILES  Percentiles of the fitted params across rats.
%
%   [prc_bnds, fit_params] = fitParamPercentiles(fitDir, prct)
%
% Inputs
%   fitDir  folder with rFit_<id>.mat files (each holding rFitResult, as saved
%           by fitRatDataRCwrapper)
%   prct    1 x 2 percentiles, e.g. [5 95]
%
% Outputs
%   prc_bnds    2 x n_params, row 1 = prct(1) percentile, row 2 = prct(2)
%               percentile of each param across fits
%   fit_params  n_fits x n_params fitted params, one row per rat (per day group
%               for daywise/weekly fits). Rows with NaN (skipped groups) are dropped.

files = dir(fullfile(fitDir, 'rFit_*.mat'));
if isempty(files)
    error('No rFit_*.mat files in %s', fitDir);
end

fit_params = [];
for f = 1 : numel(files)
    S = load(fullfile(files(f).folder, files(f).name), 'rFitResult');
    p = S.rFitResult.params;                 % 1 x n_groups x n_params
    p = reshape(p, [], size(p, 3));          % n_groups x n_params
    fit_params = [fit_params; p]; %#ok<AGROW>
end

fit_params = fit_params(~any(isnan(fit_params), 2), :);
if isempty(fit_params)
    error('All fits in %s contain NaN params', fitDir);
end

prc_bnds = prctile(fit_params, prct, 1);

end
