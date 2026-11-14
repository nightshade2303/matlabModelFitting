function [fit_params, cohort, fit_info] = loadFitParams(fitDirs, opts)
%LOADFITPARAMS  Fitted params of all rats in one or more fit folders, without bad fits.
%
%   [fit_params, cohort, fit_info] = loadFitParams(fitDirs, opts)
%
% Inputs
%   fitDirs  a folder, or a cell array of folders, with rFit_<id>.mat files
%            (each holding rFitResult, as saved by fitRatDataRCwrapper)
%   opts     (optional) struct with field
%              max_nll_per_trial  (log(4))  fits with nll/nTrials >= this are
%                                 excluded; the default is the NLL per trial of
%                                 choosing among 4 arms at random
%
% A fit (one rat, or one day group for daywise/weekly fits) is excluded if
%   - any param is NaN (group skipped), or
%   - its nll is not finite or <= 0, or it has no trials (nothing was fit), or
%   - its nll per trial is >= max_nll_per_trial (no better than chance)
%
% Outputs
%   fit_params  n_kept x n_params fitted params of the kept fits
%   cohort      n_kept x 1 index into fitDirs of the folder each row came from
%   fit_info    table with one row per fit (kept and excluded): rat, cohort,
%               file, group, nll, nTrials, nll_per_trial, pR2 (for reference
%               only), excluded, reason

if ~iscell(fitDirs)
    fitDirs = {fitDirs};
end
if nargin < 2, opts = struct; end
if ~isfield(opts, 'max_nll_per_trial'), opts.max_nll_per_trial = log(4); end

fit_params = [];
rat = strings(0,1); coh = []; file = strings(0,1); grp = [];
nll = []; pR2 = []; nTr = [];
for d = 1 : numel(fitDirs)
    files = dir(fullfile(fitDirs{d}, 'rFit_*.mat'));
    if isempty(files)
        error('No rFit_*.mat files in %s', fitDirs{d});
    end
    for f = 1 : numel(files)
        S = load(fullfile(files(f).folder, files(f).name), 'rFitResult');
        r = S.rFitResult;
        p = reshape(r.params, [], size(r.params, 3));   % n_groups x n_params
        n_g = size(p, 1);
        fit_params = [fit_params; p]; %#ok<AGROW>
        rat  = [rat;  repmat(string(r.ratName), n_g, 1)]; %#ok<AGROW>
        coh  = [coh;  d*ones(n_g, 1)]; %#ok<AGROW>
        file = [file; repmat(string(files(f).name), n_g, 1)]; %#ok<AGROW>
        grp  = [grp;  (1:n_g)']; %#ok<AGROW>
        nll  = [nll;  reshape(r.nll, [], 1)]; %#ok<AGROW>
        pR2  = [pR2;  reshape(r.pR2, [], 1)]; %#ok<AGROW>
        if isfield(r, 'nTrialsGroup')
            nTr = [nTr; reshape(r.nTrialsGroup, [], 1)]; %#ok<AGROW>
        else
            nTr = [nTr; repmat(r.nTrials, n_g, 1)]; %#ok<AGROW>
        end
    end
end

% reasons for excluding each fit (first one that applies)
reason = strings(size(fit_params, 1), 1);
nll_per_trial = nll ./ nTr;
reason(nll_per_trial >= opts.max_nll_per_trial) = "nll per trial >= " + sprintf('%.3f', opts.max_nll_per_trial) + " (no better than chance)";
reason(~isfinite(nll) | nll <= 0 | nTr == 0) = "nll <= 0 or NaN / no trials (nothing was fit)";
reason(any(isnan(fit_params), 2)) = "NaN params";
excluded = reason ~= "";

fit_info = table(rat, coh, file, grp, nll, nTr, nll_per_trial, pR2, excluded, reason, ...
    'VariableNames', {'rat', 'cohort', 'file', 'group', 'nll', 'nTrials', 'nll_per_trial', 'pR2', 'excluded', 'reason'});

fit_params = fit_params(~excluded, :);
cohort = coh(~excluded);
if isempty(fit_params)
    error('All fits were excluded');
end

end
