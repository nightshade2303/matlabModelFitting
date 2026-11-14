function [bnds, transforms, info] = fitParamBounds(fit_params, model_bnds, free_params, opts)
%FITPARAMBOUNDS  Simulation bounds from Gaussian fits to the rats' fitted params.
%
%   [bnds, transforms, info] = fitParamBounds(fit_params, model_bnds, free_params, opts)
%
% For each free param:
% 1. Space: the first transform under which the rats look normal (Anderson-
%    Darling test, adtest), trying 'none' -> 'log' -> 'logit' (logit only for
%    params whose model bounds lie in [0, 1]). If none passes, the raw space
%    ('none') is used and the param is flagged as not normal.
% 2. Two Gaussian fits in that space:
%    - mle:   fitdist(y, 'Normal'): mu is the maximum-likelihood mean;
%             fitdist reports sigma with the unbiased (n-1) variance
%    - curve: least-squares fit of a*exp(-((x-b)/c)^2) to the histogram
%             counts (n_bins bins), mu = b, sd = c/sqrt(2). b is kept inside
%             the data range and c below sqrt(2)*range; a fit that ends at
%             one of these limits is flagged (curve_ok = false), since it is
%             using the tail of a Gaussian to follow a skewed histogram
% 3. Bounds: mu +/- n_sd*sd of the fit chosen by fit_method, transformed back
%    to raw values and clipped to the model bounds.
%
% Inputs
%   fit_params   n_rats x n_params fitted params (from loadFitParams)
%   model_bnds   2 x n_params [lb; ub] (from modelParamBounds)
%   free_params  indices of the free params
%   opts         (optional) struct with fields
%                  alpha       (0.05)   significance level of the normality test
%                  n_sd        (2)      half-width of the range, in SDs
%                  offset      (1e-3)   c in transformParams
%                  fit_method  ('mle')  fit that sets the bounds: 'mle' or 'curve'
%                                       ('curve' uses mle for params whose curve
%                                       fit failed or is flagged)
%                  n_bins      (8)      histogram bins for the curve fit
%
% Outputs
%   bnds        2 x n_params raw [lb; ub] for simulation (fixed params unchanged)
%   transforms  1 x n_params string array, space each param is sampled in
%   info        struct of per-param results (1 x n_params arrays / cells):
%                 tried, pvals     transforms tested and their A-D p-values
%                 space, normal    chosen space; false if no transform passed
%                 mle_mu, mle_sd, mle_mu_ci, mle_sd_ci (2 x n_params)
%                 curve_a, curve_mu, curve_sd, curve_mu_ci, curve_sd_ci,
%                 curve_r2, curve_ok
%                 hist_edges, hist_counts   the histogram the curve was fit to
%                 method_used, mu, sd       fit that set the bounds and its values
%               plus opts

defaults = struct('alpha', 0.05, 'n_sd', 2, 'offset', 1e-3, 'fit_method', 'mle', 'n_bins', 8);
if nargin < 4, opts = struct; end
for f = fieldnames(defaults)'
    if ~isfield(opts, f{1}), opts.(f{1}) = defaults.(f{1}); end
end
c = opts.offset;

n_params   = size(model_bnds, 2);
bnds       = model_bnds;
transforms = repmat("none", 1, n_params);

info = struct;
info.opts        = opts;
info.tried       = cell(1, n_params);
info.pvals       = cell(1, n_params);
info.space       = repmat("none", 1, n_params);
info.normal      = true(1, n_params);
info.mle_mu      = NaN(1, n_params);
info.mle_sd      = NaN(1, n_params);
info.mle_mu_ci   = NaN(2, n_params);
info.mle_sd_ci   = NaN(2, n_params);
info.curve_a     = NaN(1, n_params);
info.curve_ok    = false(1, n_params);
info.curve_mu    = NaN(1, n_params);
info.curve_sd    = NaN(1, n_params);
info.curve_mu_ci = NaN(2, n_params);
info.curve_sd_ci = NaN(2, n_params);
info.curve_r2    = NaN(1, n_params);
info.hist_edges  = cell(1, n_params);
info.hist_counts = cell(1, n_params);
info.method_used = strings(1, n_params);
info.mu          = NaN(1, n_params);
info.sd          = NaN(1, n_params);

for p = free_params(:)'
    x = fit_params(:, p);

    %% 1. space: first transform that looks normal, else raw
    candidates = ["none", "log"];
    if model_bnds(1,p) >= 0 && model_bnds(2,p) <= 1
        candidates(end+1) = "logit"; %#ok<AGROW>
    end
    pvals = NaN(1, numel(candidates));
    chosen = "";
    for k = 1 : numel(candidates)
        y = transformParams(x, candidates(k), c, 'forward');
        if std(y) == 0
            continue; % all rats at one value: nothing to test
        end
        % adtest warns when p is beyond its table (reported as 0.0005 / 0.5); that's fine here
        w = warning('off', 'stats:adtest:OutOfRangePLow');
        w(2) = warning('off', 'stats:adtest:OutOfRangePHigh');
        [h, pvals(k)] = adtest(y, 'Alpha', opts.alpha);
        warning(w);
        if h == 0
            chosen = candidates(k);
            break;
        end
    end
    info.tried{p} = candidates(1:k);
    info.pvals{p} = pvals(1:k);
    if chosen == ""
        chosen = "none";
        info.normal(p) = false;
    end
    info.space(p) = chosen;
    transforms(p) = chosen;
    y = transformParams(x, chosen, c, 'forward');

    if std(y) == 0
        % all rats at one value: fix the param there
        info.method_used(p) = "constant";
        info.mu(p) = y(1); info.sd(p) = 0;
        bnds(:, p) = transformParams([y(1); y(1)], chosen, c, 'inverse');
        bnds(:, p) = min(max(bnds(:, p), model_bnds(1,p)), model_bnds(2,p));
        continue;
    end

    %% 2a. maximum likelihood fit
    pd = fitdist(y, 'Normal');
    ci = paramci(pd);                         % [mu_lo sd_lo; mu_hi sd_hi]
    info.mle_mu(p) = pd.mu;
    info.mle_sd(p) = pd.sigma;
    info.mle_mu_ci(:, p) = ci(:, 1);
    info.mle_sd_ci(:, p) = ci(:, 2);

    %% 2b. least-squares curve fit to the histogram
    [counts, edges] = histcounts(y, opts.n_bins);
    info.hist_edges{p} = edges;
    info.hist_counts{p} = counts;
    if nnz(counts) >= 3
        centers = (edges(1:end-1) + edges(2:end))' / 2;
        bw = edges(2) - edges(1);
        a0 = numel(y) * bw / (pd.sigma * sqrt(2*pi)); % peak height of the MLE curve in counts
        rng_y = edges(end) - edges(1);
        lo = [0,   edges(1),   1e-6*rng_y];
        hi = [Inf, edges(end), sqrt(2)*rng_y];
        start = min(max([a0, pd.mu, pd.sigma*sqrt(2)], lo), hi);
        fo = fitoptions('gauss1', 'StartPoint', start, 'Lower', lo, 'Upper', hi);
        [cf, gof] = fit(centers, counts(:), 'gauss1', fo);
        cci = confint(cf);                    % rows lo/hi, cols a1 b1 c1
        tol = 1e-3 * rng_y;
        info.curve_ok(p) = abs(cf.b1 - lo(2)) > tol && abs(cf.b1 - hi(2)) > tol && abs(cf.c1 - hi(3)) > tol;
        info.curve_a(p)  = cf.a1;
        info.curve_mu(p) = cf.b1;
        info.curve_sd(p) = cf.c1 / sqrt(2);
        info.curve_mu_ci(:, p) = cci(:, 2);
        info.curve_sd_ci(:, p) = cci(:, 3) / sqrt(2);
        info.curve_r2(p) = gof.rsquare;
    end

    %% 3. bounds from the chosen fit
    if strcmp(opts.fit_method, 'curve') && info.curve_ok(p)
        info.method_used(p) = "curve";
        info.mu(p) = info.curve_mu(p); info.sd(p) = info.curve_sd(p);
    else
        info.method_used(p) = "mle"; % also when the curve fit failed or is flagged
        info.mu(p) = info.mle_mu(p); info.sd(p) = info.mle_sd(p);
    end
    y_bnds = info.mu(p) + [-1; 1] * opts.n_sd * info.sd(p);
    bnds(:, p) = transformParams(y_bnds, chosen, c, 'inverse');

    % keep within the model bounds
    bnds(:, p) = min(max(bnds(:, p), model_bnds(1,p)), model_bnds(2,p));
end

end
