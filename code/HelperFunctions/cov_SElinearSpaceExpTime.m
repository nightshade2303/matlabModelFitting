function K = cov_SElinearExpTime(hyp, x, z, i)

if nargin<2, K = '4'; return; end
if nargin<3, z = []; end

xeqz = isempty(z);
dg = strcmp(z,'diag');

% unpack hyperparameters
ell = hyp(1);
ell_max = hyp(2);      % unused; retained for compatibility
tee = hyp(3);
sd_signal = hyp(4);

if dg
    K = sd_signal^2 * ones(size(x,1),1);

else
    if xeqz
        dist_ell = x(:,1) - x(:,1)';
        dist_tee = abs(x(:,2) - x(:,2)');
    else
        dist_ell = x(:,1) - z(:,1)';
        dist_tee = abs(x(:,2) - z(:,2)');
    end

    K = sd_signal^2 ...
        * exp(-dist_ell.^2 / (2*ell^2)) ...
        .* exp(-dist_tee / tee);
end

if nargin > 3
    K = [];
end