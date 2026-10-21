function K = cov_SEexpTime(hyp, x, z, i)

if nargin<2, K = '3'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                    % determine mode

% unpack hyperparams
ell = hyp(1);
tee = hyp(2);
sd_signal = hyp(3);

if dg
    K = sd_signal^2*ones(size(x,1),1);
else
    if xeqz
        dist_ell = squeeze(abs(x(:,1)-x(:,1)'));
        dist_tee = abs(x(:,2)-x(:,2)');
        K = sd_signal^2 * exp(-dist_ell.^2 / (2*ell^2)) .* exp(-dist_tee / tee);
    else
        dist_ell = squeeze(abs(x(:,1)-z(:,1)'));
        dist_tee = abs(x(:,2)-z(:,2)');
        K = sd_signal^2 * exp(-dist_ell.^2 / (2*ell^2)) .* exp(-dist_tee / tee);
    end
end

if nargin > 3
    K = [];
end