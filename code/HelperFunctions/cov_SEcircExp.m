function K = cov_SEcircExp(hyp, x, z, i)



if nargin<2, K = '4'; return; end              % report number of parameters
if nargin<3, z = []; end                                   % make sure, z exists
xeqz = isempty(z); dg = strcmp(z,'diag');                       % determine mode

% unpack hyperparams
ell = hyp(1);
ell_max = hyp(2); % num of arms per circle
tee = hyp(3);
sd_signal = hyp(4);

if dg
    K = sd_signal^2*ones(size(x,1),1);
else
    if xeqz
        dist_ell = squeeze(min(cat(3, abs(x(:,1)-x(:,1)'), ell_max-abs(x(:,1)-x(:,1)')), [], 3));
        dist_tee = abs(x(:,2)-x(:,2)');
        K = sd_signal^2 * exp(-dist_ell.^2 / (2*ell^2)) .* exp(-dist_tee / tee);
    else
        dist_ell = squeeze(min(cat(3, abs(x(:,1)-z(:,1)'), ell_max-abs(x(:,1)-z(:,1)')), [], 3));
        dist_tee = abs(x(:,2)-z(:,2)');
        K = sd_signal^2 * exp(-dist_ell.^2 / (2*ell^2)) .* exp(-dist_tee / tee);
    end
end

if nargin > 3
    K = [];
end

