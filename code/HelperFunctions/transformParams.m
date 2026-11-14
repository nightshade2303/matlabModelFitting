function y = transformParams(x, type, c, direction)
%TRANSFORMPARAMS  Forward or inverse transform of parameter values.
%
%   y = transformParams(x, type, c, direction)
%
%   type       'none', 'log' or 'logit'
%   c          offset that keeps values at the bounds finite
%   direction  'forward' (raw -> transformed) or 'inverse' (transformed -> raw)
%
%   log:   forward log(x + c),               inverse exp(y) - c
%   logit: forward logit(clip(x, c, 1-c)),   inverse 1 ./ (1 + exp(-y))
%          (for params bounded in [0, 1])

switch lower(string(type))
    case "none"
        y = x;
    case "log"
        if strcmp(direction, 'forward')
            y = log(x + c);
        else
            y = exp(x) - c;
        end
    case "logit"
        if strcmp(direction, 'forward')
            xc = min(max(x, c), 1 - c);
            y = log(xc ./ (1 - xc));
        else
            y = 1 ./ (1 + exp(-x));
        end
    otherwise
        error('Unknown transform "%s" (use none, log or logit)', type);
end

end
