function [p0_max, n_evals] = find_p0_max(G, c_fbl1, p0_lo, p0_hi, tol, sim_opts)
% FIND_P0_MAX  Bisection search for the largest source amplitude p0 for
% which the numerical scheme of simulate_1d_harmonics does NOT diverge
% (NaN/Inf), for a given G and c_fbl1.
%
%   p0_max = find_p0_max(G, c_fbl1)
%   p0_max = find_p0_max(G, c_fbl1, p0_lo, p0_hi, tol, sim_opts)
%
% Inputs:
%   G, c_fbl1  as in simulate_1d_harmonics
%   p0_lo      starting p0 known NOT to diverge (default 500 Pa)
%   p0_hi      starting p0 expected to diverge (default 200e3 Pa)
%   tol        final resolution of the search [Pa] (default 200 Pa)
%   sim_opts   cell array of name-value pairs forwarded as they are to
%              simulate_1d_harmonics (e.g. {'N_g', 1e11, 'f', 20e3})
%
% Outputs:
%   p0_max     the LAST p0 that gave a finite solution (WITHOUT safety
%              margin; that is applied in generate_dataset_1d.m).
%   n_evals    number of simulations run (informative).
%
% If p0_lo already diverges, an error is raised (lower p0_lo). If p0_hi does
% NOT diverge, a warning is issued and p0_hi is returned as it is (raise
% p0_hi and repeat).

if nargin < 3 || isempty(p0_lo),  p0_lo = 500;    end
if nargin < 4 || isempty(p0_hi),  p0_hi = 200e3;  end
if nargin < 5 || isempty(tol),    tol = 200;      end
if nargin < 6, sim_opts = {}; end

[~, ~, div_lo] = simulate_1d_harmonics(G, c_fbl1, p0_lo, sim_opts{:});
n_evals = 1;
if div_lo
    error('find_p0_max:p0_lo_diverges', ...
        'p0_lo=%.0f Pa already diverges for G=%.0f Pa -- lower p0_lo.', p0_lo, G);
end

[~, ~, div_hi] = simulate_1d_harmonics(G, c_fbl1, p0_hi, sim_opts{:});
n_evals = n_evals + 1;
if ~div_hi
    warning('find_p0_max:p0_hi_no_diverge', ...
        ['p0_hi=%.0f Pa does not diverge for G=%.0f Pa -- p0_hi is returned ' ...
         'as it is. Raise p0_hi to explore higher amplitudes.'], p0_hi, G);
    p0_max = p0_hi;
    return
end

max_iter = 40;
iter = 0;
while (p0_hi - p0_lo) > tol && iter < max_iter
    p0_mid = 0.5*(p0_lo + p0_hi);
    [~, ~, div_mid] = simulate_1d_harmonics(G, c_fbl1, p0_mid, sim_opts{:});
    n_evals = n_evals + 1;
    iter = iter + 1;
    if div_mid
        p0_hi = p0_mid;
    else
        p0_lo = p0_mid;
    end
end

p0_max = p0_lo;
end
