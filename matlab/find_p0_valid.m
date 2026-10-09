function [p0_valid, n_evals] = find_p0_valid(G, c_fbl1, p0_lo, p0_hi, tol, eps_max, sim_opts)
% FIND_P0_VALID  Bisection search for the p0 at which epsilon = max(|v|)/V0
% crosses the threshold eps_max (default 0.5, the validity criterion of
% Carreras-Casanova & Vanhille, Acoustics 2025, 7, 67), for a given G and
% c_fbl1. It is the sibling of find_p0_max.m: the same bisection, but on the
% PHYSICAL condition (v << V0) instead of the NUMERICAL divergence (NaN/Inf).
%
%   p0_valid = find_p0_valid(G, c_fbl1, p0_lo, p0_hi)
%   p0_valid = find_p0_valid(G, c_fbl1, p0_lo, p0_hi, tol, eps_max, sim_opts)
%
% Inputs:
%   G, c_fbl1  as in simulate_1d_harmonics / diag_vV0
%   p0_lo      starting p0 known to satisfy epsilon <= eps_max
%   p0_hi      starting p0 expected to exceed eps_max (normally the
%              numerical limit p0_max of that G: searching further makes no
%              sense, because the scheme has already diverged there)
%   tol        final resolution of the search [Pa] (default 200 Pa)
%   eps_max    epsilon threshold (default 0.5)
%   sim_opts   cell array forwarded as it is to diag_vV0
%
% Outputs:
%   p0_valid   the LAST p0 with epsilon <= eps_max. If epsilon does not
%              exceed eps_max even at p0_hi, a warning is issued and p0_hi
%              is returned as it is (physics is not the limiting factor in
%              that range; typically the numerical scheme is).
%   n_evals    number of simulations run (informative).

if nargin < 5 || isempty(tol),      tol = 200;   end
if nargin < 6 || isempty(eps_max),  eps_max = 0.5; end
if nargin < 7, sim_opts = {}; end

[mv_lo, ~] = diag_vV0(G, c_fbl1, p0_lo, sim_opts{:});
n_evals = 1;
if mv_lo > eps_max
    error('find_p0_valid:p0_lo_exceeds', ...
        'At p0_lo=%.0f Pa epsilon=%.2f is already exceeded (max|v|/V0=%.3f) for G=%.0f Pa -- lower p0_lo.', ...
        p0_lo, eps_max, mv_lo, G);
end

[mv_hi, ~] = diag_vV0(G, c_fbl1, p0_hi, sim_opts{:});
n_evals = n_evals + 1;
if mv_hi <= eps_max
    warning('find_p0_valid:p0_hi_not_exceeded', ...
        ['epsilon does not exceed %.2f at p0_hi=%.0f Pa for G=%.0f Pa -- ' ...
         'physics is not the limiting factor in this range. p0_hi is returned as it is.'], ...
        eps_max, p0_hi, G);
    p0_valid = p0_hi;
    return
end

max_iter = 40;
iter = 0;
while (p0_hi - p0_lo) > tol && iter < max_iter
    p0_mid = 0.5*(p0_lo + p0_hi);
    [mv_mid, ~] = diag_vV0(G, c_fbl1, p0_mid, sim_opts{:});
    n_evals = n_evals + 1;
    iter = iter + 1;
    if mv_mid > eps_max
        p0_hi = p0_mid;
    else
        p0_lo = p0_mid;
    end
end

p0_valid = p0_lo;
end
