function generate_dataset_1d(varargin)
% GENERATE_DATASET_1D  Generate the dataset (G, p0) -> (p2, p3) used to
% train the MLP surrogate. For every G in c_fbl1_table.csv, p0 is swept up
% to the p0_max that applies to that medium:
%
%   - The NUMERICAL p0_max (find_p0_max.m: where the scheme diverges) is
%     computed. If at that point epsilon = max(|v|)/V0 already exceeds
%     eps_max (0.5 by default, the criterion of Carreras-Casanova & Vanhille,
%     Acoustics 2025, 7, 67, p. 21), the PHYSICAL p0_max is also computed
%     (find_p0_valid.m: where epsilon crosses eps_max).
%   - The SMALLER of the two is used as the upper limit of the sweep, as
%     analyze_p0_limits.m does to build p0_limits_by_G.csv. For almost all
%     G (all except G = 0, water) the real limit is the physical one.
%   - safety_margin is then applied to that already corrected p0_max (not
%     to the raw numerical one), so that the sweep does not sample exactly
%     at the edge.
%
% Usage (run from the matlab/ folder):
%   generate_dataset_1d()
%   generate_dataset_1d('n_p0', 25, 'safety_margin', 0.95)
%   generate_dataset_1d('force', true)   % redo EVERYTHING, even what exists
%
% Reads:
%   ../data/c_fbl1_table.csv   columns: G_kPa, c_fbl1[, origin]
%     Rows with an empty c_fbl1 are skipped, so the script can be relaunched
%     as values are added without losing what has already been computed.
%
% Writes/extends:
%   ../data/dataset_1d_p2p3.csv   columns: G_kPa, c_fbl1, p0_Pa, p2_Pa, p3_Pa
%     Each row is a simulated case. G values that already have rows are
%     skipped unless 'force' is true.
%
% It depends on diag_vV0.m, find_p0_valid.m, find_p0_max.m and simulate_1d_harmonics.m, 
% which must be in the same folder or on the path.)
%
% Optional parameters:
%   n_p0            TOTAL number of p0 values sampled per G (default 25),
%                   split into two segments (see below)
%   dense_frac      fraction of those n_p0 points placed in the dense
%                   segment near p0_max (default 0.4 -> with n_p0 = 25,
%                   10 dense points and 16 in the rest of the range)
%   dense_zone_frac fraction of the p0 RANGE (counted from p0_max downwards)
%                   that is considered the "dense zone" (default 0.3 -> the
%                   upper 30% of the range). With dense_frac = 0 or
%                   dense_zone_frac = 0 uniform sampling (linspace) is
%                   recovered.
%   safety_margin   factor applied to the (physically corrected) p0_max to
%                   avoid the exact edge (default 0.9)
%   eps_max         threshold of epsilon = max(|v|)/V0 for physical validity
%                   (default 0.5)
%   p0_min          first p0 of the sweep [Pa] (default 500)
%   p0_search_hi    starting p0 "expected to diverge" for the bisection
%                   (default 200e3 Pa; raise it if find_p0_max warns that it
%                   does not diverge)
%   tol             resolution of the p0_max bisection [Pa] (default 200)
%   force           true to recompute also the G already present in the
%                   output file (default false)
%   table_file, out_file   file names (to change the defaults)

ip = inputParser;
ip.addParameter('n_p0', 25);
ip.addParameter('dense_frac', 0.4);
ip.addParameter('dense_zone_frac', 0.3);
ip.addParameter('safety_margin', 0.9);
ip.addParameter('eps_max', 0.5);
ip.addParameter('p0_min', 500);
ip.addParameter('p0_search_hi', 200e3);
ip.addParameter('tol', 200);
ip.addParameter('force', false);
ip.addParameter('table_file', '../data/c_fbl1_table.csv');
ip.addParameter('out_file', '../data/dataset_1d_p2p3.csv');
ip.parse(varargin{:});
o = ip.Results;

[G_table, c_table] = read_two_col_csv(o.table_file);

out_rows = zeros(0, 5);
G_done = [];
if exist(o.out_file, 'file') && ~o.force
    out_rows = read_dataset_csv(o.out_file);
    if ~isempty(out_rows)
        G_done = unique(out_rows(:,1));
    end
end

for k = 1:numel(G_table)
    G_kPa = G_table(k);
    c_fbl1 = c_table(k);

    if isnan(c_fbl1)
        fprintf('G=%g kPa: c_fbl1 pending in %s, skipped.\n', G_kPa, o.table_file);
        continue
    end
    if ~o.force && any(G_done == G_kPa)
        fprintf('G=%g kPa: already in %s, skipped (use ''force'', true to redo it).\n', G_kPa, o.out_file);
        continue
    end

    G = G_kPa * 1e3;  % kPa -> Pa
    fprintf('G=%g kPa (c_fbl1=%.4g m/s): searching p0_max...\n', G_kPa, c_fbl1);

    p0_max_numerical = find_p0_max(G, c_fbl1, o.p0_min, o.p0_search_hi, o.tol);
    [eps_at_numerical, ~] = diag_vV0(G, c_fbl1, p0_max_numerical);

    if eps_at_numerical <= o.eps_max
        % physics does not limit within what the scheme allows to explore:
        % the numerical limit rules (typical case: G = 0, water)
        p0_max_raw = p0_max_numerical;
        fprintf('  p0_max_numerical = %.0f Pa (epsilon there = %.3f, does not exceed %.2f) -> limited by the NUMERICAL SCHEME\n', ...
            p0_max_numerical, eps_at_numerical, o.eps_max);
    else
        p0_max_physical = find_p0_valid(G, c_fbl1, o.p0_min, p0_max_numerical, o.tol, o.eps_max);
        p0_max_raw = p0_max_physical;
        fprintf('  p0_max_numerical = %.0f Pa (epsilon there = %.3f, exceeds %.2f) -> limited by PHYSICS: p0_max_physical = %.0f Pa\n', ...
            p0_max_numerical, eps_at_numerical, o.eps_max, p0_max_physical);
    end

    p0_max = o.safety_margin * p0_max_raw;
    fprintf('  p0_max used in the sweep, with safety margin (%.0f%%): %.0f Pa\n', ...
        o.safety_margin*100, p0_max);

    % Sampling in TWO segments, so that more points fall close to p0_max.
    % Why: close to p0_max the bubbles oscillate with large amplitude, the medium behaves in a much more nonlinear way
    % and p2, p3 change quickly with p0. A uniform grid would describe that
    % region with the same spacing as the easy one, so extra points are placed there.
    %
    % How: the range of p0 is cut at p0_split into
    %   - lower segment: [p0_min, p0_split], n_bottom equally spaced points
    %   - dense segment: [p0_split, p0_max],  n_top equally spaced points
    % with p0_split = p0_max - dense_zone_frac*(p0_max - p0_min), i.e. the
    % dense segment is the upper dense_zone_frac of the range. The point
    % p0_split belongs to both segments, so n_bottom + n_top - 1 = n_p0.
    %
    n_top = max(2, round(o.dense_frac * o.n_p0));
    n_bottom = o.n_p0 - n_top + 1;
    if n_bottom < 2
        n_bottom = 2;
        n_top = o.n_p0 - n_bottom + 1;
    end
    p0_split = p0_max - o.dense_zone_frac * (p0_max - o.p0_min);
    p0_bottom = linspace(o.p0_min, p0_split, n_bottom);
    p0_top = linspace(p0_split, p0_max, n_top);
    p0_array = unique([p0_bottom, p0_top]);

    for i = 1:numel(p0_array)
        [p2, p3, diverged] = simulate_1d_harmonics(G, c_fbl1, p0_array(i));
        if diverged
            fprintf('  WARNING: p0=%.0f Pa diverges despite the safety margin, omitted.\n', p0_array(i));
            continue
        end
        out_rows(end+1, :) = [G_kPa, c_fbl1, p0_array(i), p2, p3]; %#ok<AGROW>
        fprintf('  p0=%7.0f Pa -> p2=%.4g Pa, p3=%.4g Pa\n', p0_array(i), p2, p3);
    end

    write_dataset_csv(o.out_file, out_rows);
    fprintf('  saved in %s (%d rows in total)\n\n', o.out_file, size(out_rows,1));
end

fprintf('Done. %d rows in %s.\n', size(out_rows,1), o.out_file);
end

% ------------------------------------------------------------------
function [G_col, c_col] = read_two_col_csv(filename)
fid = fopen(filename, 'r');
if fid < 0
    error('generate_dataset_1d:no_table', 'Cannot find %s', filename);
end
fgetl(fid);  % header
G_col = [];
c_col = [];
while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line) || isempty(strtrim(line))
        continue
    end
    parts = strsplit(line, ',');
    G_col(end+1) = str2double(parts{1}); %#ok<AGROW>
    if numel(parts) < 2 || isempty(strtrim(parts{2}))
        c_col(end+1) = NaN; %#ok<AGROW>
    else
        c_col(end+1) = str2double(parts{2}); %#ok<AGROW>
    end
end
fclose(fid);
end

% ------------------------------------------------------------------
function rows = read_dataset_csv(filename)
fid = fopen(filename, 'r');
rows = zeros(0, 5);
if fid < 0
    return
end
fgetl(fid);  % header
while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line) || isempty(strtrim(line))
        continue
    end
    parts = strsplit(line, ',');
    if numel(parts) < 5
        continue
    end
    rows(end+1, :) = cellfun(@str2double, parts(1:5)); %#ok<AGROW>
end
fclose(fid);
end

% ------------------------------------------------------------------
function write_dataset_csv(filename, rows)
fid = fopen(filename, 'w');
fprintf(fid, 'G_kPa,c_fbl1,p0_Pa,p2_Pa,p3_Pa\n');
for r = 1:size(rows,1)
    fprintf(fid, '%.10g,%.10g,%.10g,%.10g,%.10g\n', rows(r,:));
end
fclose(fid);
end
