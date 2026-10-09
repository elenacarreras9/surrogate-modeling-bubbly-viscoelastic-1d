function analyze_p0_limits(varargin)
% For every G in c_fbl1_table.csv (with known c_fbl),
% compute and compare the TWO possible limits of p0:
%
%   p0_max_numerical  where the numerical scheme diverges (find_p0_max.m)
%   p0_max_physical   where epsilon = max(|v|)/V0 crosses 0.5 (find_p0_valid.m;
%                     criterion of Carreras-Casanova & Vanhille, Acoustics
%                     2025, 7, 67, p. 21). If epsilon does not exceed 0.5 even
%                     at p0_max_numerical it is left blank: physics is not the
%                     limiting factor in that medium.
%
% and from them:
%   p0_max     = min(p0_max_numerical, p0_max_physical), the one that rules
%   limited_by = 'physical'   if p0_max_physical < p0_max_numerical
%                'numerical'  if epsilon does not reach 0.5 even at
%                             p0_max_numerical (physics would allow more,
%                             but the scheme breaks first)
%
% It only reads c_fbl1_table.csv and writes p0_limits_by_G.csv; it does not
% touch the dataset.
%
% Usage (run from the matlab/ folder):
%   analyze_p0_limits()
%   analyze_p0_limits('table_file', '../data/c_fbl1_table.csv', 'eps_max', 0.5)

% Archivo que crea el .csv para guardar los valores de p0_max. Meramente informativo.
% Son los mismos valores de p0 que utiliza generate_dataset_1d, 
% con una diferencia: generate_dataset_1d multiplica el límite por 0.9 (el safety_margin) y analyze_p0_limits no.

ip = inputParser;
ip.addParameter('table_file', '../data/c_fbl1_table.csv');
ip.addParameter('out_file', '../data/p0_limits_by_G.csv');
ip.addParameter('p0_min', 500);
ip.addParameter('p0_search_hi', 200e3);
ip.addParameter('tol', 200);
ip.addParameter('eps_max', 0.5);
ip.parse(varargin{:});
o = ip.Results;

[G_table, c_table] = read_two_col_csv(o.table_file);
n = numel(G_table);

p0_max_numerical = nan(n,1);
p0_max_physical  = nan(n,1);
p0_max           = nan(n,1);
eps_at_numerical = nan(n,1);
limited_by       = cell(n,1);

fprintf('%6s %14s %14s %12s %8s %12s\n', 'G_kPa', 'p0max_numer.', 'p0max_phys.', 'p0_max', 'eps_num', 'limited_by');
for k = 1:n
    G_kPa = G_table(k);
    c_fbl1 = c_table(k);
    if isnan(c_fbl1)
        fprintf('G=%g kPa: c_fbl1 pending in %s, skipped.\n', G_kPa, o.table_file);
        continue
    end
    G = G_kPa * 1e3;

    pnum = find_p0_max(G, c_fbl1, o.p0_min, o.p0_search_hi, o.tol);
    p0_max_numerical(k) = pnum;

    [mv_at_num, ~] = diag_vV0(G, c_fbl1, pnum);
    eps_at_numerical(k) = mv_at_num;

    if mv_at_num <= o.eps_max
        % epsilon does not reach 0.5 even at the numerical limit itself:
        % physics does NOT limit in this medium, within what the scheme
        % allows to explore. The real limit is the numerical one.
        p0_max_physical(k) = NaN;
        p0_max(k) = pnum;
        limited_by{k} = 'numerical';
    else
        pphys = find_p0_valid(G, c_fbl1, o.p0_min, pnum, o.tol, o.eps_max);
        p0_max_physical(k) = pphys;
        p0_max(k) = pphys;
        limited_by{k} = 'physical';
    end

    fprintf('%6g %14.0f %14s %12.0f %8.3f %12s\n', G_kPa, p0_max_numerical(k), ...
        fmt_or_dash(p0_max_physical(k)), p0_max(k), eps_at_numerical(k), limited_by{k});
end

% ---- CSV output ----
fid = fopen(o.out_file, 'w');
fprintf(fid, 'G_kPa,c_fbl1,p0_max_numerical,p0_max_physical,p0_max,epsilon_at_p0max_numerical,limited_by\n');
for k = 1:n
    if isnan(p0_max(k)), continue; end
    fprintf(fid, '%.10g,%.10g,%.10g,%s,%.10g,%.6g,%s\n', G_table(k), c_table(k), ...
        p0_max_numerical(k), fmt_or_empty(p0_max_physical(k)), p0_max(k), eps_at_numerical(k), limited_by{k});
end
fclose(fid);
fprintf('\nSaved in %s.\n', o.out_file);

% ---- summary: media where physics would allow more but the scheme breaks first ----
idx_numerical = find(strcmp(limited_by, 'numerical'));
fprintf('\nMedia limited by the NUMERICAL SCHEME (physics would allow a larger p0):\n');
if isempty(idx_numerical)
    fprintf('  (none -- in every computed medium physics (eps>%.2f) limits before the scheme does)\n', o.eps_max);
else
    for k = idx_numerical(:)'
        fprintf('  G=%g kPa: p0_max_numerical=%.0f Pa, epsilon there only reaches %.3f (< %.2f)\n', ...
            G_table(k), p0_max_numerical(k), eps_at_numerical(k), o.eps_max);
    end
end
end

% ------------------------------------------------------------------
function s = fmt_or_dash(x)
if isnan(x)
    s = '--';
else
    s = sprintf('%.0f', x);
end
end

% ------------------------------------------------------------------
function s = fmt_or_empty(x)
if isnan(x)
    s = '';
else
    s = sprintf('%.10g', x);
end
end

% ------------------------------------------------------------------
function [G_col, c_col] = read_two_col_csv(filename)
fid = fopen(filename, 'r');
if fid < 0
    error('analyze_p0_limits:no_table', 'Cannot find %s', filename);
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
