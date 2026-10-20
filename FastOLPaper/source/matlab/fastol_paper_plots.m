function fastol_paper_plots(only)
% fastol_paper_plots() redraws every line-plot figure of the FastOL paper from saved data.
% fastol_paper_plots('gpu' | 'summary' | 'spatial' | 'cost' | 'kolmogorov' | 'fields') redraws one group.
%
% Reporting only: no simulation, fitting or learning is run. Inputs (all in the repository):
%   source/gpu_data/final_validated_models.csv               GPU operator study, 160 fitted models
%   evidence/spatial/report_final/spatial_report.json         scattered-node convergence (field 'convergence')
%   evidence/kolmogorov/report_r1/report.json                 Kolmogorov refinement study
%   E:/fluid_datasets/<case>/data*.mat (read only)            benchmark fields ('fields' group): row 1 of
%       each data set is cached by fastol_extract_fields in matlab/cache/fastol_fields_first_sample.mat
% Output: source/figures/fastol_<what>.pdf (export_fig; vector for line plots, -opengl -r600 for fields).
% Every line-plot panel has the same axes box and 9/10 pt LaTeX text (fastol_plot_style), so each PDF is placed at its natural size.
if nargin < 1, only = 'all'; end
here = fileparts(mfilename('fullpath'));
src = fileparts(here);                                  % FastOLPaper/source
evd = fullfile(fileparts(src), 'evidence');             % FastOLPaper/evidence
out = fullfile(src, 'figures');
if ~exist(out, 'dir'), mkdir(out); end
efd = getenv('EXPORT_FIG_DIR');                         % export_fig (github.com/altmany/export_fig)
if isempty(efd), efd = 'E:\kernelpack-tools\export_fig'; end
if exist(efd, 'dir'), addpath(efd); end
gs = getenv('FASTOL_GHOSTSCRIPT_DIR');                  % optional; export_fig finds a system Ghostscript otherwise
if isempty(gs), gs = 'E:\texlive\2025\tlpkg\tlgs'; end  % Ghostscript shipped with TeX Live
if exist(gs, 'dir')
    setenv('GS_LIB', [gs '\Resource\Init;' gs '\lib;' gs '\Resource\Font;' gs '\kanji']);
    user_string('ghostscript', [gs '\bin\gswin64c.exe']);
end
set(groot, 'defaultTextInterpreter', 'latex', 'defaultLegendInterpreter', 'latex', ...
    'defaultAxesTickLabelInterpreter', 'latex');

if any(strcmp(only, {'all', 'gpu'}))
    gpu_figures(gpu_csv(src), out);
end
if any(strcmp(only, {'all', 'summary'}))
    gpu_summary(gpu_csv(src), out);
end
if any(strcmp(only, {'all', 'spatial'}))
    spatial_convergence(fullfile(evd, 'spatial', 'report_final', 'spatial_report.json'), out);
end
if any(strcmp(only, {'all', 'cost'}))
    cost_scaling(fullfile(evd, 'cost', 'cost_scaling.json'), out);
end
if any(strcmp(only, {'all', 'kolmogorov'}))
    kolmogorov(fullfile(evd, 'kolmogorov', 'report_r1', 'report.json'), out);
end
if any(strcmp(only, {'all', 'fields'}))
    cache = fullfile(here, 'cache');
    if ~exist(cache, 'dir'), mkdir(cache); end
    F = fastol_extract_fields(fullfile(cache, 'fastol_fields_first_sample.mat'));
    fastol_field_figures(F, out);
end
end

% ---------------------------------------------------------------- GPU operator study
function file = gpu_csv(src)
% Campaign table with the cost columns of the fill-reducing replay, when build_gpu_results.py made it.
file = fullfile(src, 'gpu_data', 'final_validated_models_reordered.csv');
if ~exist(file, 'file'), file = fullfile(src, 'gpu_data', 'final_validated_models.csv'); end
end

function gpu_figures(file, out)
S = fastol_plot_style();
T = readtable(file, 'TextType', 'string', 'Delimiter', ',', 'VariableNamingRule', 'preserve');
assert(height(T) == 160, 'expected 160 fitted models');
N = [500 1000 5000 10000];
cases = {'forced_turb', 'Forced isotropic turbulence'; 'species_transport', 'Turbulent species transport';
    'merge_vortices_easier', 'Merging vortices'; 'lid_cavity_flow', 'Lid-driven cavity';
    'flow_cylinder_laminar', 'Cylinder flow (steady)'; 'flow_cylinder_shedding', 'Cylinder flow (vortex shedding)';
    'taylor_green', 'Taylor--Green vortices (space)'; 'taylor_green_time', 'Taylor--Green vortices (spacetime)';
    'backward_facing_step', 'Backward-facing step'; 'buoyancy_cavity_flow', 'Buoyancy-driven cavity'};
metrics = {   % name, column, multiplier, y label, log scale
    'error',   'test_error',                                      1,    'relative test error',       true;
    'density', 'kernel_density',                                  100,  'matrix density (\%)',        false;
    'warm',    'warm_100query_median_seconds',                    1000, 'time for 100 fields (ms)',   true;
    'storage', 'accounted_resident_storage_ESTIMATE_decimal_GB',  1,    'storage (GB)',               true};
groups = {'main', 1:4, 2; 'additional', 5:10, 3};
for m = 1:size(metrics, 1)
    for g = 1:size(groups, 1)
        idx = groups{g, 2}; nc = groups{g, 3}; nr = ceil(numel(idx)/nc);
        lcol = 2*(nc == 2) + 4*(nc ~= 2);               % two-column grids are too narrow for one legend row
        [fig, ax] = grid_figure(nr, nc, 4/lcol);
        h = gobjects(1, 4);
        for k = 1:numel(idx)
            a = ax(k); c = cases{idx(k), 1};
            ys = [];
            for j = 1:4
                y = zeros(size(N));
                for n = 1:numel(N)
                    row = T(T.('case') == c & T.N == N(n) & T.model == S.models{j}, :);
                    assert(height(row) == 1, 'missing %s N=%d %s', c, N(n), S.models{j});
                    y(n) = row.(metrics{m, 2}) * metrics{m, 3};
                end
                ys = [ys y]; %#ok<AGROW>
                h(j) = model_line(a, N, y, j);
            end
            set(a, 'XScale', 'log', 'XLim', [420 11900], 'XTick', N, ...
                'XTickLabel', {'$500$', '$1000$', '$5000$', '$10^4$'});
            if metrics{m, 5}
                log_axis(a, ys);
            else
                set(a, 'YLim', [-3 103], 'YTick', 0:25:100);
            end
            title(a, cases{idx(k), 2});
            if mod(k-1, nc) == 0, ylabel(a, metrics{m, 4}); end
            if k > numel(idx) - nc, xlabel(a, 'training functions $N$'); end
            finish_axes(a);
        end
        top_legend(fig, ax, h, strcat(S.model_name, '\quad'), lcol);   % pad: MATLAB under-measures LaTeX entries
        export(fig, fullfile(out, sprintf('fastol_gpu_%s_%s', metrics{m, 1}, groups{g, 1})), 'tight');
    end
end
end

function h = model_line(a, x, y, j)
S = fastol_plot_style();
col = S.model_color{j};
if S.model_filled(j), face = col; else, face = 'w'; end
h = plot(a, x, y, S.model_line{j}, 'Color', col, 'LineWidth', S.lw, 'Marker', S.model_marker{j}, ...
    'MarkerSize', S.ms, 'MarkerFaceColor', face, 'MarkerEdgeColor', col);
end

% ---------------------------------------------------------------- GPU summary at N = 10000
function gpu_summary(file, out)
% Horizontal dot plot over all ten benchmarks at N = 10000.
% Left: error of the fixed-shape kernel without augmentation divided by the error of each variant.
% Right: selected kernel density d_K = nnz(K_UU)/N^2 of all four models.
S = fastol_plot_style();
T = readtable(file, 'TextType', 'string', 'Delimiter', ',', 'VariableNamingRule', 'preserve');
assert(height(T) == 160, 'expected 160 fitted models');
cases = {'forced_turb', 'Forced isotropic turbulence'; 'species_transport', 'Turbulent species transport';
    'merge_vortices_easier', 'Merging vortices';
    'lid_cavity_flow', 'Lid-driven cavity'; 'flow_cylinder_laminar', 'Cylinder flow (steady)';
    'flow_cylinder_shedding', 'Cylinder flow (vortex shedding)'; 'backward_facing_step', 'Backward-facing step';
    'buoyancy_cavity_flow', 'Buoyancy-driven cavity';
    'taylor_green', 'Taylor--Green vortices (space)'; 'taylor_green_time', 'Taylor--Green vortices (spacetime)'};
% rows top to bottom, with a thin gap between the three groups of benchmarks
yrow = [1 2 3, 4.5 5.5 6.5 7.5 8.5 9.5, 11 12];
yrow = yrow(1:10);
nc = size(cases, 1);
err = zeros(nc, 4); dens = zeros(nc, 4);
for i = 1:nc
    for j = 1:4
        row = T(T.('case') == cases{i, 1} & T.N == 10000 & T.model == S.models{j}, :);
        assert(height(row) == 1, 'missing %s %s', cases{i, 1}, S.models{j});
        err(i, j) = row.test_error; dens(i, j) = row.kernel_density;
    end
end
ratio = err(:, 1) ./ err(:, 2:4);

left = 4.6; gap = 0.6; axh = 6.6; bottom = 1.15; top = 0.15;   % room for the longest benchmark name
W = left + 2*S.axw + gap + 0.4;
fig = figure('Units', 'centimeters', 'Position', [2 2 W bottom + axh + top], 'Color', 'w', 'Visible', 'off');
ax(1) = axes(fig, 'Units', 'centimeters', 'Position', [left bottom S.axw axh]);
ax(2) = axes(fig, 'Units', 'centimeters', 'Position', [left + S.axw + gap bottom S.axw axh]);
hold(ax(1), 'on'); hold(ax(2), 'on');

% left panel: error reduction of the three variants
a = ax(1);
plot(a, [1 1], [0 13], '-', 'Color', S.grey, 'LineWidth', S.guidelw);
off = [-0.22 0 0.22];
for j = 2:4
    model_dots(a, ratio(:, j-1), yrow + off(j-1), j);
end
set(a, 'XScale', 'log', 'XLim', [0.5 1e5], 'XTick', 10.^(0:5), ...
    'XTickLabel', {'$1$', '', '$10^2$', '', '$10^4$', ''});
xlabel(a, 'test-error reduction over fixed shape');

% right panel: selected density of all four models
a = ax(2);
off = [-0.27 -0.09 0.09 0.27];
h = gobjects(1, 4);
for j = 1:4
    h(j) = model_dots(a, dens(:, j), yrow + off(j), j);
end
set(a, 'XLim', [0 1], 'XTick', 0:0.25:1, 'XTickLabel', {'$0$', '', '$0.5$', '', '$1$'});
xlabel(a, 'selected density $d_K$');

for k = 1:2
    set(ax(k), 'YDir', 'reverse', 'YLim', [0.4 yrow(end) + 0.6], 'YTick', yrow);
    finish_axes(ax(k));
    set(ax(k), 'XTickLabelRotation', 0, 'Layer', 'bottom');
    ax(k).YGrid = 'off';
end
set(ax(1), 'YTickLabel', cases(:, 2)');
set(ax(2), 'YTickLabel', {});
% thin row guides across both panels
for k = 1:2
    xl = ax(k).XLim;
    for i = 1:nc
        plot(ax(k), xl, [yrow(i) yrow(i)], '-', 'Color', [0.92 0.92 0.92], 'LineWidth', 0.5);
    end
    uistack(findobj(ax(k), 'Type', 'line', 'Color', [0.92 0.92 0.92]), 'bottom');
end

lg = legend(ax(1), h, S.model_name, 'FontSize', S.font, 'NumColumns', 4, 'Orientation', 'horizontal', ...
    'Box', 'off', 'AutoUpdate', 'off');
lg.ItemTokenSize = [12 8];
lg.Units = 'centimeters'; drawnow;
p1 = ax(1).Position; p2 = ax(2).Position;
ytop = p1(2) + p1(4) + 0.15;
lg.Position(1:2) = [(p1(1) + p2(1) + p2(3))/2 - lg.Position(3)/2, ytop];
fig.Position(4) = ytop + lg.Position(4) + 0.1;
fig.Position(3) = max(fig.Position(3), lg.Position(1) + lg.Position(3) + 0.3);
export(fig, fullfile(out, 'fastol_gpu_summary'), 'tight');
end

function h = model_dots(a, x, y, j)
S = fastol_plot_style();
col = S.model_color{j};
if S.model_filled(j), face = col; else, face = 'w'; end
h = plot(a, x, y, 'LineStyle', 'none', 'Color', col, 'LineWidth', S.lw, 'Marker', S.model_marker{j}, ...
    'MarkerSize', S.ms, 'MarkerFaceColor', face, 'MarkerEdgeColor', col);
end

% ---------------------------------------------------------------- local vs global reconstruction cost
function cost_scaling(file, out)
% Setup time and memory of the local reconstruction (degree 4, cubic PHS) and of a global
% divergence-free Matern reconstruction (dense dm x dm Gram + Cholesky) against the number m of
% nodes, from evidence/cost/cost_scaling.json (cost only, warm runs). Dotted: global cost
% extrapolated as (dm)^3 time and (dm)^2 memory from its largest measured size.
S = fastol_plot_style();
R = jsondecode(fileread(file));
col = {[0 0.35 0.75], [0.85 0.40 0]}; mk = {'o', 's'};
[fig, ax] = grid_figure(1, 2, 1, S.gapx_label);
h = gobjects(1, 4); xs = []; ts = []; bs = [];
for k = 1:2
    d = k + 1; c = col{k};
    L = R.local([R.local.d] == d); G = R.dense_global([R.dense_global.d] == d);
    [~, o] = sort([L.m]); L = L(o); [~, o] = sort([G.m]); G = G(o);
    mL = [L.m]; mG = [G.m];
    tG = [G.assembly_s] + [G.cholesky_s]; bG = [G.matrix_bytes];
    mx = logspace(log10(mG(end)), log10(mL(end)), 20);
    tx = tG(end) * (mx / mG(end)).^3; bx = bG(end) * (mx / mG(end)).^2;
    h(k) = plot(ax(1), mL, [L.warm_s], '-', 'Color', c, 'LineWidth', S.lw, 'Marker', mk{k}, ...
        'MarkerSize', S.ms, 'MarkerFaceColor', c, 'MarkerEdgeColor', c);
    h(k + 2) = plot(ax(1), mG, tG, '--', 'Color', c, 'LineWidth', S.lw, 'Marker', mk{k}, ...
        'MarkerSize', S.ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', c);
    plot(ax(1), mx, tx, ':', 'Color', c, 'LineWidth', S.guidelw);
    plot(ax(2), mL, [L.bytes] / 1e9, '-', 'Color', c, 'LineWidth', S.lw, 'Marker', mk{k}, ...
        'MarkerSize', S.ms, 'MarkerFaceColor', c, 'MarkerEdgeColor', c);
    plot(ax(2), mG, bG / 1e9, '--', 'Color', c, 'LineWidth', S.lw, 'Marker', mk{k}, ...
        'MarkerSize', S.ms, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', c);
    plot(ax(2), mx, bx / 1e9, ':', 'Color', c, 'LineWidth', S.guidelw);
    xs = [xs mL mG]; ts = [ts [L.warm_s] tG tx]; bs = [bs [L.bytes] bG bx]; %#ok<AGROW>
end
for a = ax
    set(a, 'XScale', 'log', 'XLim', [min(xs) / 1.3, max(xs) * 1.3], 'XTick', [1e3 1e4 1e5], ...
        'XTickLabel', {'$10^3$', '$10^4$', '$10^5$'});
    xlabel(a, 'output points $m$');
end
log_axis(ax(1), ts); log_axis(ax(2), bs / 1e9);
ylabel(ax(1), 'setup time (s)'); ylabel(ax(2), 'memory (GB)');
title(ax(1), 'Setup time'); title(ax(2), 'Memory');
finish_axes(ax(1)); finish_axes(ax(2));
top_legend(fig, ax, h, {'local, 2D', 'local, 3D', 'global, 2D', 'global, 3D'}, 4);
export(fig, fullfile(out, 'fastol_cost_scaling'), 'tight');
end

% ---------------------------------------------------------------- scattered-node convergence
function spatial_convergence(file, out)
S = fastol_plot_style();
R = jsondecode(fileread(file));
C = R.convergence; if iscell(C), C = [C{:}]; end
[fig, ax] = grid_figure(1, 2, 2);
xt = {[0.03 0.05 0.07 0.1], [0.1 0.15 0.2 0.25]};
for d = [2 3]
    a = ax(d-1); ys = []; xs = [];
    for p = [2 4 6]
        col = S.order_color(p); mk = S.order_marker(p);
        for q = 1:2
            c = C([C.dimension] == d & [C.order] == p & strcmp({C.kernel}, S.kernel{q}));
            assert(numel(c) == 1 && c.checked && c.four_levels_available);
            if S.kernel_filled(q), face = col; else, face = 'w'; end
            plot(a, c.h, c.error, S.kernel_line{q}, 'Color', col, 'LineWidth', S.lw, 'Marker', mk, ...
                'MarkerSize', S.ms, 'MarkerFaceColor', face, 'MarkerEdgeColor', col);
            ys = [ys; c.error(:)]; xs = [xs; c.h(:)]; %#ok<AGROW>
            if q == 1   % slope-p reference through the finest PHS point (saved in the report)
                plot(a, c.h, c.reference, ':', 'Color', col, 'LineWidth', S.guidelw);
                ys = [ys; c.reference(:)]; %#ok<AGROW>
            end
        end
    end
    set(a, 'XScale', 'log', 'XLim', [min(xs)/1.08 max(xs)*1.08], 'XTick', xt{d-1});
    log_axis(a, ys);
    title(a, sprintf('Irregular %dD domain', d));
    xlabel(a, 'sampled fill distance $h_Q$');
    if d == 2, ylabel(a, 'relative RMS error $E_{\rm sp}$'); end
    finish_axes(a);
end
h = order_handles(ax(1));
h(end+1) = plot(ax(1), nan, nan, 'k-', 'LineWidth', S.lw);
h(end+1) = plot(ax(1), nan, nan, 'k--', 'LineWidth', S.lw);
h(end+1) = plot(ax(1), nan, nan, ':', 'Color', S.grey, 'LineWidth', S.guidelw);
top_legend(fig, ax, h, {'order 2', 'order 4', 'order 6', 'PHS', 'Mat\''ern', '$h_Q^{p}$ guide'}, 3);
export(fig, fullfile(out, 'fastol_spatial_convergence'));
end

% ---------------------------------------------------------------- Kolmogorov refinement
function kolmogorov(file, out)
S = fastol_plot_style();
R = jsondecode(fileread(file));
L = R.local; if iscell(L), L = [L{:}]; end
mask = logical(R.reference.certified_interval);
r = R.reference.radius(mask);
[fig, ax] = grid_figure(1, 2, 3, S.gapx_label);

% (a) structure functions, degree five (order 6) at three of the six clouds
a = ax(1); col6 = S.order_color(6);
h = gobjects(0); names = {};
h(end+1) = plot(a, r, R.reference.pair_mean_s2(mask), 'k-', 'LineWidth', 1.4); names{end+1} = 'reference';
hs = [0.26 0.101 0.05]; shade = [0.40 0.70 1.0];
ys = R.reference.pair_mean_s2(mask);
for k = 1:3
    row = L([L.order] == 6 & abs([L.h] - hs(k)) < 1e-12); assert(numel(row) == 1);
    col = 1 - shade(k)*(1 - col6);              % light to dark green
    s2 = row.mean_s2(mask);
    h(end+1) = plot(a, r, s2, '-', 'Color', col, 'LineWidth', S.lw, 'Marker', '^', ...
        'MarkerSize', S.ms, 'MarkerFaceColor', col, 'MarkerEdgeColor', col); %#ok<AGROW>
    names{end+1} = sprintf('$h=%g$', hs(k)); %#ok<AGROW>
    ys = [ys; s2(:)]; %#ok<AGROW>
end
g = R.reference.two_thirds_guide(mask);
h(end+1) = plot(a, r, g, ':', 'Color', S.grey, 'LineWidth', S.guidelw); names{end+1} = '$r^{2/3}$ guide';
ys = [ys; g(:)];
set(a, 'XScale', 'log', 'XLim', [min(r)/1.15 max(r)*1.15], 'XTick', [0.03 0.1 0.3 0.8]);
log_axis(a, ys);
xlabel(a, 'separation $r$'); ylabel(a, '$S_2(r)$');
finish_axes(a);
panel_legend(fig, a, h, names, 2);

% (b) worst-direction relative S2 error at r = 0.8 against prescribed spacing h
a = ax(2); ys = [];
h = order_handles(a);
for p = [2 4 6]
    rows = L([L.order] == p); assert(numel(rows) == 6);
    [hh, o] = sort([rows.h]); e = [rows.worst_relative_s2_error_r08]; e = e(o);
    col = S.order_color(p);
    plot(a, hh, e, '-', 'Color', col, 'LineWidth', S.lw, 'Marker', S.order_marker(p), ...
        'MarkerSize', S.ms, 'MarkerFaceColor', col, 'MarkerEdgeColor', col);
    ys = [ys e]; %#ok<AGROW>
end
h(end+1) = plot(a, R.guides.h, R.guides.two_thirds_error, ':', 'Color', S.grey, 'LineWidth', S.guidelw);
ys = [ys R.guides.two_thirds_error(:)'];
set(a, 'XScale', 'log', 'XLim', [0.045 0.29], 'XTick', [0.05 0.1 0.2], ...
    'XTickLabel', {'$0.05$', '$0.1$', '$0.2$'});
log_axis(a, ys, [0.03 0.3]);
xlabel(a, 'node spacing $h$'); ylabel(a, '$E_{S_2}(0.8)$');
finish_axes(a);
panel_legend(fig, a, h, {'order 2', 'order 4', 'order 6', '$h^{2/3}$ guide'}, 2);
export(fig, fullfile(out, 'fastol_kolmogorov_refinement'));
end

% ---------------------------------------------------------------- shared helpers
function [fig, ax] = grid_figure(nr, nc, legend_rows, gapx)
% Axes of identical size S.axw x S.axh (cm) on an nr x nc grid, titles above each panel,
% a legend strip of legend_rows rows on top.
S = fastol_plot_style();
if nargin < 4, gapx = S.gapx; end
W = S.left + nc*S.axw + (nc-1)*gapx + 0.4;
H = S.bottom + nr*S.axh + (nr-1)*S.gapy + S.title + legend_rows*0.42 + 0.35;
fig = figure('Units', 'centimeters', 'Position', [2 2 W H], 'Color', 'w', 'Visible', 'off');
ax = gobjects(1, nr*nc);
for r = 1:nr
    for c = 1:nc
        k = (r-1)*nc + c;
        ax(k) = axes(fig, 'Units', 'centimeters', 'Position', ...
            [S.left + (c-1)*(S.axw + gapx), S.bottom + (nr-r)*(S.axh + S.gapy), S.axw, S.axh]);
        hold(ax(k), 'on');
    end
end
end

function finish_axes(a)
S = fastol_plot_style();
set(a, 'FontSize', S.font, 'LineWidth', S.axlw, 'Box', 'on', 'TickLabelInterpreter', 'latex', ...
    'XMinorTick', 'off', 'YMinorTick', 'off', 'TickLength', [0.02 0.02], 'Layer', 'top');
grid(a, 'on'); a.XMinorGrid = 'off'; a.YMinorGrid = 'off'; a.GridAlpha = 0.12;
a.XLabel.FontSize = S.label; a.YLabel.FontSize = S.label;
a.Title.FontSize = S.label; a.Title.FontWeight = 'normal';
end

function log_axis(a, ys, lims)
% Limits pad the data by a factor 1.6; ticks at decades, or at 1-2-5 if fewer than two decades fit.
% Labels are decimals when every tick lies in [0.01, 1000], powers of ten otherwise.
ys = ys(isfinite(ys) & ys > 0);
if nargin < 3, lims = [min(ys)/1.6, max(ys)*1.6]; end
set(a, 'YScale', 'log', 'YLim', lims);
e = ceil(log10(lims(1))):floor(log10(lims(2)));
if numel(e) >= 2
    t = 10.^e;
else
    t = [];
    for k = floor(log10(lims(1))):ceil(log10(lims(2)))
        v = [1 2 5]*10^k;
        t = [t v(v >= lims(1) & v <= lims(2))]; %#ok<AGROW>
    end
end
k = floor(log10(t) + 1e-9); m = round(t./10.^k);
if all(k >= -2 & k <= 3)
    lab = arrayfun(@(v) sprintf('$%g$', v), t, 'UniformOutput', false);
else
    lab = arrayfun(@(mm, kk) sprintf('$10^{%d}$', kk), m, k, 'UniformOutput', false);
    lab(m ~= 1) = arrayfun(@(mm, kk) sprintf('$%d{\\cdot}10^{%d}$', mm, kk), m(m ~= 1), k(m ~= 1), ...
        'UniformOutput', false);
end
if numel(t) > 6, lab(2:2:end) = {''}; end
set(a, 'YTick', t, 'YTickLabel', lab);
end

function h = order_handles(a)
S = fastol_plot_style();
h = gobjects(0);
for p = [2 4 6]
    col = S.order_color(p);
    h(end+1) = plot(a, nan, nan, '-', 'Color', col, 'LineWidth', S.lw, 'Marker', S.order_marker(p), ...
        'MarkerSize', S.ms, 'MarkerFaceColor', col, 'MarkerEdgeColor', col); %#ok<AGROW>
end
end

function top_legend(fig, ax, h, names, ncol)
% one horizontal legend centred above the whole panel grid
S = fastol_plot_style();
lg = legend(ax(1), h, names, 'FontSize', S.font, 'NumColumns', ncol, 'Orientation', 'horizontal', 'Box', 'off', 'AutoUpdate', 'off');
lg.ItemTokenSize = [20 8];
lg.Units = 'centimeters'; drawnow;
p1 = ax(1).Position; p2 = ax(end).Position;
xc = (p1(1) + p2(1) + p2(3))/2;
ytop = p1(2) + p1(4) + S.title;
lg.Position(1:2) = [xc - lg.Position(3)/2, ytop];
fig.Position(4) = max(fig.Position(4), ytop + lg.Position(4) + 0.1);
end

function panel_legend(fig, a, h, names, ncol)
% legend centred above one panel, above its title strip
S = fastol_plot_style();
lg = legend(a, h, names, 'FontSize', S.font, 'NumColumns', ncol, 'Orientation', 'horizontal', 'Box', 'off', 'AutoUpdate', 'off');
lg.ItemTokenSize = [20 8];
lg.Units = 'centimeters'; drawnow;
p = a.Position;
lg.Position(1:2) = [p(1) + p(3)/2 - lg.Position(3)/2, p(2) + p(4) + 0.15];
fig.Position(4) = max(fig.Position(4), p(2) + p(4) + 0.15 + lg.Position(4) + 0.1);
end

function export(fig, file, mode)
% mode 'tight': export the whole page, then crop to the vector bounding box with pdfcrop.
% export_fig's own crop is computed from a bitmap in which LaTeX text is narrower than in
% the PDF, so it can cut the last glyph of a wide legend entry or row label.
drawnow;
if nargin > 2 && strcmp(mode, 'tight')
    % pad the page by 1 cm on every side so no text reaches its edge, then crop
    kids = findall(fig, '-property', 'Units', '-depth', 1);
    kids = kids(isprop(kids, 'Position'));
    set(kids, 'Units', 'centimeters');
    for k = 1:numel(kids), kids(k).Position(1:2) = kids(k).Position(1:2) + 1; end
    fig.Position(3:4) = fig.Position(3:4) + 2;
    drawnow;
    export_fig(fig, file, '-pdf', '-painters', '-transparent', '-nocrop');
    % pdfcrop (TeX Live) fails on Windows for paths it must link, so crop a copy in tempdir
    d = tempname; mkdir(d);
    copyfile([file '.pdf'], fullfile(d, 'in.pdf'));
    crop = 'E:\texlive\2025\bin\windows\pdfcrop.exe';
    [st, msg] = system(sprintf('cd /d "%s" && "%s" --margins 1 in.pdf out.pdf', d, crop));
    assert(st == 0, 'pdfcrop failed: %s', msg);
    copyfile(fullfile(d, 'out.pdf'), [file '.pdf'], 'f');
    rmdir(d, 's');
else
    export_fig(fig, file, '-pdf', '-painters', '-transparent');
end
close(fig);
fprintf('wrote %s.pdf\n', file);
end
