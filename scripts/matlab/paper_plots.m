function paper_plots(only)
% paper_plots() draws every figure; paper_plots('velocity') only the velocity panels and their colorbar.
% Convergence and timing plots of the paper from Outputs_and_results/paper_figures/plot_data.json
% (written by scripts/paper_figures.py). One PDF per panel and one legend PDF per figure, written with
% export_fig to KernelPackJaxPaper/Images/plots/; the paper arranges them in LaTeX. Panels are drawn at
% half the SIAM text width (6.4 cm) with 9 pt fonts and placed at 1:1, so they read at 100% zoom.
%   forward_<problem>_<geometry>.pdf, inverse_<problem>_<geometry>.pdf, legend_convergence.pdf,
%   timings_setup.pdf, timings_step.pdf, legend_timings.pdf,
%   inverse_timings_<problem>.pdf, legend_inverse_timings.pdf
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath('E:\kernelpack-tools\export_fig');
gs = 'E:\texlive\2025\tlpkg\tlgs';               % Ghostscript shipped with TeX Live
setenv('GS_LIB', [gs '\Resource\Init;' gs '\lib;' gs '\Resource\Font;' gs '\kanji']);
user_string('ghostscript', [gs '\bin\gswin64c.exe']);
D = jsondecode(fileread(fullfile(root, 'Outputs_and_results', 'paper_figures', 'plot_data.json')));
out = fullfile(root, 'KernelPackJaxPaper', 'Images', 'plots');
if ~exist(out, 'dir'), mkdir(out); end
if nargin > 0 && strcmp(only, 'velocity'), velocity(D.velocity, out); return; end
convergence(D.forward, 'forward', out);
convergence(D.inverse, 'inverse', out);
convergence_legend(out);
timings(D.timings, out);
inverse_timings(D.inverse_timings, out);
geometry(D.geometry, out);
node_clouds(D.node_clouds, out);
node_stack(D.node_stack, out);
convergence(D.obstacle_forward, 'obstacle', out);
convergence(D.obstacle_inverse, 'obstacle', out);
obstacle_slice(D.obstacle_slice, out);
velocity(D.velocity, out);
end

function s = style()
s.width = 6.4; s.height = 5.0; s.font = 9; s.lw = 0.9; s.fitlw = 1.6; s.ms = 5;
s.color = containers.Map({0, 2, 4, 6}, {[0 0 0], [0 0.35 0.75], [0.85 0.40 0], [0 0.55 0.45]});
s.marker = containers.Map({0, 2, 4, 6}, {'d', 'o', 's', '^'});
end

function key = slug(s)
key = regexprep(lower(regexprep(s, '[^A-Za-z]+', '_')), '^_|_$', '');
end

function [fig, ax] = panel()
S = style();
fig = figure('Units', 'centimeters', 'Position', [2 2 S.width S.height], 'Color', 'w');
ax = axes(fig); hold(ax, 'on');
end

function decades(ax, lims)
% a tick at every decade; with more than six decades, labels at every other decade
e = floor(log10(lims(1))):ceil(log10(lims(2)));
ax.YTick = 10.^e;
lab = arrayfun(@(k) sprintf('$10^{%d}$', k), e, 'UniformOutput', false);
if numel(e) > 7, lab(mod(e, 2) ~= 0) = {''}; end
ax.YTickLabel = lab;
end

function finish(ax, fig, file)
S = style();
set(ax, 'XScale', 'log', 'YScale', 'log', 'FontSize', S.font, 'LineWidth', 0.7, 'Box', 'on', ...
    'TickLabelInterpreter', 'latex', 'YMinorTick', 'off');
grid(ax, 'on'); ax.XMinorGrid = 'off'; ax.YMinorGrid = 'off';
export_fig(fig, file, '-pdf', '-painters', '-transparent');
close(fig);
end

function convergence(C, prefix, out)
S = style();
for r = 1:numel(C.rows)
    row = C.rows(r);
    for c = 1:numel(row.panels)
        P = row.panels(c);
        [fig, ax] = panel();
        ys = [];
        for k = 1:numel(P.curves)
            cv = P.curves(k); col = S.color(cv.p); ys = [ys; cv.y(:)]; %#ok<AGROW>
            plot(ax, cv.x, cv.y, ['-' S.marker(cv.p)], 'Color', col, 'LineWidth', S.lw, ...
                'MarkerSize', S.ms, 'MarkerFaceColor', col);
            if ~isempty(cv.fit)
                f = cv.fit; xa = f.x0/1.06; xb = f.x1*1.06;
                plot(ax, [xa xb], exp(f.c)*[xa xb].^f.k, '--', 'Color', col, 'LineWidth', S.fitlw);
                ys = [ys; exp(f.c)*[xa; xb].^f.k]; %#ok<AGROW>
                text(ax, xa/1.05, exp(f.c)*xa^f.k, sprintf('%.1f', f.k), 'Color', col*0.85, 'FontSize', S.font, ...
                    'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle', 'Interpreter', 'latex');
            end
        end
        lims = [10^floor(log10(min(ys)/3)) 10^ceil(log10(max(ys)*3))];
        set(ax, 'XLim', [0.019 0.125], 'XTick', [0.02 0.03 0.05 0.1], 'YLim', lims);
        set(ax, 'XScale', 'log', 'YScale', 'log'); decades(ax, lims);
        title(ax, row.title, 'FontSize', S.font+1, 'FontWeight', 'normal', 'Interpreter', 'latex');
        xlabel(ax, sprintf('$N^{-1/%d}$', P.d), 'Interpreter', 'latex', 'FontSize', S.font+1);
        ylabel(ax, C.ylabel, 'Interpreter', 'latex', 'FontSize', S.font+1);
        finish(ax, fig, fullfile(out, sprintf('%s_%s_%s', prefix, slug(row.title), slug(P.geometry))));
    end
end
end

function legend_only(handles_fn, names, file, ncol)
S = style();
fig = figure('Units', 'centimeters', 'Position', [2 2 2*S.width 1.2], 'Color', 'w');
ax = axes(fig); hold(ax, 'on'); axis(ax, 'off');
h = handles_fn(ax);
lg = legend(ax, h, names, 'Interpreter', 'latex', 'FontSize', S.font, 'Orientation', 'horizontal', ...
    'NumColumns', ncol, 'Box', 'off', 'Location', 'north');
lg.ItemTokenSize = [22 10];
export_fig(fig, file, '-pdf', '-painters', '-transparent');
close(fig);
end

function convergence_legend(out)
S = style();
    function h = mk(ax)
        h = gobjects(0);
        for p = [2 4 6]
            col = S.color(p);
            h(end+1) = plot(ax, nan, nan, ['-' S.marker(p)], 'Color', col, 'LineWidth', S.lw, ...
                'MarkerSize', S.ms, 'MarkerFaceColor', col); %#ok<AGROW>
        end
        h(end+1) = plot(ax, nan, nan, 'k--', 'LineWidth', S.fitlw);
    end
legend_only(@mk, {'$p=2$', '$p=4$', '$p=6$', 'fit over $h\leq0.08$'}, ...
    fullfile(out, 'legend_convergence'), 4);
end

function timings(T, out)
S = style();
parts = {T.setup, T.darcy, T.rd, T.step};
titles = {'Setup', 'Darcy: solve', 'Reaction--diffusion: step', 'Advection--diffusion: step'};
files = {'timings_setup', 'timings_darcy', 'timings_rd', 'timings_step'};
ylims = {[0.5 1000], [1e-3 1], [1e-3 0.1], [1e-3 0.1]};
for k = 1:4
    [fig, ax] = panel(); P = parts{k};
    for j = 1:numel(P.series)
        s = P.series(j); col = S.color(s.p);
        if strcmp(s.geometry, 'treasure_chest'), face = col; else, face = 'w'; end
        plot(ax, s.N, s.t, S.marker(s.p), 'Color', col, 'MarkerFaceColor', face, 'MarkerSize', S.ms+1, 'LineWidth', S.lw);
    end
    nmax = 0; tmax = 0;
    for j = 1:numel(P.fits)
        f = P.fits(j); col = S.color(f.p); xf = [f.N0/1.1 f.N1*1.1];
        plot(ax, xf, exp(f.c)*xf.^f.a, '--', 'Color', col, 'LineWidth', S.fitlw);
        nmax = max(nmax, f.N1); tmax = max(tmax, exp(f.c)*f.N1^f.a);
    end
    plot(ax, [100 nmax], tmax*[100 nmax]/nmax, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.6);
    set(ax, 'XLim', [80 5e4], 'YLim', ylims{k}, 'XTick', 10.^(2:4));
    set(ax, 'XScale', 'log', 'YScale', 'log'); decades(ax, ylims{k});
    title(ax, titles{k}, 'FontSize', S.font+1, 'FontWeight', 'normal', 'Interpreter', 'latex');
    xlabel(ax, 'number of nodes $N$', 'Interpreter', 'latex', 'FontSize', S.font+1);
    ylabel(ax, 'wall-clock time (s)', 'Interpreter', 'latex', 'FontSize', S.font+1);
    finish(ax, fig, fullfile(out, files{k}));
end
    function h = mk(ax)
        h = gobjects(0);
        for p = [0 2 4 6]
            col = S.color(p);
            h(end+1) = plot(ax, nan, nan, S.marker(p), 'Color', col, 'MarkerFaceColor', col, ...
                'MarkerSize', S.ms+1, 'LineWidth', S.lw); %#ok<AGROW>
        end
        h(end+1) = plot(ax, nan, nan, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', S.ms+1);
        h(end+1) = plot(ax, nan, nan, 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', S.ms+1);
        h(end+1) = plot(ax, nan, nan, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.6);
    end
legend_only(@mk, {'node generation', '$p=2$', '$p=4$', '$p=6$', 'treasure chest', 'H-channel', '$O(N)$'}, ...
    fullfile(out, 'legend_timings'), 4);
end

function inverse_timings(T, out)
% time of one inverse solve (median over the families) against N; one panel per problem
S = style();
probs = {'darcy', 'rd', 'transport'}; titles = {'Darcy', 'Reaction--diffusion', 'Transport'};
for k = 1:3
    [fig, ax] = panel(); P = T.(probs{k});
    for j = 1:numel(P.series)
        s = P.series(j); col = S.color(s.p);
        if strcmp(s.geometry, 'treasure_chest'), face = col; else, face = 'w'; end
        plot(ax, s.N, s.t, S.marker(s.p), 'Color', col, 'MarkerFaceColor', face, 'MarkerSize', S.ms+1, 'LineWidth', S.lw);
    end
    nmax = 0; tmax = 0;
    for j = 1:numel(P.fits)
        f = P.fits(j); col = S.color(f.p); xf = [f.N0/1.1 f.N1*1.1];
        plot(ax, xf, exp(f.c)*xf.^f.a, '--', 'Color', col, 'LineWidth', S.fitlw);
        nmax = max(nmax, f.N1); tmax = max(tmax, exp(f.c)*f.N1^f.a);
    end
    plot(ax, [100 nmax], tmax*[100 nmax]/nmax, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.6);
    lims = [0.1 1000];
    set(ax, 'XLim', [80 5e4], 'YLim', lims, 'XTick', 10.^(2:4));
    set(ax, 'XScale', 'log', 'YScale', 'log'); decades(ax, lims);
    title(ax, titles{k}, 'FontSize', S.font+1, 'FontWeight', 'normal', 'Interpreter', 'latex');
    xlabel(ax, 'number of nodes $N$', 'Interpreter', 'latex', 'FontSize', S.font+1);
    ylabel(ax, 'time per inverse solve (s)', 'Interpreter', 'latex', 'FontSize', S.font+1);
    finish(ax, fig, fullfile(out, ['inverse_timings_' probs{k}]));
end
    function h = mk(ax)
        h = gobjects(0);
        for p = [2 4 6]
            col = S.color(p);
            h(end+1) = plot(ax, nan, nan, S.marker(p), 'Color', col, 'MarkerFaceColor', col, ...
                'MarkerSize', S.ms+1, 'LineWidth', S.lw); %#ok<AGROW>
        end
        h(end+1) = plot(ax, nan, nan, 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', S.ms+1);
        h(end+1) = plot(ax, nan, nan, 'ko', 'MarkerFaceColor', 'w', 'MarkerSize', S.ms+1);
        h(end+1) = plot(ax, nan, nan, ':', 'Color', [0.45 0.45 0.45], 'LineWidth', 1.6);
    end
legend_only(@mk, {'$p=2$', '$p=4$', '$p=6$', 'treasure chest', 'H-channel', '$O(N)$'}, ...
    fullfile(out, 'legend_inverse_timings'), 6);
end

function geometry(G, out)
% geometry errors: parametric layer vs input sample spacing, implicit layer vs site spacing h
S = style();
gs = {'h_channel', 'treasure_chest'}; col = {S.color(2), S.color(4)}; mk = {'o', 's'};
specs = {{'delta', 'dist', 'fit_dist', 'Parametric layer: position', 'sample spacing $\delta$', 'max distance', [1e-11 1e-2], 'geometry_parametric_position'}, ...
         {'delta', 'angle', 'fit_angle', 'Parametric layer: normals', 'sample spacing $\delta$', 'max angle (degrees)', [1e-7 1], 'geometry_parametric_normals'}};
for k = 1:2
    sp = specs{k}; [fig, ax] = panel();
    for j = 1:2
        g = G.(gs{j}); x = g.(sp{1}); y = g.(sp{2}); f = g.(sp{3});
        plot(ax, x, y, ['-' mk{j}], 'Color', col{j}, 'MarkerFaceColor', col{j}, 'MarkerSize', S.ms+1, 'LineWidth', S.lw);
        xf = [f.x0/1.1 f.x1*1.1]; plot(ax, xf, exp(f.c)*xf.^f.a, '--', 'Color', col{j}, 'LineWidth', S.fitlw);
        text(ax, f.x1*1.15, exp(f.c)*f.x1^f.a, sprintf('%.1f', f.a), 'Color', col{j}, 'FontSize', S.font, 'Interpreter', 'latex');
    end
    set(ax, 'XLim', [3e-3 0.2], 'YLim', sp{7}); set(ax, 'XScale', 'log', 'YScale', 'log'); decades(ax, sp{7});
    title(ax, sp{4}, 'FontSize', S.font+1, 'FontWeight', 'normal', 'Interpreter', 'latex');
    xlabel(ax, sp{5}, 'Interpreter', 'latex', 'FontSize', S.font+1);
    ylabel(ax, sp{6}, 'Interpreter', 'latex', 'FontSize', S.font+1);
    finish(ax, fig, fullfile(out, sp{8}));
end
[fig, ax] = panel(); lims = [1e-5 0.1];
for j = 1:2
    g = G.(gs{j});
    plot(ax, g.h, g.far, ['-' mk{j}], 'Color', col{j}, 'MarkerFaceColor', col{j}, 'MarkerSize', S.ms+1, 'LineWidth', S.lw);
    plot(ax, g.h, g.all, [':' mk{j}], 'Color', col{j}, 'MarkerFaceColor', 'w', 'MarkerSize', S.ms+1, 'LineWidth', S.lw);
    for fn = {'fit_far', 'fit_all'}
        f = g.(fn{1}); xf = [f.x0/1.1 f.x1*1.1]; plot(ax, xf, exp(f.c)*xf.^f.a, '--', 'Color', col{j}, 'LineWidth', S.fitlw);
        text(ax, f.x1*1.12, exp(f.c)*f.x1^f.a, sprintf('%.1f', f.a), 'Color', col{j}, 'FontSize', S.font, 'Interpreter', 'latex');
    end
end
set(ax, 'XLim', [0.03 0.25], 'YLim', lims, 'XTick', [0.04 0.08 0.16]); set(ax, 'XScale', 'log', 'YScale', 'log'); decades(ax, lims);
title(ax, 'Implicit layer', 'FontSize', S.font+1, 'FontWeight', 'normal', 'Interpreter', 'latex');
xlabel(ax, 'site spacing $h$', 'Interpreter', 'latex', 'FontSize', S.font+1);
ylabel(ax, 'max $|\phi|/\|\nabla\phi\|$','Interpreter', 'latex', 'FontSize', S.font+1);
finish(ax, fig, fullfile(out, 'geometry_implicit'));
    function h = mkl(ax)
        h = gobjects(0);
        for jj = 1:2
            h(end+1) = plot(ax, nan, nan, ['-' mk{jj}], 'Color', col{jj}, 'MarkerFaceColor', col{jj}, 'MarkerSize', S.ms+1, 'LineWidth', S.lw); %#ok<AGROW>
        end
        h(end+1) = plot(ax, nan, nan, 'k:o', 'MarkerFaceColor', 'w', 'MarkerSize', S.ms+1, 'LineWidth', S.lw);
        h(end+1) = plot(ax, nan, nan, 'k--', 'LineWidth', S.fitlw);
    end
legend_only(@mkl, {'H-channel', 'treasure chest', 'all points', 'fit'}, fullfile(out, 'legend_geometry'), 4);
end

function node_clouds(C, out)
% node sets of the H-channel: interior (blue dots), boundary (black dots), ghost (red crosses)
S = style();
specs = {'hchannel', 'H-channel, $h=0.08$', 8.6, [0 1 2], [0 0.5 1], 'nodes_hchannel'; ...
         'slot', 'Slot, $h=0.04$', 4.6, [0.9 1.1 1.3], [0.4 0.6 0.8], 'nodes_slot'};
for k = 1:2
    c = C.(specs{k, 1}); box = c.box(:)';
    w = specs{k, 3}; hgt = w * (box(4) - box(3)) / (box(2) - box(1)) + 1.6;
    fig = figure('Units', 'centimeters', 'Position', [2 2 w+1.4 hgt], 'Color', 'w');
    ax = axes(fig); hold(ax, 'on');
    xi = c.int_nodes; xb = c.bdry_nodes; xg = c.ghost_nodes;
    ms = 4 + 2*(k == 2);
    plot(ax, xi(:,1), xi(:,2), '.', 'Color', [0 0.30 0.75], 'MarkerSize', ms);
    plot(ax, xb(:,1), xb(:,2), '.', 'Color', 'k', 'MarkerSize', ms + 2);
    plot(ax, xg(:,1), xg(:,2), 'x', 'Color', [0.80 0.10 0.10], 'MarkerSize', 2.5 + 1.5*(k == 2), 'LineWidth', 0.8);
    axis(ax, 'equal'); set(ax, 'XLim', box(1:2), 'YLim', box(3:4), 'XTick', specs{k, 4}, 'YTick', specs{k, 5}, ...
        'FontSize', S.font, 'LineWidth', 0.7, 'Box', 'on', 'TickLabelInterpreter', 'latex');
    title(ax, specs{k, 2}, 'FontSize', S.font+1, 'FontWeight', 'normal', 'Interpreter', 'latex');
    export_fig(fig, fullfile(out, specs{k, 6}), '-pdf', '-painters', '-transparent');
    close(fig);
end
end

function obstacle_slice(c, out)
% treasure chest with an internal obstacle: nodes in the slab |x_3| < h/2, projected on (x_1, x_2)
S = style();
box = [-0.05 1.85 -0.65 0.65]; w = 8.6;
fig = figure('Units', 'centimeters', 'Position', [2 2 w+1.4 w*(box(4)-box(3))/(box(2)-box(1))+1.6], 'Color', 'w');
ax = axes(fig); hold(ax, 'on');
xi = c.int_nodes; xb = c.bdry_nodes; xg = c.ghost_nodes;
plot(ax, xi(:,1), xi(:,2), '.', 'Color', [0 0.30 0.75], 'MarkerSize', 5);
plot(ax, xb(:,1), xb(:,2), '.', 'Color', 'k', 'MarkerSize', 8);
plot(ax, xg(:,1), xg(:,2), 'x', 'Color', [0.80 0.10 0.10], 'MarkerSize', 3, 'LineWidth', 0.8);
axis(ax, 'equal'); set(ax, 'XLim', box(1:2), 'YLim', box(3:4), 'XTick', [0 0.6 1.2 1.8], 'YTick', [-0.6 0 0.6], ...
    'FontSize', S.font, 'LineWidth', 0.7, 'Box', 'on', 'TickLabelInterpreter', 'latex');
xlabel(ax, '$x_1$', 'Interpreter', 'latex', 'FontSize', S.font+1);
ylabel(ax, '$x_2$', 'Interpreter', 'latex', 'FontSize', S.font+1);
title(ax, sprintf('$|x_3|<h/2$, $h=%.2f$', c.h), 'FontSize', S.font+1, 'FontWeight', 'normal', 'Interpreter', 'latex');
export_fig(fig, fullfile(out, 'nodes_obstacle_slice'), '-pdf', '-painters', '-transparent');
close(fig);
end

function velocity(V, out)
% velocity fields: speed (color) and streamlines; 2D fields and 3D sections y = 0 (in-plane components)
S = style(); vmax = 1.18;
names = fieldnames(V);
for k = 1:numel(names)
    P = V.(names{k});
    x = P.x(:)'; y = P.y(:)';
    u = P.u; v = P.v; s = P.speed;
    w = 6.4; hgt = w*(max(y)-min(y))/(max(x)-min(x)) + 1.3;
    fig = figure('Units', 'centimeters', 'Position', [2 2 w+1.2 hgt], 'Color', 'w');
    ax = axes(fig); hold(ax, 'on');
    K = jsondecode(fileread(fullfile(fileparts(out), '..', '..', 'Outputs_and_results', 'paper_figures', 'plot_masks.json')));
    g = regexprep(names{k}, '_(forward|transport_smooth|transport_c1)$', '');
    Mk = cell2mat(cellfun(@(r) r == '1', K.(g).rows, 'UniformOutput', false));
    [xf, yf, sf] = upsample(x, y, s, 4, Mk);
    im = imagesc(ax, xf, yf, sf); set(im, 'AlphaData', ~isnan(sf));
    set(ax, 'YDir', 'normal'); colormap(ax, parula(256)); clim(ax, [0 vmax]);
    uu = u; vv = v; uu(isnan(uu)) = 0; vv(isnan(vv)) = 0;
    [X, Y] = meshgrid(x, y);
    hs = streamslice(ax, X, Y, uu, vv, 1.2);
    set(hs, 'Color', [0 0 0], 'LineWidth', 0.5);
    axis(ax, 'equal'); set(ax, 'XLim', [min(x) max(x)], 'YLim', [min(y) max(y)], 'FontSize', S.font, ...
        'LineWidth', 0.7, 'Box', 'on', 'TickLabelInterpreter', 'latex', 'Layer', 'top');
    xlabel(ax, '$x$', 'Interpreter', 'latex', 'FontSize', S.font+1);
    if startsWith(names{k}, 'h_channel'), yl = '$y$'; else, yl = '$z$'; end
    ylabel(ax, yl, 'Interpreter', 'latex', 'FontSize', S.font+1);
    export_fig(fig, fullfile(out, ['velocity_' names{k}]), '-pdf', '-opengl', '-r600');   % raster: painters downsamples images
    close(fig);
end
% colorbar for all panels
fig = figure('Units', 'centimeters', 'Position', [2 2 6.4 2.6], 'Color', 'w');
ax = axes(fig, 'Units', 'centimeters', 'Position', [0.6 2.3 5.2 0.1]); axis(ax, 'off');
colormap(ax, parula(256)); clim(ax, [0 vmax]);
cb = colorbar(ax, 'southoutside');                    % ticks and label below the bar
cb.TickLabelInterpreter = 'latex'; cb.FontSize = S.font; cb.Label.String = 'speed $\|v\|$';
cb.Label.Interpreter = 'latex'; cb.Label.FontSize = S.font+1;
export_fig(fig, fullfile(out, 'legend_velocity'), '-pdf', '-painters', '-transparent');
close(fig);
end

function node_stack(N, out)
% stacked representation on the treasure chest, one PDF per panel:
%   node_stack_parameters.pdf, node_stack_parametric.pdf, node_stack_implicit.pdf, node_stack_nodes.pdf
S = style();
hl = [0.90 0.45 0.00];                       % highlighted support, its image, and its ball
blue = [0 0.30 0.75]; red = [0.80 0.10 0.10];
w = 6.1; hgt = 5.0;                          % identical canvas for all four panels

% (a) parameter domain of the cap
[fig, ax] = flat_panel(w, hgt);
o = N.outline;
fill(ax, o(:,1), o(:,2), [0.92 0.95 1.00], 'EdgeColor', 'none');
c = N.cells;
for k = 1:size(c, 1)
    rectangle(ax, 'Position', [c(k,1) c(k,2) c(k,3)-c(k,1) c(k,4)-c(k,2)], 'EdgeColor', [0.55 0.55 0.55], 'LineWidth', 0.4);
end
plot(ax, o([1:end 1],1), o([1:end 1],2), 'k-', 'LineWidth', 1.1);
r = N.cell_j; s = N.support_j;
fill(ax, r([1 3 3 1]), r([2 2 4 4]), hl, 'FaceAlpha', 0.45, 'EdgeColor', 'none');
rectangle(ax, 'Position', [s(1) s(2) s(3)-s(1) s(4)-s(2)], 'EdgeColor', hl, 'LineStyle', '--', 'LineWidth', 1.1);
text(ax, s(3), s(4), '$S_j$', 'Interpreter', 'latex', 'FontSize', S.font+1, 'Color', hl, ...
    'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');
axis(ax, 'equal'); set(ax, 'XLim', [-1.05 1.05], 'YLim', [-1.05 1.05], 'XTick', -1:1, 'YTick', -1:1);
xlabel(ax, '$z_1$', 'Interpreter', 'latex'); ylabel(ax, '$z_2$', 'Interpreter', 'latex');
flat_finish(ax, fig, '(a) Cap parameters', fullfile(out, 'node_stack_parameters'));

% (b) parametric layer in 3D
fig = figure('Units', 'centimeters', 'Position', [2 2 w hgt], 'Color', 'w');
ax = axes(fig, 'Position', [0 0 1 1]); hold(ax, 'on');
b = N.box; v = N.view(:)';
corners = [b(1) b(3) b(5); b(2) b(3) b(5); b(1) b(4) b(5); b(2) b(4) b(5); ...
           b(1) b(3) b(6); b(2) b(3) b(6); b(1) b(4) b(6); b(2) b(4) b(6)];
for i = 1:8
    for k = i+1:8
        d = corners(k,:) - corners(i,:);
        if nnz(d) ~= 1, continue, end
        % an edge is hidden if both faces that contain it face away from the viewer
        fixed = find(d == 0); vis = false;
        for a = fixed
            n = zeros(1, 3); n(a) = sign(corners(i,a) - mean(b(2*a-1:2*a)));
            if a == 1 && corners(i,a) == b(2), continue, end     % the face x = L carries the cap
            vis = vis || dot(n, v) > 0;
        end
        if vis, st = {'k-', 'LineWidth', 1.0}; else, st = {'--', 'Color', [0.6 0.6 0.6], 'LineWidth', 0.6}; end
        plot3(ax, corners([i k],1), corners([i k],2), corners([i k],3), st{:});
    end
end
C = N.cap;
surf(ax, C(:,:,1), C(:,:,2), C(:,:,3), 'FaceColor', [0.93 0.93 0.93], 'EdgeColor', [0.65 0.65 0.65], ...
    'EdgeAlpha', 0.6, 'LineWidth', 0.3, 'FaceAlpha', 0.9);
R = N.patch;
surf(ax, R(:,:,1), R(:,:,2), R(:,:,3), 'FaceColor', hl, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
rim = N.rim; plot3(ax, rim(:,1), rim(:,2), rim(:,3), 'k-', 'LineWidth', 1.0);
[sx, sy, sz] = sphere(40); cb = N.ball_center; rb = N.ball_radius;
surf(ax, cb(1)+rb*sx, cb(2)+rb*sy, cb(3)+rb*sz, 'FaceColor', hl, 'FaceAlpha', 0.12, 'EdgeColor', 'none');
xf = N.front_nodes; xg = N.ghosts_near;
plot3(ax, xf(:,1), xf(:,2), xf(:,3), '.', 'Color', 'k', 'MarkerSize', 3);
plot3(ax, xg(:,1), xg(:,2), xg(:,3), 'x', 'Color', red, 'MarkerSize', 3, 'LineWidth', 0.7);
axis(ax, 'equal', 'off'); view(ax, v); camproj(ax, 'orthographic'); camzoom(ax, 1.15); ax.Clipping = 'off';
export_fig(fig, fullfile(out, 'node_stack_parametric'), '-pdf', '-opengl', '-r400', '-nocrop');
close(fig);

% (c) implicit layer in the slice x_3 = 0
[fig, ax] = flat_panel(w, hgt);
sec = N.section; lim = [-0.15 1.95 -0.75 0.75];
fill(ax, sec(:,1), sec(:,2), [0.70 0.80 1.00], 'EdgeColor', 'none');
t = linspace(0, 2*pi, 73);
dd = N.discs;
for k = 1:size(dd, 1)
    fill(ax, dd(k,1)+dd(k,3)*cos(t), dd(k,2)+dd(k,3)*sin(t), [0.86 0.86 0.86], 'EdgeColor', 'none');
end
g0 = N.grid_lo; g1 = N.grid_hi; h = N.cell;
gx = g0(1):h:g1(1)+h/2; gy = g0(2):h:g1(2)+h/2;
plot(ax, [gx; gx], repmat([g0(2); g1(2)], 1, numel(gx)), '-', 'Color', [0.80 0.80 0.80], 'LineWidth', 0.25);
plot(ax, repmat([g0(1); g1(1)], 1, numel(gy)), [gy; gy], '-', 'Color', [0.80 0.80 0.80], 'LineWidth', 0.25);
for k = 1:size(dd, 1)
    plot(ax, dd(k,1)+dd(k,3)*cos(t), dd(k,2)+dd(k,3)*sin(t), '-', 'Color', [0.55 0.55 0.55], 'LineWidth', 0.3);
end
plot(ax, sec(:,1), sec(:,2), 'k-', 'LineWidth', 1.1);
fi = N.free_inside; [~, m] = min(vecnorm(fi - mean(fi, 1), 2, 2));
text(ax, fi(m,1), fi(m,2), '$\sigma=+1$', 'Interpreter', 'latex', 'FontSize', S.font+1, ...
    'BackgroundColor', 'w', 'Margin', 1, 'HorizontalAlignment', 'center');
text(ax, lim(2)-0.03, lim(3)+0.03, '$\sigma=-1$', 'Interpreter', 'latex', 'FontSize', S.font+1, ...
    'BackgroundColor', 'w', 'Margin', 1, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'bottom');
axis(ax, 'equal'); set(ax, 'XLim', lim(1:2), 'YLim', lim(3:4), 'XTick', [0 0.6 1.2 1.8], 'YTick', [-0.6 0 0.6]);
xlabel(ax, '$x_1$', 'Interpreter', 'latex'); ylabel(ax, '$x_2$', 'Interpreter', 'latex');
flat_finish(ax, fig, '(c) Implicit layer, $x_3=0$', fullfile(out, 'node_stack_implicit'));

% (d) nodes of the slab |x_3| < h/2
[fig, ax] = flat_panel(w, hgt);
fill(ax, sec(:,1), sec(:,2), [0.94 0.96 1.00], 'EdgeColor', 'none');
plot(ax, sec(:,1), sec(:,2), 'k-', 'LineWidth', 0.8);
xi = N.int_nodes; xb = N.bdry_nodes; xg = N.ghost_nodes;
plot(ax, xi(:,1), xi(:,2), '.', 'Color', blue, 'MarkerSize', 6);
plot(ax, xb(:,1), xb(:,2), '.', 'Color', 'k', 'MarkerSize', 8);
plot(ax, xg(:,1), xg(:,2), 'x', 'Color', red, 'MarkerSize', 3.5, 'LineWidth', 0.8);
axis(ax, 'equal'); set(ax, 'XLim', lim(1:2), 'YLim', lim(3:4), 'XTick', [0 0.6 1.2 1.8], 'YTick', [-0.6 0 0.6]);
xlabel(ax, '$x_1$', 'Interpreter', 'latex'); ylabel(ax, '$x_2$', 'Interpreter', 'latex');
flat_finish(ax, fig, '(d) Nodes, $|x_3|<h/2$', fullfile(out, 'node_stack_nodes'));
end

function [fig, ax] = flat_panel(w, hgt)
fig = figure('Units', 'centimeters', 'Position', [2 2 w hgt], 'Color', 'w');
ax = axes(fig, 'Units', 'centimeters', 'Position', [1.15 0.95 w-1.35 hgt-1.15]); hold(ax, 'on');
end

function flat_finish(ax, fig, ttl, file)
S = style();
set(ax, 'FontSize', S.font, 'LineWidth', 0.7, 'Box', 'on', 'TickLabelInterpreter', 'latex', 'Layer', 'top');
% panel titles are set in LaTeX, so that all four panels share one canvas and one font
export_fig(fig, file, '-pdf', '-painters', '-nocrop');
close(fig);
end

function [xf, yf, F] = upsample(x, y, F, k, M)
% Display only: values on a k times finer grid by local PHS + polynomial interpolation with the basis of
% the paper (Section 3.1) for p = 4 in the plane (d = 2): r^m with m = max(p+1,5) = 5, polynomials of total
% degree p+1 = 5, and the n_s = 2*nchoosek(p+1+d,d)+1 = 43 nearest grid values inside the domain, in
% coordinates centered and scaled per stencil. M is the exact domain mask on the fine grid
% (scripts/paper_plot_masks.py), so the edge is not drawn in whole cells of the computation grid.
p = 4; d = 2; m = max(p+1, 5); deg = p+1; ns = 2*nchoosek(deg+d, d)+1;
xf = linspace(x(1), x(end), k*(numel(x)-1)+1); yf = linspace(y(1), y(end), k*(numel(y)-1)+1);
[X, Y] = meshgrid(x, y); [Xf, Yf] = meshgrid(xf, yf);
inside = ~isnan(F);
P = [X(inside) Y(inside)]; f = F(inside);
Q = [Xf(M) Yf(M)];
[ex, ey] = meshgrid(0:deg, 0:deg); keep = ex + ey <= deg; ex = ex(keep)'; ey = ey(keep)'; np = numel(ex);
vals = zeros(size(Q, 1), 1);
chunk = 2000;
for s = 1:chunk:size(Q, 1)
    I = s:min(s+chunk-1, size(Q, 1)); q = Q(I, :); nq = numel(I);
    D2 = (q(:,1) - P(:,1)').^2 + (q(:,2) - P(:,2)').^2;          % nq x |P|
    [~, nb] = mink(D2, ns, 2);                                     % nq x ns nearest inside values
    sx = reshape(P(nb, 1), nq, ns); sy = reshape(P(nb, 2), nq, ns);
    cx = mean(sx, 2); cy = mean(sy, 2);
    sc = max(sqrt((sx - cx).^2 + (sy - cy).^2), [], 2);             % scale per stencil
    sx = (sx - cx)./sc; sy = (sy - cy)./sc; qx = (q(:,1) - cx)./sc; qy = (q(:,2) - cy)./sc;
    A = zeros(ns+np, ns+np, nq); b = zeros(ns+np, 1, nq);
    R = sqrt((permute(sx, [2 3 1]) - permute(sx, [3 2 1])).^2 + (permute(sy, [2 3 1]) - permute(sy, [3 2 1])).^2);
    A(1:ns, 1:ns, :) = R.^m;
    Pm = permute(sx, [2 3 1]).^ex .* permute(sy, [2 3 1]).^ey;    % ns x np x nq
    A(1:ns, ns+1:end, :) = Pm; A(ns+1:end, 1:ns, :) = permute(Pm, [2 1 3]);
    rq = sqrt((permute(sx, [2 3 1]) - permute(qx, [2 3 1])).^2 + (permute(sy, [2 3 1]) - permute(qy, [2 3 1])).^2);
    b(1:ns, 1, :) = rq.^m;
    b(ns+1:end, 1, :) = permute(qx, [2 3 1]).^(ex(:)) .* permute(qy, [2 3 1]).^(ey(:));
    w = pagemldivide(A, b);                                        % interpolation weights per point
    fv = reshape(f(nb)', ns, 1, nq);
    vals(I) = squeeze(sum(w(1:ns, 1, :).*fv, 1));
end
F = NaN(size(Xf)); F(M) = vals;
end
