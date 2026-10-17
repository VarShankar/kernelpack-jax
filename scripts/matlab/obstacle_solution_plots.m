function obstacle_solution_plots()
% Solution panels of the problem with an internal obstacle (Section 6.5) on the section y = 0, from
% Outputs_and_results/paper_figures/obstacle_solution.json (scripts/paper_obstacle_solution_fields.py):
%   obstacle_solution_u.pdf          manufactured solution u*(T)
%   obstacle_solution_reaction.pdf   recovered reaction rate R(u*, x; theta_h)
%   obstacle_solution_error.pdf      |R(u*, x; theta_h) - R(u*, x; theta*)| (log scale)
% Written to KernelPackJaxPaper/Images/plots/ with export_fig, in the style of the velocity panels.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath('E:\kernelpack-tools\export_fig');
gs = 'E:\texlive\2025\tlpkg\tlgs';
setenv('GS_LIB', [gs '\Resource\Init;' gs '\lib;' gs '\Resource\Font;' gs '\kanji']);
user_string('ghostscript', [gs '\bin\gswin64c.exe']);
D = jsondecode(fileread(fullfile(root, 'Outputs_and_results', 'paper_figures', 'obstacle_solution.json')));
out = fullfile(root, 'KernelPackJaxPaper', 'Images', 'plots');
x = D.x(:)'; y = D.y(:)';
draw(x, y, grid2(D.u_exact), false, '$u_\star(\cdot,T)$', fullfile(out, 'obstacle_solution_u'));
draw(x, y, grid2(D.reaction_fit), false, '$R(u_\star,x;\theta_h)$', fullfile(out, 'obstacle_solution_reaction'));
draw(x, y, grid2(D.reaction_error), true, '$|R_h-R_\star|$', fullfile(out, 'obstacle_solution_error'));
end

function F = grid2(c)
% JSON rows (null -> NaN) to a matrix
if iscell(c)
    F = cell2mat(cellfun(@(r) reshape(fillnan(r), 1, []), c, 'UniformOutput', false));
else
    F = c;
end
end

function r = fillnan(r)
if iscell(r), r(cellfun(@isempty, r)) = {NaN}; r = cell2mat(r); end
end

function draw(x, y, F, logscale, label, file)
font = 9; w = 6.4; hgt = w*(max(y)-min(y))/(max(x)-min(x)) + 1.9;
fig = figure('Units', 'centimeters', 'Position', [2 2 w+1.2 hgt], 'Color', 'w');
ax = axes(fig); hold(ax, 'on');
G = F; if logscale, G = log10(max(F, realmin)); end
im = imagesc(ax, x, y, G); set(im, 'AlphaData', ~isnan(F));
set(ax, 'YDir', 'normal'); colormap(ax, parula(256));
axis(ax, 'equal'); set(ax, 'XLim', [min(x) max(x)], 'YLim', [min(y) max(y)], 'FontSize', font, ...
    'LineWidth', 0.7, 'Box', 'on', 'TickLabelInterpreter', 'latex', 'Layer', 'top');
xlabel(ax, '$x$', 'Interpreter', 'latex', 'FontSize', font+1);
ylabel(ax, '$z$', 'Interpreter', 'latex', 'FontSize', font+1);
cb = colorbar(ax, 'southoutside'); cb.TickLabelInterpreter = 'latex'; cb.FontSize = font;
cb.Label.String = label; cb.Label.Interpreter = 'latex'; cb.Label.FontSize = font+1;
if logscale
    hi = ceil(max(G(:), [], 'omitnan')); lo = hi - 6;      % six decades below the largest error
    G(G < lo) = lo; set(im, 'CData', G);
    clim(ax, [lo hi]); cb.Ticks = lo:2:hi;
    cb.TickLabels = arrayfun(@(k) sprintf('$10^{%d}$', k), lo:2:hi, 'UniformOutput', false);
end
export_fig(fig, file, '-pdf', '-painters', '-transparent');
close(fig);
end
