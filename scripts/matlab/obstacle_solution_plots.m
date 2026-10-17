function obstacle_solution_plots()
% Panels of the figure of the problem with an internal obstacle (Section 6.5), on the section y = 0:
%   obstacle_velocity.pdf            velocity 1.18 w, colored by speed, with streamlines
%   obstacle_solution_u.pdf          manufactured solution u*(T)
%   obstacle_solution_reaction.pdf   recovered reaction rate R(u*, x; theta_h)
%   obstacle_solution_error.pdf      |R(u*, x; theta_h) - R(u*, x; theta*)|
% Data: Outputs_and_results/paper_figures/obstacle_solution.json (scripts/paper_obstacle_solution_fields.py)
% and the obstacle_chest_forward entry of plot_data.json (scripts/paper_figures.py). All four panels have
% the same axes size, fonts, and a colorbar on the right, so they align in a 2x2 array at width 0.48\textwidth.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath('E:\kernelpack-tools\export_fig');
gs = 'E:\texlive\2025\tlpkg\tlgs';
setenv('GS_LIB', [gs '\Resource\Init;' gs '\lib;' gs '\Resource\Font;' gs '\kanji']);
user_string('ghostscript', [gs '\bin\gswin64c.exe']);
D = jsondecode(fileread(fullfile(root, 'Outputs_and_results', 'paper_figures', 'obstacle_solution.json')));
P = jsondecode(fileread(fullfile(root, 'Outputs_and_results', 'paper_figures', 'plot_data.json')));
V = P.velocity.obstacle_chest_forward;
K = jsondecode(fileread(fullfile(root, 'Outputs_and_results', 'paper_figures', 'plot_masks.json')));
M = cell2mat(cellfun(@(r) r == '1', K.obstacle_chest.rows, 'UniformOutput', false));
out = fullfile(root, 'KernelPackJaxPaper', 'Images', 'plots');

x = V.x(:)'; y = V.y(:)';
[fig, ax] = panel(x, y, grid2(V.speed), M, [0 1.18], 'speed $\|v\|$');
[X, Y] = meshgrid(x, y);
uu = grid2(V.u); vv = grid2(V.v); uu(isnan(uu)) = 0; vv(isnan(vv)) = 0;
hs = streamslice(ax, X, Y, uu, vv, 1.2);
set(hs, 'Color', [0 0 0], 'LineWidth', 0.5);
save_panel(fig, fullfile(out, 'obstacle_velocity'));

x = D.x(:)'; y = D.y(:)';
[fig, ~] = panel(x, y, grid2(D.u_exact), M, [], '$u_\star(\cdot,T)$');
save_panel(fig, fullfile(out, 'obstacle_solution_u'));
[fig, ~] = panel(x, y, grid2(D.reaction_fit), M, [], '$R(u_\star,x;\theta_h)$');
save_panel(fig, fullfile(out, 'obstacle_solution_reaction'));
[fig, ~] = panel(x, y, grid2(D.reaction_error), M, [], '$|R_h-R_\star|$');
save_panel(fig, fullfile(out, 'obstacle_solution_error'));
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

function [fig, ax] = panel(x, y, F, M, lims, label)
% fixed layout in cm: axes 5.6 wide, height from the aspect ratio, colorbar to the right
font = 9; aw = 5.6; ah = aw*(max(y)-min(y))/(max(x)-min(x));
left = 1.05; bottom = 0.95; cbw = 0.22; gap = 0.2; right = 1.9; top = 0.15;
fig = figure('Units', 'centimeters', 'Position', [2 2 left+aw+gap+cbw+right bottom+ah+top], 'Color', 'w');
ax = axes(fig, 'Units', 'centimeters', 'Position', [left bottom aw ah]); hold(ax, 'on');
e = floor(log10(max(abs(F(:)), [], 'omitnan')));          % small fields: scale and state the factor
if e <= -3, F = F/10^e; label = [label sprintf(' $(\\times10^{%d})$', e)]; end
[xf, yf, Ff] = upsample(x, y, F, 4, M);
im = imagesc(ax, xf, yf, Ff); set(im, 'AlphaData', ~isnan(Ff));
set(ax, 'YDir', 'normal'); colormap(ax, parula(256));
if ~isempty(lims), clim(ax, lims); else, clim(ax, [min(F(:), [], 'omitnan') max(F(:), [], 'omitnan')]); end
set(ax, 'XLim', [min(x) max(x)], 'YLim', [min(y) max(y)], 'FontSize', font, 'LineWidth', 0.7, ...
    'Box', 'on', 'TickLabelInterpreter', 'latex', 'Layer', 'top', 'XTick', 0:0.5:1.5, 'YTick', [-0.3 0 0.3]);
xlabel(ax, '$x$', 'Interpreter', 'latex', 'FontSize', font+1);
ylabel(ax, '$z$', 'Interpreter', 'latex', 'FontSize', font+1);
cb = colorbar(ax, 'eastoutside'); cb.Units = 'centimeters';
cb.Position = [left+aw+gap bottom cbw ah]; cb.TickLabelInterpreter = 'latex'; cb.FontSize = font;
cb.Label.String = label; cb.Label.Interpreter = 'latex'; cb.Label.FontSize = font+1;
set(ax, 'Position', [left bottom aw ah]);                % colorbar must not shrink the axes
end

function save_panel(fig, file)
export_fig(fig, file, '-pdf', '-opengl', '-r600', '-nocrop');   % raster: painters downsamples images
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
