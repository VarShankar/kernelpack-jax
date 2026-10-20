function fastol_field_figures(F, out)
% Benchmark field figures (appendix): first supplied simulation of each data set, two panels,
% 2D colour fields from linear interpolation of the supplied samples (scatteredInterpolant,
% 'linear', no extrapolation) on a fine grid, masked by the exact domain; the 3D benchmarks
% (species transport, forced turbulence) use local cubic PHS interpolation (phs_local). Panels of the same quantity share a colour
% scale. Exported raster (-opengl -r600) because vector export downsamples images.
%   fastol_fields_<case>.pdf
S = fastol_plot_style();
W = 4.8;                                    % axes width (cm) of every field panel
twopi = {'$0$', '$\pi$', '$2\pi$'};
sp = @(u) sqrt(sum(u.^2, 2));

% ---- cylinder: (A) initial speed, (B) speed at t = 10, shared scale
C = F.cylinder; xc = [1.5 7]; R = 0.8;
in = @(X, Y) (X - xc(1)).^2 + (Y - xc(2)).^2 >= R^2;
[x, y] = grid2([0 20], [0 14], 900);
A = field2(C.points, sp(C.u0), x, y, in); B = field2(C.points, sp(C.u), x, y, in);
P = panels(W, W*14/20, 'shared');
draw(P.ax(1), x, y, A, P.clim_shared([A B])); draw(P.ax(2), x, y, B, P.clim_shared([A B]));
for a = P.ax, circle(a, xc, R); axes_xy(a, [0 20], [0 14], 0:10:20, 0:7:14); end
title(P.ax(1), '(A) Initial speed'); title(P.ax(2), '(B) Speed at $t=10$');
P.colorbar(1, 'speed $\|\mathbf{u}\|$'); P.save(fullfile(out, 'fastol_fields_flow_cylinder_shedding'));

% ---- Taylor--Green: (A) initial speed, (B) speed at t = 1, shared scale
T = F.taylor_green;
[x, y] = grid2([0 2*pi], [0 2*pi], 700);
A = field2(T.points, sp(T.u0), x, y, []); B = field2(T.points, sp(T.u), x, y, []);
P = panels(W, W, 'shared'); cl = P.clim_shared([A B]);
draw(P.ax(1), x, y, A, cl); draw(P.ax(2), x, y, B, cl);
for a = P.ax, axes_xy(a, [0 2*pi], [0 2*pi], [0 pi 2*pi], [0 pi 2*pi], twopi); end
title(P.ax(1), '(A) Initial speed'); title(P.ax(2), '(B) Speed at $t=1$');
P.colorbar(1, 'speed $\|\mathbf{u}\|$'); P.save(fullfile(out, 'fastol_fields_taylor_green'));

% ---- lid-driven cavity: (A) initial speed, (B) speed at t = 5, shared scale
L = F.lid_cavity;
[x, y] = grid2([0 1], [0 1], 700);
A = field2(L.points, sp(L.u0), x, y, []); B = field2(L.points, sp(L.u), x, y, []);
P = panels(W, W, 'shared'); cl = P.clim_shared([A B]);
draw(P.ax(1), x, y, A, cl); draw(P.ax(2), x, y, B, cl);
for a = P.ax, axes_xy(a, [0 1], [0 1], 0:0.5:1, 0:0.5:1); end
title(P.ax(1), '(A) Initial speed'); title(P.ax(2), '(B) Speed at $t=5$');
P.colorbar(1, 'speed $\|\mathbf{u}\|$'); P.save(fullfile(out, 'fastol_fields_lid_cavity_flow'));

% ---- backward-facing step: (A) inlet profile, (B) speed at t = 5 (transverse direction expanded)
Bs = F.backward_step;
yb = Bs.bc_points(:, 2); ub = Bs.inlet(:);
ub([1 2]) = ub([2 1]);                      % stored inlet: first two sample positions exchanged
phi = 16*yb.*(0.5 - yb);                    % unit parabola on [0, 0.5]
amp = phi \ ub;                             % amplitude of the drawn profile
fprintf('backward step: inlet amplitude %.6g, relative residual %.3g\n', amp, norm(ub - amp*phi)/norm(ub));
[x, y] = grid2([0 15], [-0.5 0.5], 1500, 400);
B = field2(Bs.points, sp(Bs.u), x, y, []);
P = panels(W, 0.7*W, 'separate');
a = P.ax(1); yy = linspace(0, 0.5, 201);
plot(a, amp*16*yy.*(0.5 - yy), yy, '-', 'Color', S.model_color{1}, 'LineWidth', 1.2);
plot(a, ub, yb, 'o', 'Color', S.model_color{1}, 'MarkerFaceColor', S.model_color{1}, 'MarkerSize', S.ms);
set(a, 'XLim', [0 1.12*amp], 'YLim', [-0.5 0.5], 'YTick', -0.5:0.25:0.5);
xlabel(a, 'inlet velocity $u_x$'); ylabel(a, '$y$'); grid(a, 'on'); a.GridAlpha = 0.12;
finish_field_axes(a);
draw(P.ax(2), x, y, B, [0 max(B(:))]);
axes_xy(P.ax(2), [0 15], [-0.5 0.5], 0:5:15, -0.5:0.5:0.5);
title(P.ax(1), '(A) Inlet profile'); title(P.ax(2), '(B) Speed at $t=5$');
P.colorbar(2, 'speed $\|\mathbf{u}\|$'); P.save(fullfile(out, 'fastol_fields_backward_facing_step'));

% ---- buoyancy cavity: (A) initial temperature, (B) speed at t = 5
Q = F.buoyancy;
[x, y] = grid2([0 1], [0 1], 700);
A = field2(Q.points, Q.T0, x, y, []); B = field2(Q.points, sp(Q.u), x, y, []);
P = panels(W, W, 'separate');
draw(P.ax(1), x, y, A, [min(A(:)) max(A(:))]); draw(P.ax(2), x, y, B, [0 max(B(:))]);
for a = P.ax, axes_xy(a, [0 1], [0 1], 0:0.5:1, 0:0.5:1); end
title(P.ax(1), '(A) Initial temperature'); title(P.ax(2), '(B) Speed at $t=5$');
P.colorbar(1, 'temperature $T$'); P.colorbar(2, 'speed $\|\mathbf{u}\|$');
P.save(fullfile(out, 'fastol_fields_buoyancy_cavity_flow'));

% ---- merging vortices: (A) initial vorticity, (B) final speed (t = 0.4)
M = F.merge;
[x, y] = grid2([0 2*pi], [0 2*pi], 700);
A = field2(M.points(:, 1:2), M.w0, x, y, []); B = field2(M.points(:, 1:2), sp(M.u), x, y, []);
P = panels(W, W, 'separate');
draw(P.ax(1), x, y, A, [min(A(:)) max(A(:))]); draw(P.ax(2), x, y, B, [0 max(B(:))]);
for a = P.ax, axes_xy(a, [0 2*pi], [0 2*pi], [0 pi 2*pi], [0 pi 2*pi], twopi); end
title(P.ax(1), '(A) Initial vorticity'); title(P.ax(2), '(B) Final speed');
P.colorbar(1, 'vorticity $\omega$'); P.colorbar(2, 'speed $\|\mathbf{u}\|$');
P.save(fullfile(out, 'fastol_fields_merge_vortices_easier'));

% ---- species transport: (A) inlet velocity on z = 0, (B) speed at t = 0.5 in the mixer (3D)
species_figure(F.species, F.species_mesh, out);

% ---- forced turbulence: (A) forcing magnitude, (B) speed at t = 0.5, on three faces of the box
forced_figure(F.forced, out);
end

% ------------------------------------------------------------------ 3D benchmarks
function species_figure(Sp, Me, out)
S = fastol_plot_style();
sp = @(u) sqrt(sum(u.^2, 2));
R = 0.01; L = 0.26;
fig = figure('Units', 'centimeters', 'Position', [2 2 17 7.6], 'Color', 'w', 'Visible', 'off');
% (A) inlet velocity on the inlet faces of the mesh (exact geometry), colour by nearest sample
a = axes(fig, 'Units', 'centimeters', 'Position', [1.3 2.6 4.2 4.2]); hold(a, 'on');
inlet = [Me.markers.inlet_gas; Me.markers.inlet_air];
c = (Me.nodes(inlet(:, 1), 1:2) + Me.nodes(inlet(:, 2), 1:2) + Me.nodes(inlet(:, 3), 1:2) + Me.nodes(inlet(:, 4), 1:2))/4;
cv = Sp.inlet(knnsearch(Sp.bc_points(:, 1:2), c));
patch(a, 'Faces', inlet, 'Vertices', Me.nodes(:, 1:2), 'FaceVertexCData', cv, 'FaceColor', 'flat', 'EdgeColor', 'none');
t = linspace(0, 2*pi, 400); plot(a, R*cos(t), R*sin(t), 'k-', 'LineWidth', 0.5);
axis(a, 'equal'); set(a, 'XLim', 1.05*[-R R], 'YLim', 1.05*[-R R], 'XTick', [-R 0 R], 'YTick', [-R 0 R]);
colormap(a, parula(256)); clim(a, [min(cv) max(cv)]);
xlabel(a, '$x$'); ylabel(a, '$y$'); title(a, '(A) Inlet velocity, $z=0$');
finish_field_axes(a);
cb = colorbar(a, 'southoutside'); cb_style(cb, S, 'inlet velocity');
cb.Units = 'centimeters'; cb.Position = [1.7 1.15 3.4 0.22];
% (B) speed at t = 0.5 on the plane x = 0. The samples are interpolated by natural neighbours
% (no overshoot across the blades); plotted with the axis z horizontal and the transverse
% direction expanded threefold.
b = axes(fig, 'Units', 'centimeters', 'Position', [6.4 1.9 10.3 5.2]); hold(b, 'on');
um = sp(Sp.u);
I = scatteredInterpolant(Sp.points, um, 'natural', 'nearest');
[z, y] = meshgrid(linspace(0, L, 1200), linspace(-R, R, 140));
v = reshape(I(zeros(numel(z), 1), y(:), z(:)), size(z));
surf(b, z, zeros(size(z)), y, v, 'EdgeColor', 'none');               % plot axes: (z, x, y)
plot3(b, zeros(size(t)), R*cos(t), R*sin(t), 'k-', 'LineWidth', 0.4);   % outline of the pipe
plot3(b, L*ones(size(t)), R*cos(t), R*sin(t), 'k-', 'LineWidth', 0.4);
for s = [-1 1]
    plot3(b, [0 L], [0 0], s*[R R], 'k-', 'LineWidth', 0.4);
end
colormap(b, parula(256)); clim(b, [0 max(um)]); shading(b, 'interp');
daspect(b, [3 1 1]); view(b, [-28 22]); axis(b, 'tight'); axis(b, 'off');
text(b, L/2, 0, -1.9*R, '$z$', 'Interpreter', 'latex', 'FontSize', S.label, 'HorizontalAlignment', 'center');
title(b, '(B) Speed at $t=0.5$', 'FontWeight', 'normal', 'FontSize', S.label, 'Interpreter', 'latex');
cb = colorbar(b, 'southoutside'); cb_style(cb, S, 'speed $\|\mathbf{u}\|$');
cb.Units = 'centimeters'; cb.Position = [8.9 1.15 5.6 0.22];
export_fig(fig, fullfile(out, 'fastol_fields_species_transport'), '-pdf', '-opengl', '-r600');
close(fig); fprintf('wrote %s.pdf\n', fullfile(out, 'fastol_fields_species_transport'));
end

function forced_figure(Ft, out)
S = fastol_plot_style();
sp = @(u) sqrt(sum(u.^2, 2));
Lp = 2*pi;
[pp, idx] = periodic_images(Ft.points, Lp, 6*Lp/32);
n = 160; g = linspace(0, Lp, n); [G1, G2] = meshgrid(g, g);
faces = {[zeros(n*n, 1) G1(:) G2(:)], [G1(:) zeros(n*n, 1) G2(:)], [G1(:) G2(:) Lp*ones(n*n, 1)]};
fig = figure('Units', 'centimeters', 'Position', [2 2 15 7.8], 'Color', 'w', 'Visible', 'off');
vals = {sp(Ft.f), sp(Ft.u)};
names = {'(A) Forcing', '(B) Speed at $t=0.5$'};
labels = {'forcing magnitude $\|\mathbf{f}\|$', 'speed $\|\mathbf{u}\|$'};
comp = {Ft.f, Ft.u};
for k = 1:2
    a = axes(fig, 'Units', 'centimeters', 'Position', [0.9 + (k-1)*7.3, 2.2, 5.9, 5.0]); hold(a, 'on');
    for fq = 1:3
        q = faces{fq}; U = zeros(size(q, 1), 3);
        for c = 1:3, U(:, c) = phs_local(pp, comp{k}(idx, c), q, 30); end
        V = reshape(sp(U), n, n);
        Q = reshape(q, n, n, 3);
        surf(a, Q(:, :, 1), Q(:, :, 2), Q(:, :, 3), V, 'EdgeColor', 'none');
    end
    e = [0 Lp];                                   % edges of the box
    for i = e, for j = e
        plot3(a, [0 Lp], [i i], [j j], 'k-', 'LineWidth', 0.4);
        plot3(a, [i i], [0 Lp], [j j], 'k-', 'LineWidth', 0.4);
        plot3(a, [i i], [j j], [0 Lp], 'k-', 'LineWidth', 0.4);
    end, end
    colormap(a, parula(256)); clim(a, [0 max(vals{k})]); shading(a, 'interp');
    daspect(a, [1 1 1]); view(a, [-35 25]); axis(a, 'tight'); axis(a, 'off');
    title(a, names{k}, 'FontWeight', 'normal', 'FontSize', S.label, 'Interpreter', 'latex');
    cb = colorbar(a, 'southoutside'); cb_style(cb, S, labels{k});
    cb.Units = 'centimeters'; cb.Position = [1.8 + (k-1)*7.3, 1.15, 4.2, 0.22];
end
export_fig(fig, fullfile(out, 'fastol_fields_forced_turb'), '-pdf', '-opengl', '-r600');
close(fig); fprintf('wrote %s.pdf\n', fullfile(out, 'fastol_fields_forced_turb'));
end

function cb_style(cb, S, label)
cb.TickLabelInterpreter = 'latex'; cb.FontSize = S.font; cb.LineWidth = 0.5;
cb.Label.Interpreter = 'latex'; cb.Label.String = label; cb.Label.FontSize = S.label;
end

function v = phs_local(p, f, q, k)
% Local cubic PHS + linear polynomial interpolation of the samples (p, f) at the points q,
% on the k nearest samples of each point (the same kind of basis as the paper's local reconstruction).
d = size(p, 2); nq = size(q, 1); v = zeros(nq, 1); chunk = 8000;
for s = 1:chunk:nq
    e = min(nq, s + chunk - 1); qq = q(s:e, :); m = e - s + 1;
    id = knnsearch(p, qq, 'K', k);
    X = reshape(p(id', :), k, m, d);                       % k x m x d
    X = permute(X, [1 3 2]) - permute(qq, [3 2 1]);        % k x d x m, centred at the query
    h = max(sqrt(sum(X.^2, 2)), [], 1); X = X./h;          % scale each stencil to unit radius
    D = sqrt(sum((permute(X, [1 4 2 3]) - permute(X, [4 1 2 3])).^2, 3));  % k x k x 1 x m
    Phi = reshape(D, k, k, m).^3;
    P = cat(2, ones(k, 1, m), X);                          % k x (d+1) x m
    A = cat(1, cat(2, Phi, P), cat(2, permute(P, [2 1 3]), zeros(d + 1, d + 1, m)));
    rhs = cat(1, reshape(f(id'), k, 1, m), zeros(d + 1, 1, m));
    c = pagemldivide(A, rhs);
    r3 = reshape(sqrt(sum(X.^2, 2)).^3, k, 1, m);           % kernel at the query (origin)
    v(s:e) = reshape(sum(c(1:k, 1, :).*r3, 1) + c(k + 1, 1, :), m, 1);
end
end

% ------------------------------------------------------------------ helpers
function [x, y] = grid2(xl, yl, nx, ny)
if nargin < 4, ny = round(nx*diff(yl)/diff(xl)); end
x = linspace(xl(1), xl(2), nx); y = linspace(yl(1), yl(2), ny);
end

function V = field2(p, v, x, y, inside)
% linear interpolation of scattered samples on the grid, NaN outside the hull or the domain
I = scatteredInterpolant(p(:, 1), p(:, 2), v(:), 'linear', 'none');
[X, Y] = meshgrid(x, y); V = I(X, Y);
if ~isempty(inside), V(~inside(X, Y)) = NaN; end
end

function draw(a, x, y, V, cl)
im = imagesc(a, x, y, V); set(im, 'AlphaData', ~isnan(V));
set(a, 'YDir', 'normal'); colormap(a, parula(256));
if cl(2) <= cl(1), cl = cl(1) + [-1 1]*max(1, abs(cl(1)))*1e-3; end   % constant field
clim(a, cl);
end

function circle(a, c, r)
t = linspace(0, 2*pi, 400);
plot(a, c(1) + r*cos(t), c(2) + r*sin(t), 'k-', 'LineWidth', 0.6);
end

function axes_xy(a, xl, yl, xt, yt, lab)
set(a, 'XLim', xl, 'YLim', yl, 'XTick', xt, 'YTick', yt);
if nargin > 5, set(a, 'XTickLabel', lab, 'YTickLabel', lab); end
xlabel(a, '$x$'); ylabel(a, '$y$');
finish_field_axes(a);
end

function finish_field_axes(a)
S = fastol_plot_style();
set(a, 'FontSize', S.font, 'LineWidth', S.axlw, 'Box', 'on', 'TickLabelInterpreter', 'latex', ...
    'Layer', 'top', 'TickDir', 'in', 'TickLength', [0.015 0.015]);
a.XLabel.FontSize = S.label; a.YLabel.FontSize = S.label;
a.Title.FontSize = S.label; a.Title.FontWeight = 'normal';
end

function [pp, idx] = periodic_images(p, Lp, w)
% samples plus their periodic images within distance w of the box [0, Lp]^3
pp = p; idx = (1:size(p, 1))';
for s = [-1 0 1]
    for t = [-1 0 1]
        for u = [-1 0 1]
            if s == 0 && t == 0 && u == 0, continue; end
            q = p + Lp*[s t u];
            keep = all(q > -w & q < Lp + w, 2);
            pp = [pp; q(keep, :)]; idx = [idx; find(keep)]; %#ok<AGROW>
        end
    end
end
end

function P = panels(axw, axh, mode)
% two field panels of size axw x axh (cm); colour bars below: one shared or one per panel
S = fastol_plot_style();
left = 1.25; gap = 1.55; title_h = 0.65;
cb_h = 0.22; cb_strip = 1.55; xlab = 1.05;
bottom = cb_strip + xlab;
Wf = left + 2*axw + gap + 0.5; Hf = bottom + axh + title_h;
fig = figure('Units', 'centimeters', 'Position', [2 2 Wf Hf], 'Color', 'w', 'Visible', 'off');
for k = 1:2
    ax(k) = axes(fig, 'Units', 'centimeters', 'Position', [left + (k-1)*(axw + gap), bottom, axw, axh]); %#ok<AGROW>
    hold(ax(k), 'on');
end
P.fig = fig; P.ax = ax;
P.clim_shared = @(V) [0 max(V(:))];
P.colorbar = @add_colorbar;
P.save = @save_fig;
    function add_colorbar(k, label)
        pos = ax(k).Position;
        if strcmp(mode, 'shared')
            x0 = ax(1).Position(1); x1 = ax(2).Position(1) + axw;
            w = 1.1*axw; xs = (x0 + x1)/2 - w/2;
        else
            w = 0.9*axw; xs = pos(1) + (axw - w)/2;
        end
        cb = colorbar(ax(k), 'southoutside');
        cb.Units = 'centimeters'; cb.Position = [xs, cb_strip - 0.55, w, cb_h];
        ax(k).Position = pos;
        cb.TickLabelInterpreter = 'latex'; cb.FontSize = S.font; cb.LineWidth = 0.5;
        cb.Label.Interpreter = 'latex'; cb.Label.String = label; cb.Label.FontSize = S.label;
        cl = clim(ax(k)); cb.Ticks = nice_ticks(cl);
    end
    function save_fig(file)
        for kk = 1:2, ax(kk).Position([3 4]) = [axw axh]; end
        drawnow;
        export_fig(fig, file, '-pdf', '-opengl', '-r600');
        close(fig); fprintf('wrote %s.pdf\n', file);
    end
end

function t = nice_ticks(cl)
r = diff(cl); st = 10^floor(log10(r/2));
for m = [1 2 2.5 5 10]
    if r/(m*st) <= 5, st = m*st; break; end
end
t = (ceil(cl(1)/st)*st):st:cl(2);
end
