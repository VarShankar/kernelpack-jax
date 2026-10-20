function fastol_field_figures(F, out)
% Benchmark field figures (appendix): first supplied simulation of each data set, two panels,
% colour fields from linear interpolation of the supplied samples (scatteredInterpolant,
% 'linear', no extrapolation) on a fine grid, masked by the exact domain (or, for species
% transport, by the supplied SU2 hexahedral mesh). Panels of the same quantity share a colour
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

% ---- species transport: (A) inlet velocity on z = 0, (B) speed at t = 0.5 on x = 0 (y expanded)
Sp = F.species; Me = F.species_mesh;
[x, y] = grid2([-0.01 0.01], [-0.01 0.01], 700);
inlet = [Me.markers.inlet_gas; Me.markers.inlet_air];
tri = [inlet(:, [1 2 3]); inlet(:, [1 3 4])];
Tin = triangulation(tri, Me.nodes(:, 1:2));
[X, Y] = meshgrid(x, y);
mA = reshape(~isnan(pointLocation(Tin, [X(:) Y(:)])), size(X));
A = field2(Sp.bc_points(:, 1:2), Sp.inlet(:), x, y, []); A(~mA) = NaN;
[z, y2] = grid2([0 0.26], [-0.01 0.01], 1500, 500);
[Z, Y2] = meshgrid(z, y2);
mB = section_mask(Me, z, y2);
I = scatteredInterpolant(Sp.points, sp(Sp.u), 'linear', 'none');
B = reshape(I(zeros(numel(Z), 1), Y2(:), Z(:)), size(Z)); B(~mB) = NaN;
P = panels(W, W, 'separate');
draw(P.ax(1), x, y, A, [min(A(:)) max(A(:))]); draw(P.ax(2), z, y2, B, [0 max(B(:))]);
axes_xy(P.ax(1), [-0.01 0.01], [-0.01 0.01], -0.01:0.01:0.01, -0.01:0.01:0.01);
axes_xy(P.ax(2), [0 0.26], [-0.01 0.01], 0:0.13:0.26, -0.01:0.01:0.01);
xlabel(P.ax(2), '$z$');
title(P.ax(1), '(A) Inlet velocity, $z=0$'); title(P.ax(2), '(B) Speed at $t=0.5$, $x=0$');
P.colorbar(1, 'inlet velocity'); P.colorbar(2, 'speed $\|\mathbf{u}\|$');
P.save(fullfile(out, 'fastol_fields_species_transport'));

% ---- forced turbulence on z = pi: (A) forcing magnitude, (B) speed at t = 0.5 (periodic images)
Ft = F.forced;
[pp, idx] = periodic_images(Ft.points, 2*pi, 3*2*pi/32);
[x, y] = grid2([0 2*pi], [0 2*pi], 700);
[X, Y] = meshgrid(x, y); Zq = pi*ones(numel(X), 1);
fm = sp(Ft.f); um = sp(Ft.u);
I = scatteredInterpolant(pp, fm(idx), 'linear', 'none'); A = reshape(I(X(:), Y(:), Zq), size(X));
I.Values = um(idx); B = reshape(I(X(:), Y(:), Zq), size(X));
P = panels(W, W, 'separate');
draw(P.ax(1), x, y, A, [0 max(A(:))]); draw(P.ax(2), x, y, B, [0 max(B(:))]);
for a = P.ax, axes_xy(a, [0 2*pi], [0 2*pi], [0 pi 2*pi], [0 pi 2*pi], twopi); end
title(P.ax(1), '(A) Forcing, $z=\pi$'); title(P.ax(2), '(B) Speed at $t=0.5$, $z=\pi$');
P.colorbar(1, 'forcing magnitude $\|\mathbf{f}\|$'); P.colorbar(2, 'speed $\|\mathbf{u}\|$');
P.save(fullfile(out, 'fastol_fields_forced_turb'));
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

function T = hex_to_tet(H, k)
% split each hexahedron (SU2/VTK ordering: bottom 1-4, top 5-8) into the five tetrahedra k
T = zeros(5*size(H, 1), 4);
for j = 1:5, T(j:5:end, :) = H(:, k(j, :)); end
end

function m = section_mask(Me, z, y)
% grid points (0, y_i, z_j) inside the hexahedral mesh. Each hexahedron crossing x = 0 is split
% into tetrahedra in both alternating five-tetrahedron ways (the mixer hexahedra are twisted, so
% one split alone leaves slivers), and grid points in a tetrahedron's (z, y) bounding box are
% tested barycentrically. Isolated pixels with at least 6 of 8 inside neighbours are then filled.
m = false(numel(y), numel(z));
k1 = [1 2 4 5; 2 3 4 7; 2 5 6 7; 4 5 7 8; 2 4 5 7];
k2 = [1 2 3 6; 1 3 4 8; 1 6 5 8; 3 6 7 8; 1 3 6 8];
for K = {k1, k2}
    Tt = hex_to_tet(Me.hex, K{1}); nt = size(Tt, 1);
    X = reshape(Me.nodes(Tt', 1), 4, nt); Tt = Tt(min(X) <= 0 & max(X) >= 0, :);
    for e = 1:size(Tt, 1)
        v = Me.nodes(Tt(e, :), :);                  % 4 x 3, columns x y z
        jz = find(z >= min(v(:, 3)) - 1e-12 & z <= max(v(:, 3)) + 1e-12);
        iy = find(y >= min(v(:, 2)) - 1e-12 & y <= max(v(:, 2)) + 1e-12);
        if isempty(jz) || isempty(iy), continue; end
        [J, I] = meshgrid(jz, iy);
        q = [zeros(numel(I), 1) y(I(:))' z(J(:))'];
        A = [v(2, :) - v(1, :); v(3, :) - v(1, :); v(4, :) - v(1, :)]';
        if abs(det(A)) < eps*max(abs(A(:)))^3, continue; end
        l = A \ (q - v(1, :))';
        in = all(l >= -1e-10, 1) & sum(l, 1) <= 1 + 1e-10;
        m(sub2ind(size(m), I(in), J(in))) = true;
    end
end
n = conv2(double(m), [1 1 1; 1 0 1; 1 1 1], 'same');
m = m | n >= 6;
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
