function S = fastol_plot_style()
% One style for every FastOL paper panel, matching the KernelPack-JAX paper
% (scripts/matlab/paper_plots.m there): 9 pt LaTeX tick labels, 10 pt axis labels
% and titles, thin lines, filled/open markers, export_fig vector PDF.
% Every panel of every figure has the same axes box (S.axw x S.axh cm), and figures
% are placed in LaTeX at their natural size, so the printed fonts are 9/10 pt.
S.font = 9;                 % tick labels and legend
S.label = 10;               % axis labels and panel titles
S.axw = 4.0;                % axes box width (cm)
S.axh = 3.2;                % axes box height (cm)
S.left = 1.55;              % room for y tick labels and y label (cm)
S.gapx = 1.25;              % between columns: y tick labels of the next panel
S.gapx_label = 1.65;        % between columns when every panel carries a y label
S.bottom = 1.15;            % x tick labels and x label (cm)
S.gapy = 1.30;              % between rows: x tick labels and next title
S.title = 0.55;             % panel title strip (cm)
S.lw = 0.9;                 % data lines
S.guidelw = 1.1;            % dotted guides
S.ms = 4.5;                 % marker size
S.axlw = 0.7;               % axes box
S.grey = [0.45 0.45 0.45];

% Orders (requested smooth-field order p), as in KernelPack-JAX: colour and marker.
S.order_color = containers.Map({2, 4, 6}, {[0 0.35 0.75], [0.85 0.40 0], [0 0.55 0.45]});
S.order_marker = containers.Map({2, 4, 6}, {'o', 's', '^'});

% Operator models: colour = kernel (default blue, tuned orange),
% line = augmentation (solid none, dashed affine), filled markers without augmentation, open with it.
S.models = {'fixed_no_poly', 'trainable_no_poly', 'fixed_affine', 'trainable_affine'};
S.model_name = {'Fixed shape', 'Fitted shape', 'Fixed shape + affine', 'Fitted shape + affine'};
S.model_color = {[0 0.35 0.75], [0.85 0.40 0], [0 0.35 0.75], [0.85 0.40 0]};
S.model_line = {'-', '-', '--', '--'};
S.model_marker = {'o', 's', '^', 'd'};
S.model_filled = [true true false false];

% Spatial kernels: line style and marker fill.
S.kernel = {'phs', 'matern'};
S.kernel_name = {'PHS', 'Mat\''ern'};
S.kernel_line = {'-', '--'};
S.kernel_filled = [true false];
end
