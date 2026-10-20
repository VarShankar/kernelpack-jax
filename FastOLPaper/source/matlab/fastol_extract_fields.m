function F = fastol_extract_fields(cacheFile, root)
% F = fastol_extract_fields(cacheFile) reads the FIRST supplied simulation (row 1) of each
% data set used in the paper's benchmark figures from E:\fluid_datasets (read only) and saves
% the small result to cacheFile. Later calls return the cache. The data files are MAT v7
% (not 7.3), so each variable is loaded whole and row 1 is kept; this takes a few minutes.
% Files and fields follow scripts/fastol_gpu_campaign.py (CASES) of the FastOL worktree.
if nargin < 2
    root = getenv('FASTOL_DATA_ROOT');                  % benchmark data sets of Sharma et al.
    if isempty(root), root = 'E:\fluid_datasets'; end
end
if exist(cacheFile, 'file'), F = load(cacheFile); return; end
first = @(a) reshape(a(1, :, :), size(a, 2), []);      % row 1 as points x components

F = struct();
f = fullfile(root, 'flow_cylinder', 'data_shedding.mat');
F.cylinder = struct('points', getv(f, 'points'), 'u0', first(getv(f, 'init_velocity')), ...
    'u', first(getv(f, 'velocity')));
f = fullfile(root, 'taylor_green', 'data_exact_matt.mat');
F.taylor_green = struct('points', getv(f, 'points'), 'u0', first(getv(f, 'init_velocity')), ...
    'u', first(getv(f, 'velocity')));
f = fullfile(root, 'lid_cavity_flow', 'data.mat');
F.lid_cavity = struct('points', getv(f, 'points'), 'u0', first(getv(f, 'init_velocity')), ...
    'u', first(getv(f, 'velocity')));
f = fullfile(root, 'backward_facing_step', 'data.mat');
F.backward_step = struct('points', getv(f, 'points'), 'bc_points', getv(f, 'bc_points'), ...
    'inlet', first(getv(f, 'init_velocity')), 'u', first(getv(f, 'velocity')));
f = fullfile(root, 'buoyancy_cavity_flow', 'data.mat');
F.buoyancy = struct('points', getv(f, 'points'), 'T0', first(getv(f, 'init_temperature')), ...
    'u', first(getv(f, 'velocity')));
f = fullfile(root, 'merge_vortices_easier', 'data.mat');
F.merge = struct('points', getv(f, 'points'), 'w0', first(getv(f, 'init_vorticity')), ...
    'u', first(getv(f, 'velocity')));
f = fullfile(root, 'species_transport', 'data.mat');
F.species = struct('points', getv(f, 'points'), 'bc_points', getv(f, 'bc_points'), ...
    'inlet', first(getv(f, 'init_velocity')), 'u', first(getv(f, 'velocity')));
F.species_mesh = read_su2(fullfile(root, 'species_transport', 'fluid.su2'));
f = fullfile(root, 'forced_turb', 'data.mat');
F.forced = struct('points', getv(f, 'points'), 'f', first(getv(f, 'forcing')), ...
    'u', first(getv(f, 'velocity')));
F.source = root;
F.created = char(datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss'));
save(cacheFile, '-struct', 'F', '-v7');
end

function v = getv(file, name)
fprintf('reading %s from %s\n', name, file);
s = load(file, name); v = s.(name);
end

function M = read_su2(file)
% SU2 hexahedral volume mesh: node coordinates, hexahedra (type 12), boundary quads per marker.
L = readlines(file);
ne = sscanf(L(startsWith(L, 'NELEM=')), 'NELEM= %d'); ne = ne(1);
i0 = find(startsWith(L, 'NELEM='), 1);
E = sscanf(strjoin(L(i0+1:i0+ne), ' '), '%f');
E = reshape(E, 10, [])';                                   % type, 8 nodes, index
assert(all(E(:, 1) == 12), 'expected hexahedra only');
M.hex = E(:, 2:9) + 1;
ip = find(startsWith(L, 'NPOIN='));
np = sscanf(L(ip), 'NPOIN= %d');
P = sscanf(strjoin(L(ip+1:ip+np), ' '), '%f');
P = reshape(P, 4, [])';
M.nodes = P(:, 1:3);
it = find(startsWith(L, 'MARKER_TAG='));
for k = 1:numel(it)
    tag = strtrim(extractAfter(L(it(k)), 'MARKER_TAG='));
    nm = sscanf(L(it(k)+1), 'MARKER_ELEMS= %d');
    Q = sscanf(strjoin(L(it(k)+2:it(k)+1+nm), ' '), '%f');
    Q = reshape(Q, 5, [])';                                % type 9 (quad), 4 nodes
    assert(all(Q(:, 1) == 9));
    M.markers.(matlab.lang.makeValidName(tag)) = Q(:, 2:5) + 1;
end
end
