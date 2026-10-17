"""Treasure chest with an internal obstacle: a smaller, rotated copy of the chest.

The outer chest is described exactly as in treasure_chest_geometry.descriptors. Every piece of the
obstacle (back face, four sides, four shoulders, cap) is given only by point samples, a trimming loop
and normals, and enters the boundary model as a parametric partition of unity (ParametricSurfaceData).
The obstacle normals point into the obstacle, which is outward for the fluid domain.
"""
import numpy as np
import jax.numpy as jnp

from kernelpack.geometry.parametric_pu import ParametricSurfaceData
from treasure_chest_geometry import L, D, A, B, P, cap_normal, descriptors
from parametric_geometry_convergence_study import domed_square_patch_points

SCALE = 0.3
CENTER = np.array([0.6, 0.0, 0.0])
LOCAL_CENTER = np.array([0.9, 0.0, 0.0])        # middle of the chest's x-extent [0, L+D]


def _rotation(z_deg=35.0, x_deg=25.0):
    a, b = np.radians(z_deg), np.radians(x_deg)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return rz @ rx


R = _rotation()


def to_world(p):
    return CENTER + SCALE * (np.asarray(p) - LOCAL_CENTER) @ R.T


def to_local(x):
    return (np.asarray(x) - CENTER) @ R / SCALE + LOCAL_CENTER


def chest_inside(q):
    x, y, z = np.asarray(q).T
    rho = (np.abs(y) / A)**P + (np.abs(z) / B)**P
    return (x >= 0) & (np.abs(y) <= A) & (np.abs(z) <= B) & ((x <= L) | ((x <= L + D) & (((x - L) / D)**2 + rho <= 1)))


def inside(x):
    """Exact fluid domain: inside the outer chest and outside the obstacle."""
    return chest_inside(x) & ~chest_inside(to_local(x))


def _in_polygon(pts, poly):
    x, y = pts[:, 0], pts[:, 1]
    inside_ = np.zeros(len(pts), bool)
    for (x0, y0), (x1, y1) in zip(poly, np.roll(poly, -1, axis=0)):
        crosses = (y0 > y) != (y1 > y)
        xc = x0 + (y - y0) * (x1 - x0) / np.where(y1 != y0, y1 - y0, 1.0)
        inside_ ^= crosses & (x < xc)
    return inside_


def _densify(poly, spacing):
    out = []
    for p, q in zip(poly, np.roll(poly, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(q - p) / spacing)))
        out.append(p + np.linspace(0, 1, n, endpoint=False)[:, None] * (q - p))
    return np.vstack(out)


def _planar_piece(outline, embed, normal, spacing):
    """Samples of a planar face of the local chest, mapped to the obstacle; normals into the obstacle."""
    outline = np.asarray(outline, float)
    lo, hi = outline.min(0), outline.max(0)
    u, v = np.meshgrid(np.arange(lo[0], hi[0] + 1e-12, spacing), np.arange(lo[1], hi[1] + 1e-12, spacing))
    grid = np.column_stack([u.ravel(), v.ravel()])
    loop = _densify(outline, spacing)
    keep = _in_polygon(grid, outline)
    # interior grid points at least spacing/3 from the loop, so the data do not crowd the trimming curve
    d = np.min(np.linalg.norm(grid[keep][:, None, :] - loop[None, :, :], axis=2), axis=1)
    uv = np.vstack([loop, grid[keep][d > spacing / 3]])
    pos = to_world(embed(uv))
    nrm = np.broadcast_to(-(np.asarray(normal, float) @ R.T), pos.shape)
    return ParametricSurfaceData.from_geometry(pos, [to_world(embed(loop))], normals=np.array(nrm))


def obstacle_pieces(fit_resolution=41, spacing=0.06):
    pieces, labels = [], []
    rect = lambda a, b: np.array([[a[0], b[0]], [a[1], b[0]], [a[1], b[1]], [a[0], b[1]]])
    pieces.append(_planar_piece(rect((-A, A), (-B, B)), lambda uv: np.column_stack([0 * uv[:, 0], uv]), (-1, 0, 0), spacing))
    labels.append('obstacle_back')
    for s in (-1, 1):
        pieces.append(_planar_piece(rect((0, L), (-B, B)),
                                    lambda uv, s=s: np.column_stack([uv[:, 0], np.full(len(uv), s * A), uv[:, 1]]), (0, s, 0), spacing))
        labels.append(f'obstacle_side_y_{s}')
        pieces.append(_planar_piece(rect((0, L), (-A, A)),
                                    lambda uv, s=s: np.column_stack([uv[:, 0], uv[:, 1], np.full(len(uv), s * B)]), (0, 0, s), spacing))
        labels.append(f'obstacle_side_z_{s}')
    split = 2**(-1 / P)
    t = np.linspace(0, split, max(17, round(257 * fit_resolution / 21)))
    rim_quarter = np.vstack((np.column_stack((A * (1 - t**P)**(1 / P), B * t)),
                             np.column_stack((A * t[::-1], B * (1 - t[::-1]**P)**(1 / P)))[1:]))
    for sy in (-1, 1):
        for sz in (-1, 1):
            shoulder = np.vstack(([A, 0], [A, B], [0, B], rim_quarter[::-1][1:-1])) * np.array([sy, sz])
            if sy * sz < 0:
                shoulder = shoulder[::-1]
            pieces.append(_planar_piece(shoulder, lambda uv: np.column_stack([np.full(len(uv), L), uv]), (1, 0, 0), spacing))
            labels.append(f'obstacle_shoulder_{sy}_{sz}')
    # cap: the same samples as the outer cap (treasure_chest_geometry.descriptors), mapped to the obstacle
    radial = max(3, round(9 * fit_resolution / 21))
    fit_t = np.linspace(0, split, fit_resolution)
    fit_rim = np.vstack((np.column_stack((A * fit_t, B * (1 - fit_t**P)**(1 / P))),
                         np.column_stack((A * (1 - fit_t**P)**(1 / P), B * fit_t))))
    scales = np.unique(np.r_[np.linspace(0, .9, radial), (1 - np.linspace(0, np.sqrt(1 - .9**P), radial)**2)**(1 / P)])
    xyz = []
    for sy in (-1, 1):
        for sz in (-1, 1):
            s = np.repeat(scales, len(fit_rim))
            yz = np.tile(fit_rim, (len(scales), 1)) * s[:, None]
            xyz.append(np.column_stack((L + D * np.sqrt(np.maximum(0, 1 - s**P)), sy * yz[:, 0], sz * yz[:, 1])))
    xyz = np.unique(np.round(np.vstack(xyz), 14), axis=0)
    rims = np.unique(np.vstack([np.column_stack((np.full(len(rim_quarter), L), sy * rim_quarter[:, 0], sz * rim_quarter[:, 1]))
                                for sy in (-1, 1) for sz in (-1, 1)]), axis=0)
    rims = rims[np.argsort(np.arctan2(rims[:, 2] / B, rims[:, 1] / A))]
    checks = domed_square_patch_points('dome', 1537)
    pieces.append(ParametricSurfaceData.from_geometry(
        to_world(xyz), [to_world(rims)], normals=-(cap_normal(xyz) @ R.T),
        validation_positions=to_world(checks), validation_normals=-(cap_normal(checks) @ R.T)))
    labels.append('obstacle_cap')
    return pieces, labels


def descriptors_with_obstacle(h, fit_resolution=41):
    desc, labels = descriptors(h, fit_resolution=fit_resolution)
    pieces, obstacle_labels = obstacle_pieces(fit_resolution)
    n = len(pieces)
    desc = dict(desc)
    desc['bdry_desc'] = list(desc['bdry_desc']) + pieces
    desc['is_bdry_curve'] = list(desc['is_bdry_curve']) + [False] * n
    desc['is_smooth_curve'] = list(desc['is_smooth_curve']) + [False] * n
    desc['constants'] = list(desc['constants']) + [0.] * n
    desc['constant_dim'] = list(desc['constant_dim']) + [-1] * n
    desc['flip_normal'] = list(desc['flip_normal']) + [False] * n
    return desc, labels + obstacle_labels
