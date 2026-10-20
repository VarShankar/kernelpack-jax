"""Ram's local divergence-free kernel + polynomial spatial decoder.

Geometry/stencil selection uses the shared GPU selector. Basis construction, augmented
solves, temporal fits, and evaluation matrices are JAX computations on the
selected device. Spatial blocks bound training-coefficient memory.
"""

from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from collections import OrderedDict
import math
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
import numpy as np
from jax.custom_batching import custom_vmap
from kernelpack.divfree import prepare_divfree_local_system
from kernelpack.divfree.core import divfree_polynomial_row_basis
from kernelpack.divfree.matern import prepare_divfree_matern_local_system
from kernelpack.domain.stencils import select_stencils

# Node count from which non-periodic stencils use the exact cell-blocked search (grid_knn).
_GRID_KNN_NODES = 4096


@partial(jax.jit, static_argnames=("order", "exponent", "spatial_kernel", "matern_num_modes"))
def _local_system(nodes, query, *, order, exponent, spatial_kernel="phs",
                  matern_base_length_factor=16., matern_num_modes=5, matern_gamma=8 / 3,
                  coefficient_basis=None):
    if spatial_kernel == "matern_multiscale":
        return prepare_divfree_matern_local_system(nodes, query, poly_degree=order,
            base_length_factor=matern_base_length_factor, num_modes=matern_num_modes, gamma=matern_gamma,
            coefficient_basis=coefficient_basis)
    return prepare_divfree_local_system(nodes,query,poly_degree=order,phs_degree=exponent,
        coefficient_basis=coefficient_basis)


@partial(jax.jit, static_argnames=("order", "exponent", "spatial_kernel", "matern_num_modes"))
def _batch_systems(nodes, queries, *, order, exponent, spatial_kernel="phs",
                   matern_base_length_factor=16., matern_num_modes=5, matern_gamma=8 / 3,
                   coefficient_basis=None):
    return jax.vmap(
        lambda p, q: _local_system(p, q[None], order=order, exponent=exponent,
            spatial_kernel=spatial_kernel, matern_base_length_factor=matern_base_length_factor,
            matern_num_modes=matern_num_modes, matern_gamma=matern_gamma,coefficient_basis=coefficient_basis)
    )(nodes, queries)


@jax.jit
def _temporal_lagrange_weights(nodes, queries):
    """MATLAB's translated/scaled cardinal products, without a Vandermonde solve."""
    center = jnp.mean(nodes)
    scale = jnp.max(jnp.abs(nodes - center))
    scale = jnp.where(scale > 0, scale, 1.)
    t, q = (nodes - center) / scale, (queries - center) / scale
    diagonal = jnp.eye(len(t), dtype=bool)
    denominator = jnp.where(diagonal, 1., t[:, None] - t[None, :])
    factors = (q[:, None, None] - t[None, None, :]) / denominator[None]
    return jnp.prod(jnp.where(diagonal[None], 1., factors), axis=-1)


@jax.jit
def _solve_local(systems, rhs):
    return jnp.linalg.solve(systems, rhs)


@jax.jit
def _factor_local(systems):
    return jax.vmap(jl.lu_factor)(systems)


@jax.jit
def _solve_factored_local(lu, pivots, rhs):
    return jax.vmap(lambda a, p, b: jl.lu_solve((a, p), b))(lu, pivots, rhs)


@jax.jit
def _solve_checked_local_impl(systems, lu, pivots, rhs):
    coefficients = _solve_factored_local(lu, pivots, rhs)
    def repair(_):
        replacement = jnp.linalg.pinv(systems) @ rhs
        finite = jnp.all(jnp.isfinite(coefficients), axis=(1, 2))
        return jnp.where(finite[:, None, None], coefficients, replacement)
    # Keep this condition outside vmap: a vmapped conditional can execute
    # both branches, paying for every pseudoinverse even on healthy blocks.
    coefficients = jax.lax.cond(
        jnp.all(jnp.isfinite(coefficients)), lambda _: coefficients,
        repair, operand=None,
    )
    finite_rhs = jnp.all(jnp.isfinite(rhs), axis=(1, 2))
    return jnp.where(finite_rhs[:, None, None], coefficients, jnp.nan)


_batched_checked_local = custom_vmap(_solve_checked_local_impl)


@_batched_checked_local.def_vmap
def _checked_local_batch(axis_size, in_batched, systems, lu, pivots, rhs):
    # Pack extra query batches into RHS columns before the finite check.
    # Otherwise vmap turns cond into select and executes every pseudoinverse.
    if not any(in_batched[:3]):
        if not in_batched[3]:
            return _batched_checked_local(systems, lu, pivots, rhs), False
        batch, stencils, width, columns = rhs.shape
        packed = jnp.transpose(rhs, (1, 2, 0, 3)).reshape(stencils, width, batch * columns)
        solved = _batched_checked_local(systems, lu, pivots, packed)
        return jnp.transpose(solved.reshape(stencils, width, batch, columns), (2, 0, 1, 3)), True
    values = []
    for value, batched in zip((systems, lu, pivots, rhs), in_batched):
        value = value if batched else jnp.broadcast_to(value, (axis_size,) + value.shape)
        values.append(value.reshape((-1,) + value.shape[2:]))
    solved = _batched_checked_local(*values)
    return solved.reshape((axis_size, -1) + solved.shape[1:]), True


@jax.custom_jvp
def _solve_checked_local(systems, lu, pivots, rhs):
    return _batched_checked_local(systems, lu, pivots, rhs)


@_solve_checked_local.defjvp
def _checked_local_jvp(primals, tangents):
    # custom_vmap alone does not supply differentiation rules.
    return jax.jvp(_solve_checked_local_impl, primals, tangents)


@jax.jit
def _prepared_coefficients(velocity, indices, systems, lu, pivots, temporal):
    """Gather, solve batched stencils and apply the fixed temporal map."""
    selected = velocity[:, indices]
    nt = velocity.shape[2] if velocity.ndim == 4 else 1
    if velocity.ndim == 3:
        selected = selected[:, :, :, None, :]
    count, dimension = indices.shape[1], velocity.shape[-1]
    rhs = jnp.transpose(selected, (1, 4, 2, 3, 0)).reshape(
        indices.shape[0], count * dimension, nt * velocity.shape[0])
    rhs = jnp.pad(rhs, ((0, 0), (0, systems.shape[1] - rhs.shape[1]), (0, 0))).astype(systems.dtype)
    coefficients = _solve_checked_local(systems, lu, pivots, rhs)
    coefficients = jnp.transpose(
        coefficients.reshape(indices.shape[0], systems.shape[1], nt, velocity.shape[0]),
        (0, 3, 2, 1),
    )
    return jnp.einsum("tu,squc->sqtc", temporal, coefficients,
                      precision=jax.lax.Precision.HIGHEST)


@dataclass
class SpatialBlock:
    coefficients: jax.Array  # stencil x train x coefficient
    evaluation: jax.Array  # stencil x (time*component) x coefficient
    nodes: jax.Array
    indices: jax.Array

    def reconstruct(self, predicted):
        """predicted: stencil x query-example x coefficient."""
        return jnp.einsum(
            "sqc,sdc->qsd",
            predicted,
            self.evaluation,
            precision=jax.lax.Precision.HIGHEST,
        )


class LocalDFPHSDecoder:
    """Local PHS or multiscale Matérn reconstruction with an unchanged stencil rule.

    ``spatial_kernel='matern_multiscale'`` uses center-normalized physical
    Matérn derivatives at dyadic lengths. ``temporal_interpolation=
    'cubic_polynomial'`` keeps four saved spatial coefficient blocks and uses
    cubic cardinal time weights, without a temporal kernel solve.
    """
    def __init__(
        self,
        points,
        *,
        order=6,
        stencil_size=None,
        use_periodic=False,
        periodic_dims=None,
        domain_bounds=None,
        boundary_fraction=0.15,
        time_points=None,
        use_time_kernel=False,
        temporal_interpolation=None,
        spatial_kernel="phs",
        matern_base_length_factor=16.,
        matern_num_modes=5,
        matern_gamma=8 / 3,
        geometry_cache_bytes=512 * 1024 * 1024,
    ):
        if not jax.config.x64_enabled:
            raise ValueError("FastOL spatial decoding requires jax_enable_x64=True")
        self.points = jnp.asarray(points, dtype=jnp.float64)
        if self.points.device.platform!="gpu":
            raise ValueError("FastOL spatial decoding requires GPU arrays")
        if (
            self.points.ndim != 2
            or self.points.shape[1] not in (2, 3)
            or not len(self.points)
            or not bool(jnp.all(jnp.isfinite(self.points)))
        ):
            raise ValueError("points must be a nonempty finite (n,2) or (n,3) matrix")
        if int(order) != order or order < 1 or not 0 <= boundary_fraction <= 0.5:
            raise ValueError(
                "order must be positive integer and boundary_fraction in [0,.5]"
            )
        self.order = int(order)
        if spatial_kernel not in ("phs", "matern_multiscale"):
            raise ValueError("spatial_kernel must be phs or matern_multiscale")
        if (not math.isfinite(matern_base_length_factor) or matern_base_length_factor <= 0
                or not math.isfinite(matern_gamma) or int(matern_num_modes) != matern_num_modes
                or matern_num_modes < 1):
            raise ValueError("Matérn scales require a positive base length factor, positive integer mode count, and finite gamma")
        self.spatial_kernel = spatial_kernel
        self.matern_base_length_factor = float(matern_base_length_factor)
        self.matern_num_modes = int(matern_num_modes)
        self.matern_gamma = float(matern_gamma)
        self.dimension = self.points.shape[1]
        self.polynomial_row_basis=divfree_polynomial_row_basis(self.dimension,self.order,
            dtype=self.points.dtype,device=self.points.device)
        self.exponent = max(3, min(self.order - (self.order + 1) % 2, 7))
        count = self.dimension * math.comb(
            self.order + self.dimension, self.dimension
        ) - math.comb(self.order + self.dimension - 1, self.dimension)
        if stencil_size is not None and (
            int(stencil_size) != stencil_size or stencil_size < 1
        ):
            raise ValueError("stencil_size must be a positive integer")
        self.stencil_size = min(
            len(self.points),
            (
                int(stencil_size)
                if stencil_size is not None
                else 2 * math.ceil(count / self.dimension) + 1
            ),
        )
        bounds = (
            jnp.asarray(domain_bounds,dtype=self.points.dtype,device=self.points.device)
            if domain_bounds is not None
            else jnp.stack([self.points.min(0), self.points.max(0)])
        )
        if (
            bounds.shape != (2, self.dimension)
            or not bool(jnp.all(jnp.isfinite(bounds)))
            or bool(jnp.any(bounds[1] < bounds[0]))
            or bool(jnp.any(self.points < bounds[0]))
            or bool(jnp.any(self.points > bounds[1]))
        ):
            raise ValueError("invalid domain bounds")
        self.minimum, self.maximum = bounds
        mask = (
            jnp.ones(self.dimension, dtype=bool)
            if periodic_dims is None
            else jnp.asarray(periodic_dims, dtype=bool,device=self.points.device)
        )
        if mask.shape != (self.dimension,):
            raise ValueError("periodic_dims must match spatial dimension")
        self.use_periodic=bool(use_periodic)
        self.periodic = (
            mask & self.use_periodic & ((self.maximum - self.minimum) > 1e-12)
        )
        self.boundary_fraction = boundary_fraction
        self.time_points = (
            None if time_points is None else jnp.asarray(time_points, dtype=self.points.dtype,device=self.points.device)
        )
        temporal_mode = ("phs" if use_time_kernel else "none") if temporal_interpolation is None else temporal_interpolation
        if temporal_mode not in ("none", "phs", "polynomial", "cubic_polynomial"):
            raise ValueError("temporal_interpolation must be none, phs, polynomial, or cubic_polynomial")
        self.temporal_interpolation = temporal_mode
        # The explicit interpolation mode supersedes the legacy dataset flag.
        self.use_time_kernel = temporal_mode == "phs"
        if int(geometry_cache_bytes) != geometry_cache_bytes or geometry_cache_bytes < 0:
            raise ValueError("geometry_cache_bytes must be a nonnegative integer")
        self.geometry_cache_bytes = int(geometry_cache_bytes)
        self._geometry_cache = OrderedDict()
        self._geometry_cache_used = 0
        self._temporal_cache = {}

    def clear_geometry_cache(self):
        """Release cached local systems, LU factors, and evaluation maps."""
        self._geometry_cache.clear()
        self._geometry_cache_used = 0
        self._temporal_cache.clear()

    def prepare_temporal_transform(self, nt, dtype, *, query_times=None):
        """Prepare the shared saved-time interpolation map, without output RHS."""
        points = self.time_points
        if self.temporal_interpolation != "none" and (points is None or points.shape != (nt,)
                                     or not bool(jnp.all(jnp.isfinite(points)))):
            raise ValueError("time_points must match the velocity time axis")
        polynomial = self.temporal_interpolation in ("polynomial", "cubic_polynomial")
        if polynomial and bool(jnp.any(jnp.sort(points)[1:] == jnp.sort(points)[:-1])):
            raise ValueError("Polynomial temporal interpolation requires distinct time_points")
        if self.temporal_interpolation == "cubic_polynomial" and nt != 4:
            raise ValueError("cubic_polynomial requires the four saved time_points")
        if query_times is not None:
            query_times = jnp.asarray(query_times, dtype=dtype, device=self.points.device)
            if query_times.ndim != 1 or not bool(jnp.all(jnp.isfinite(query_times))):
                raise ValueError("query_times must be a finite vector")
            if not polynomial:
                raise ValueError("Intermediate query_times require polynomial temporal interpolation")
        key = (nt, str(dtype), self.temporal_interpolation,
               id(points), id(query_times))
        if key in self._temporal_cache:
            return self._temporal_cache[key][0]
        transform = jnp.eye(nt, dtype=dtype)
        if polynomial:
            t = jnp.asarray(points, dtype=dtype)
            transform = _temporal_lagrange_weights(t, t if query_times is None else query_times)
        elif self.use_time_kernel:
            t = jnp.asarray(points, dtype=dtype)
            scale = jnp.max(jnp.abs(t - jnp.mean(t)))
            tau = (t - jnp.mean(t)) / jnp.where(scale > 0, scale, 1.0)
            p = jnp.stack([tau**k for k in range(min(1, nt - 1) + 1)], axis=1)
            phi = jnp.abs(t[:, None] - t[None, :]) ** 3
            system = jnp.block([[phi, p], [p.T, jnp.zeros((p.shape[1], p.shape[1]), dtype=dtype)]])
            transform = jnp.concatenate([phi, p], axis=1) @ jnp.linalg.solve(
                system, jnp.pad(transform, ((0, p.shape[1]), (0, 0))))
        self._temporal_cache[key] = (transform, points, query_times)
        return transform

    def _prepare_block(self, queries, velocity, *, center_indices=None):
        queries=jnp.asarray(queries,dtype=self.points.dtype,device=self.points.device)
        if center_indices is not None:
            center_indices=jnp.asarray(center_indices,device=self.points.device)
        # Retain immutable array identities instead of downloading coordinates
        # to hash their contents. Cache values keep those arrays alive.
        key = (queries.shape, id(queries), str(velocity.dtype),
               str(velocity.device), id(self.points),id(self.minimum),id(self.maximum),id(self.periodic),self.order,
               self.exponent, self.stencil_size, self.boundary_fraction,self.use_periodic,id(center_indices),
               self.spatial_kernel, self.matern_base_length_factor, self.matern_num_modes, self.matern_gamma)
        if key in self._geometry_cache:
            entry, size, keepalive = self._geometry_cache.pop(key)
            self._geometry_cache[key] = (entry, size, keepalive)
            return entry
        if self.uses_grid_knn:
            # Large non-periodic node sets: the dense batched search holds one distance row per
            # node and query, so only a few queries fit per batch. The exact cell-blocked search
            # returns the same neighbours (ties by lower index) with O(m k) work. Nodal queries
            # (the default) reuse one search over all nodes, computed once.
            from kernelpack.accelerators.jax_spatial import grid_knn
            nodal = (center_indices is not None and bool(jnp.all(center_indices >= 0))
                     and bool(jnp.all(self.points[jnp.clip(center_indices, 0, len(self.points) - 1)] == queries)))
            if nodal:
                if getattr(self, "_nodal_knn", None) is None:
                    host = np.asarray(jax.device_get(self.points))
                    ids, dist = grid_knn(host, host, self.stencil_size)
                    self._nodal_knn = (jnp.asarray(ids.astype(np.int32), device=self.points.device),
                                       jnp.asarray(np.all(np.isfinite(dist), axis=1), device=self.points.device))
                indices = self._nodal_knn[0][center_indices]
                valid = self._nodal_knn[1][center_indices]
            else:
                ids, dist = grid_knn(np.asarray(jax.device_get(queries)), np.asarray(jax.device_get(self.points)),
                                     self.stencil_size)
                indices = jnp.asarray(ids.astype(np.int32), device=self.points.device)
                valid = jnp.asarray(np.all(np.isfinite(dist), axis=1), device=self.points.device)
                if center_indices is not None:
                    safe = jnp.clip(center_indices, 0, len(self.points) - 1)
                    valid = valid & ((center_indices == -1)
                                     | ((center_indices >= 0) & jnp.all(self.points[safe] == queries, axis=1)))
            nodes = self.points[indices]
        else:
            indices,nodes,valid=select_stencils(self.points,queries,k=self.stencil_size,
                periodic=self.periodic,lengths=self.maximum-self.minimum,
                center_indices=center_indices,use_periodic=self.use_periodic)
        if not bool(jnp.all(valid)):
            raise ValueError("Invalid/ambiguous stencil center or insufficient distinct locations; supply center_indices for duplicate nodes")
        # Spatial geometry, rank reduction and factors consistently use the
        # decoder's float64 precision, even for float32 operator training data.
        systems, evaluation, rank_ratio = _batch_systems(
            nodes, queries,
            order=self.order, exponent=self.exponent, spatial_kernel=self.spatial_kernel,
            matern_base_length_factor=self.matern_base_length_factor,
            matern_num_modes=self.matern_num_modes, matern_gamma=self.matern_gamma,
            coefficient_basis=self.polynomial_row_basis,
        )
        if not bool(jnp.all(jnp.isfinite(rank_ratio) & (rank_ratio > 1e-12))):
            raise ValueError(
                "A spatial stencil does not resolve the required divergence-free polynomial rank"
            )
        lu, pivots = _factor_local(systems)
        entry = (nodes, indices, systems, evaluation, lu, pivots)
        size = sum(value.nbytes for value in entry) + queries.nbytes + (0 if center_indices is None else center_indices.nbytes) + sum(
            value.nbytes for value in (self.points,self.minimum,self.maximum,self.periodic))
        if size <= self.geometry_cache_bytes:
            while self._geometry_cache and self._geometry_cache_used + size > self.geometry_cache_bytes:
                _, (_, old_size, _) = self._geometry_cache.popitem(last=False)
                self._geometry_cache_used -= old_size
            self._geometry_cache[key] = (entry,size,(queries,self.points,self.minimum,self.maximum,self.periodic,center_indices))
            self._geometry_cache_used += size
        return entry

    @property
    def uses_grid_knn(self):
        """Whether stencil selection uses the exact cell-blocked search (large, non-periodic node sets)."""
        return (not self.use_periodic) and len(self.points) >= _GRID_KNN_NODES

    def select_stencil(self, query=None, *, center_index=None):
        """Select by coordinates or explicit zero-based nodal identity.

        select_stencil(center_index=i) is the index-based MATLAB equivalent.
        A supplied query and index must refer to the same original node.
        """
        if query is None:
            if center_index is None:
                raise ValueError("A query or center_index is required")
            index=jnp.asarray(center_index,device=self.points.device)
            if index.ndim!=0 or not jnp.issubdtype(index.dtype,jnp.integer) or not bool((index>=0)&(index<len(self.points))):
                raise ValueError("center_index must be a valid integer node index")
            query=self.points[index]
        q=jnp.asarray(query,dtype=self.points.dtype,device=self.points.device).reshape(1,self.dimension)
        centers=None if center_index is None else jnp.asarray(center_index,device=self.points.device).reshape(1)
        ids,nodes,valid=select_stencils(self.points,q,k=self.stencil_size,
            periodic=self.periodic,lengths=self.maximum-self.minimum,
            center_indices=centers,use_periodic=self.use_periodic)
        if not bool(jnp.all(valid)):
            raise ValueError("Invalid/ambiguous stencil center or insufficient distinct locations")
        return nodes[0],ids[0]

    def fit_block(self, velocity, queries, *, validate=True, center_indices=None, query_times=None):
        """Fit all training fields on a bounded block of assigned stencils."""
        v = jnp.asarray(velocity,device=self.points.device)
        q = jnp.asarray(queries, dtype=self.points.dtype,device=self.points.device).reshape(-1, self.dimension)
        if not len(q) or not bool(jnp.all(jnp.isfinite(q))):
            raise ValueError("queries must be nonempty and finite")
        if (
            v.ndim not in (3, 4)
            or v.shape[1] != len(self.points)
            or v.shape[-1] != self.dimension
        ):
            raise ValueError(
                "velocity must have shape (train,points,dim) or (train,points,time,dim)"
            )
        nodes, indices, systems, evaluation, lu, pivots = self._prepare_block(q,v,center_indices=center_indices)
        selected = v[:, indices]
        nt = v.shape[2] if v.ndim == 4 else 1
        if v.ndim == 3:
            selected = selected[:, :, :, None, :]
        # component-major spatial rows; time/sample RHS columns.
        rhs = jnp.transpose(selected, (1, 4, 2, 3, 0)).reshape(
            len(q), self.stencil_size * self.dimension, nt * v.shape[0]
        )
        if validate and not bool(jnp.all(jnp.isfinite(rhs))):
            raise ValueError("Nonfinite velocity on spatial stencil")
        rhs = jnp.pad(rhs, ((0, 0), (0, systems.shape[1] - rhs.shape[1]), (0, 0))).astype(systems.dtype)
        coefficients = _solve_checked_local(systems, lu, pivots, rhs)
        if validate and not bool(jnp.all(jnp.isfinite(coefficients))):
            raise RuntimeError("Nonfinite local divergence-free spatial solution")
        width = systems.shape[1]
        coeff = jnp.transpose(
            coefficients.reshape(len(q), width, nt, v.shape[0]), (0, 3, 2, 1)
        )
        temporal_eval = self.prepare_temporal_transform(nt, systems.dtype, query_times=query_times)
        if self.use_time_kernel:
            if (
                self.time_points is None
                or self.time_points.shape != (nt,)
                or not bool(jnp.all(jnp.isfinite(self.time_points)))
            ):
                raise ValueError("time_points must match the velocity time axis")
            t = jnp.asarray(self.time_points, dtype=systems.dtype)
            scale = jnp.max(jnp.abs(t - jnp.mean(t)))
            scale = jnp.where(scale > 0, scale, 1.0)
            tau = (t - jnp.mean(t)) / scale
            p = jnp.stack([tau**k for k in range(min(1, nt - 1) + 1)], axis=1)
            phi = jnp.abs(t[:, None] - t[None, :]) ** 3
            system = jnp.block(
                [[phi, p], [p.T, jnp.zeros((p.shape[1], p.shape[1]), dtype=systems.dtype)]]
            )
            time_rhs = jnp.transpose(coeff, (2, 0, 1, 3)).reshape(nt, -1)
            time_coeff = jnp.linalg.solve(
                system, jnp.pad(time_rhs, ((0, p.shape[1]), (0, 0)))
            )
            coeff = jnp.transpose(
                time_coeff.reshape(system.shape[0], len(q), v.shape[0], width),
                (1, 2, 0, 3),
            )
            temporal_eval = jnp.concatenate([phi, p], axis=1)
        evaluations = jax.vmap(lambda e: jnp.kron(temporal_eval, e))(evaluation)
        return SpatialBlock(
            coeff.reshape(len(q), v.shape[0], -1), evaluations, nodes, indices
        )


@jax.jit
def _local_evaluation_weights(systems,lu,pivots,evaluation):
    """Compute E A^{-1} once using adjoint solves, without forming inverses."""
    rhs=jnp.swapaxes(evaluation,-1,-2)
    solved=jax.vmap(lambda a,p,b:jl.lu_solve((a,p),b,trans=1))(lu,pivots,rhs)
    def repair(_):
        fallback=jnp.swapaxes(jnp.linalg.pinv(systems),-1,-2)@rhs
        finite=jnp.all(jnp.isfinite(solved),axis=(-1,-2))
        return jnp.where(finite[:,None,None],solved,fallback)
    solved=jax.lax.cond(jnp.all(jnp.isfinite(solved)),lambda _:solved,repair,None)
    return jnp.swapaxes(solved,-1,-2)
