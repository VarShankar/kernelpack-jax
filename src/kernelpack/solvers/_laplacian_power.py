"""Sparse block representation of repeated, boundary-closed Laplacians.

The module deliberately does not form a dense Schur complement or an explicit
power of a Laplacian.  It represents each stage using the original sparse
physical Laplacian blocks and the ghost block ``L_BG``.

Node ordering is ``[interior; boundary; ghost]``.  The physical Laplacian has
rows ``[interior; boundary]`` and columns ``[interior; boundary; ghost]``;
there is one ghost per boundary node.  For an intermediate field with interior
values ``q_j`` and prescribed boundary trace ``c_j``, the ghost values at that
stage are obtained from

    L_BI q_j + L_BB c_j + L_BG z_j = c_(j+1).

The next interior trace is then ``q_(j+1) = L_II q_j + L_IB c_j + L_IG z_j``.
The caller supplies the traces ``c_0,...,c_k``.  This is essential for
nonzero or time-dependent boundary data: the module never assumes that every
intermediate Laplacian has zero boundary trace, and it never invents traces
from the original Dirichlet value alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np

from ._common import SparseCOOMatrix, sparse_matvec


@dataclass(frozen=True)
class SparseBlock:
    """A COO sparse block with local row and column indices."""

    indices: jnp.ndarray
    values: jnp.ndarray
    shape: tuple[int, int]


def _extract_block(
    matrix: SparseCOOMatrix,
    row_start: int,
    row_stop: int,
    col_start: int,
    col_stop: int,
) -> SparseBlock:
    indices = np.asarray(matrix.indices)
    values = np.asarray(matrix.values)
    keep = (
        (indices[:, 0] >= row_start)
        & (indices[:, 0] < row_stop)
        & (indices[:, 1] >= col_start)
        & (indices[:, 1] < col_stop)
    )
    local = indices[keep].copy()
    local[:, 0] -= row_start
    local[:, 1] -= col_start
    return SparseBlock(
        indices=jnp.asarray(local, dtype=jnp.int32),
        values=jnp.asarray(values[keep]),
        shape=(row_stop - row_start, col_stop - col_start),
    )


def _block_matvec(block: SparseBlock, x: jnp.ndarray) -> jnp.ndarray:
    return sparse_matvec(block.indices, block.values, block.shape[0], x)


def evaluate_boundary_trace_chain(trace_callback, t, boundary_coordinates, power: int) -> jnp.ndarray:
    """Evaluate explicitly prescribed traces ``c_j(t,x_b)`` for ``j=0..k``.

    The callback contract is exactly ``trace_callback(order, t, x_b)`` and
    each returned vector must have one value per boundary node.  The library
    does not infer higher traces from a Dirichlet callback or from a
    manufactured forcing function.
    """
    power = int(power)
    if power < 0:
        raise ValueError("power must be nonnegative")
    if not callable(trace_callback):
        raise TypeError("trace_callback must be callable as (order, t, x_b)")
    x_b = jnp.asarray(boundary_coordinates)
    if x_b.ndim != 2:
        raise ValueError("boundary_coordinates must have shape (n_boundary, dim)")
    traces = []
    for order in range(power + 1):
        trace = jnp.asarray(trace_callback(order, t, x_b)).reshape(-1)
        if int(trace.shape[0]) != int(x_b.shape[0]):
            raise ValueError(f"trace callback order {order} returned the wrong boundary length")
        traces.append(trace)
    return jnp.stack(traces, axis=0)


@dataclass(frozen=True)
class BoundaryClosedLaplacian:
    """Sparse blocks needed to apply repeated Laplacians without ``L**k``."""

    n_interior: int
    n_boundary: int
    n_ghost: int
    n_full: int
    L_I: SparseBlock
    L_BI: SparseBlock
    L_BB: SparseBlock
    L_BG: SparseBlock
    L_II: SparseBlock
    L_IB: SparseBlock
    L_IG: SparseBlock

    @classmethod
    def from_sparse(
        cls,
        laplacian: SparseCOOMatrix,
        *,
        n_interior: int,
        n_boundary: int,
    ) -> "BoundaryClosedLaplacian":
        """Split a physical-row Laplacian using ``[I; B; G]`` node ordering."""
        n_interior = int(n_interior)
        n_boundary = int(n_boundary)
        n_physical = n_interior + n_boundary
        n_full = int(laplacian.shape[1])
        n_ghost = n_full - n_physical
        if n_interior <= 0 or n_boundary <= 0:
            raise ValueError("interior and boundary node counts must be positive")
        if int(laplacian.shape[0]) != n_physical:
            raise ValueError("Laplacian rows must be ordered [interior; boundary]")
        if n_ghost != n_boundary:
            raise ValueError("this ghost closure requires one ghost node per boundary node")
        if np.asarray(laplacian.indices).ndim != 2 or np.asarray(laplacian.indices).shape[1] != 2:
            raise ValueError("Laplacian indices must be a two-column COO array")

        i0, i1 = 0, n_interior
        b0, b1 = n_interior, n_physical
        g0, g1 = n_physical, n_full
        return cls(
            n_interior=n_interior,
            n_boundary=n_boundary,
            n_ghost=n_ghost,
            n_full=n_full,
            L_I=_extract_block(laplacian, i0, i1, 0, n_full),
            L_BI=_extract_block(laplacian, b0, b1, i0, i1),
            L_BB=_extract_block(laplacian, b0, b1, b0, b1),
            L_BG=_extract_block(laplacian, b0, b1, g0, g1),
            L_II=_extract_block(laplacian, i0, i1, i0, i1),
            L_IB=_extract_block(laplacian, i0, i1, b0, b1),
            L_IG=_extract_block(laplacian, i0, i1, g0, g1),
        )

    def apply_homogeneous_power(
        self,
        interior: jnp.ndarray,
        power: int,
        solve_ghost: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        """Apply ``A**power`` for the homogeneous derived-field closure.

        ``A y = L_II y - L_IG solve(L_BG, L_BI y)``.  The supplied solve
        callback should solve ``L_BG z = rhs`` with the same fixed sparse
        factor at every stage.  It must be differentiable with respect to its
        right-hand side if this operation participates in autodiff.
        """
        power = int(power)
        y = jnp.asarray(interior)
        if power < 0:
            raise ValueError("power must be nonnegative")
        if y.ndim != 1 or int(y.shape[0]) != self.n_interior:
            raise ValueError("interior input must be a vector of length n_interior")
        for _ in range(power):
            ghost_rhs = -_block_matvec(self.L_BI, y)
            ghost = jnp.asarray(solve_ghost(ghost_rhs))
            if ghost.ndim != 1 or int(ghost.shape[0]) != self.n_ghost:
                raise ValueError("solve_ghost must return a vector of length n_ghost")
            y = _block_matvec(self.L_II, y) + _block_matvec(self.L_IG, ghost)
        return y

    def apply_from_full_state(
        self,
        full_state: jnp.ndarray,
        boundary_traces: jnp.ndarray,
        power: int,
        solve_ghost: Callable[[jnp.ndarray], jnp.ndarray],
    ) -> jnp.ndarray:
        """Apply the exact prescribed-trace chain to one existing full state.

        ``boundary_traces[j]`` is the prescribed physical boundary trace of ``Delta**j u``
        at the time being evaluated, for ``j=0,...,power``.  The first
        Laplacian action uses the full state, including its actual boundary and
        ghost values.  Later stages use the supplied ``c_j`` and ``c_(j+1)``
        in the block ghost closure.  The returned vector is ``[q_power;c_power]``.

        This does not require or check that the supplied ``c_1`` equals the
        boundary-row Laplacian of ``full_state``; callers define the intended
        trace chain explicitly, for example from analytic manufactured
        boundary data.  The method is a prescribed-trace composition.

        This supports candidate-dependent traces if the caller constructs them
        from the candidate state and parameters; no exact solution is assumed.
        """
        power = int(power)
        state = jnp.asarray(full_state)
        traces = jnp.asarray(boundary_traces)
        if power < 1:
            raise ValueError("power must be at least one for a full-state Laplacian chain")
        if state.ndim != 1 or int(state.shape[0]) != self.n_full:
            raise ValueError("full_state must be a vector of length n_full")
        if traces.ndim != 2 or tuple(traces.shape) != (power + 1, self.n_boundary):
            raise ValueError("boundary_traces must have shape (power + 1, n_boundary)")

        y = _block_matvec(self.L_I, state)
        for stage in range(1, power):
            c_j = traces[stage]
            c_next = traces[stage + 1]
            ghost_rhs = (
                c_next
                - _block_matvec(self.L_BI, y)
                - _block_matvec(self.L_BB, c_j)
            )
            ghost = jnp.asarray(solve_ghost(ghost_rhs))
            if ghost.ndim != 1 or int(ghost.shape[0]) != self.n_ghost:
                raise ValueError("solve_ghost must return a vector of length n_ghost")
            y = (
                _block_matvec(self.L_II, y)
                + _block_matvec(self.L_IB, c_j)
                + _block_matvec(self.L_IG, ghost)
            )
        return jnp.concatenate([y, traces[power]])


@dataclass(frozen=True)
class PreparedAugmentedLaplacianLayout:
    """Setup-once sparse layout for implicit ``nu*L + gamma*L**k`` systems.

    COO indices and the three value channels are assembled on the host once.
    ``system`` and ``rhs`` then use only JAX operations, and can be called
    inside jit/JVP/VJP with traced coefficients and time-varying traces.
    """

    blocks: BoundaryClosedLaplacian
    indices: jnp.ndarray
    base_values: jnp.ndarray
    lap_coefficients: jnp.ndarray
    power_coefficients: jnp.ndarray
    power: int
    stage_scale: float
    n_total: int

    @classmethod
    def from_operators(
        cls,
        laplacian: SparseCOOMatrix,
        boundary_operator: SparseCOOMatrix,
        *,
        n_interior: int,
        n_boundary: int,
        power: int,
        stage_scale: float = 1.0,
    ) -> "PreparedAugmentedLaplacianLayout":
        """Prepare all static COO structure once for repeated time steps."""
        blocks = BoundaryClosedLaplacian.from_sparse(
            laplacian, n_interior=n_interior, n_boundary=n_boundary
        )
        power = int(power)
        if power < 1:
            raise ValueError("power must be at least one")
        stage_scale = float(stage_scale)
        if not np.isfinite(stage_scale) or stage_scale <= 0.0:
            raise ValueError("stage_scale must be finite and positive")
        n_i, n_b, n_g = blocks.n_interior, blocks.n_boundary, blocks.n_ghost
        n_phys, n_full = n_i + n_b, blocks.n_full
        if tuple(boundary_operator.shape) != (n_b, n_full):
            raise ValueError("boundary_operator must have shape (n_boundary, n_full)")

        all_i = np.asarray(laplacian.indices, dtype=np.int32)
        all_v = np.asarray(laplacian.values)
        bc_i = np.asarray(boundary_operator.indices, dtype=np.int32)
        bc_v = np.asarray(boundary_operator.values)
        g0 = n_phys
        q0 = n_full
        z0 = n_full + power * n_i
        total = n_full + power * n_i + (power - 1) * n_g
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        bases: list[np.ndarray] = []
        lap_coeffs: list[np.ndarray] = []
        power_coeffs: list[np.ndarray] = []

        def add(
            row: np.ndarray,
            col: np.ndarray,
            *,
            base: np.ndarray | float = 0.0,
            lap: np.ndarray | float = 0.0,
            hyper: np.ndarray | float = 0.0,
        ) -> None:
            row = np.asarray(row, dtype=np.int32).reshape(-1)
            col = np.asarray(col, dtype=np.int32).reshape(-1)
            if row.size != col.size:
                raise ValueError("COO row and column lengths differ")

            def expand(value):
                array = np.asarray(value).reshape(-1)
                if array.size == 1 and row.size != 1:
                    array = np.full(row.size, array.item(), dtype=np.result_type(array.dtype, all_v.dtype))
                if array.size != row.size:
                    raise ValueError("COO value count differs from row count")
                return array

            if row.size:
                rows.append(row)
                cols.append(col)
                bases.append(expand(base))
                lap_coeffs.append(expand(lap))
                power_coeffs.append(expand(hyper))

        # Base physical PDE rows: I + lap_scale*L + power_scale*q_k.
        add(np.arange(n_phys), np.arange(n_phys), base=1.0)
        add(all_i[:, 0], all_i[:, 1], lap=all_v)
        qk = q0 + (power - 1) * n_i
        add(np.arange(n_i), qk + np.arange(n_i), hyper=stage_scale ** (-power))

        # Original boundary equations remain separate and unscaled.
        add(bc_i[:, 0] + n_phys, bc_i[:, 1], base=bc_v)

        # q1 = L_I u_full.
        q1_rows = n_full + np.arange(n_i)
        add(q1_rows, q0 + np.arange(n_i), base=1.0)
        keep_i = all_i[:, 0] < n_i
        add(q1_rows[all_i[keep_i, 0]], all_i[keep_i, 1], base=-stage_scale * all_v[keep_i])

        # Repeated stages q_(j+1), z_j. Boundary traces enter only the RHS.
        for stage in range(1, power):
            stage_row = n_full + n_i + (stage - 1) * (n_i + n_g)
            q_stage = q0 + (stage - 1) * n_i
            q_next = q0 + stage * n_i
            z_stage = z0 + (stage - 1) * n_g
            qrows = stage_row + np.arange(n_i)
            add(qrows, q_next + np.arange(n_i), base=1.0)
            ii = (all_i[:, 0] < n_i) & (all_i[:, 1] < n_i)
            ig = (all_i[:, 0] < n_i) & (all_i[:, 1] >= g0)
            add(qrows[all_i[ii, 0]], q_stage + all_i[ii, 1], base=-stage_scale * all_v[ii])
            add(qrows[all_i[ig, 0]], z_stage + (all_i[ig, 1] - g0), base=-stage_scale * all_v[ig])

            brows = stage_row + n_i + np.arange(n_g)
            bi = (all_i[:, 0] >= n_i) & (all_i[:, 1] < n_i)
            bg = (all_i[:, 0] >= n_i) & (all_i[:, 1] >= g0)
            add(brows[all_i[bi, 0] - n_i], q_stage + all_i[bi, 1], base=stage_scale * all_v[bi])
            add(brows[all_i[bg, 0] - n_i], z_stage + (all_i[bg, 1] - g0), base=stage_scale * all_v[bg])

        row = np.concatenate(rows) if rows else np.zeros((0,), dtype=np.int32)
        col = np.concatenate(cols) if cols else np.zeros((0,), dtype=np.int32)
        base_values = np.concatenate(bases) if bases else np.zeros((0,), dtype=all_v.dtype)
        lap_values = np.concatenate(lap_coeffs) if lap_coeffs else np.zeros((0,), dtype=all_v.dtype)
        power_values = np.concatenate(power_coeffs) if power_coeffs else np.zeros((0,), dtype=all_v.dtype)
        if not (row.size == col.size == base_values.size == lap_values.size == power_values.size):
            raise RuntimeError("prepared sparse value channels have inconsistent lengths")
        return cls(
            blocks=blocks,
            indices=jnp.asarray(np.column_stack([row, col]), dtype=jnp.int32),
            base_values=jnp.asarray(base_values),
            lap_coefficients=jnp.asarray(lap_values),
            power_coefficients=jnp.asarray(power_values),
            power=power,
            stage_scale=stage_scale,
            n_total=total,
        )

    def system(self, lap_scale, power_scale) -> SparseCOOMatrix:
        """Form runtime values from traced coefficients without host work."""
        dtype = self.base_values.dtype
        values = (
            self.base_values
            + jnp.asarray(lap_scale, dtype=dtype) * self.lap_coefficients
            + jnp.asarray(power_scale, dtype=dtype) * self.power_coefficients
        )
        return SparseCOOMatrix(self.indices, values, (self.n_total, self.n_total))

    def rhs(self, rhs_physical, rhs_boundary, boundary_traces, power_scale) -> jnp.ndarray:
        """Form the time-dependent RHS using JAX operations only."""
        n_i, n_b, n_g = self.blocks.n_interior, self.blocks.n_boundary, self.blocks.n_ghost
        traces = jnp.asarray(boundary_traces)
        if traces.ndim != 2 or tuple(traces.shape) != (self.power + 1, n_b):
            raise ValueError("boundary_traces must have shape (power + 1, n_boundary)")
        rhs_p = jnp.asarray(rhs_physical).reshape(-1)
        rhs_b = jnp.asarray(rhs_boundary).reshape(-1)
        if int(rhs_p.shape[0]) != n_i + n_b or int(rhs_b.shape[0]) != n_b:
            raise ValueError("physical and boundary RHS lengths do not match the operators")
        physical_rhs = rhs_p.at[n_i:].add(-jnp.asarray(power_scale, dtype=rhs_p.dtype) * traces[self.power])
        aux_rhs_parts: list[jnp.ndarray] = [jnp.zeros((n_i,), dtype=rhs_p.dtype)]
        for stage in range(1, self.power):
            trace_scale = self.stage_scale ** (stage + 1)
            aux_rhs_parts.append(trace_scale * _block_matvec(self.blocks.L_IB, traces[stage]))
            aux_rhs_parts.append(
                trace_scale * (traces[stage + 1] - _block_matvec(self.blocks.L_BB, traces[stage]))
            )
        result = jnp.concatenate([physical_rhs, rhs_b, *aux_rhs_parts])
        if int(result.shape[0]) != self.n_total:
            raise RuntimeError("augmented RHS length does not match the sparse system")
        return result


@dataclass(frozen=True)
class AugmentedLaplacianSystem:
    """Sparse square system and RHS for an implicit ``nu*L + gamma*L**k``."""

    matrix: SparseCOOMatrix
    rhs: jnp.ndarray
    n_full: int
    n_interior: int
    n_boundary: int
    power: int

    def physical_state(self, solution: jnp.ndarray) -> jnp.ndarray:
        """Extract the original ``[interior; boundary; ghost]`` state."""
        return jnp.asarray(solution)[: self.n_full]


def build_augmented_implicit_system(
    laplacian: SparseCOOMatrix,
    boundary_operator: SparseCOOMatrix,
    *,
    n_interior: int,
    n_boundary: int,
    power: int,
    stage_scale: float = 1.0,
    lap_scale: float,
    power_scale: float,
    rhs_physical: jnp.ndarray,
    rhs_boundary: jnp.ndarray,
    boundary_traces: jnp.ndarray,
) -> AugmentedLaplacianSystem:
    """Convenience setup wrapper; repeated callers should retain the layout."""
    layout = PreparedAugmentedLaplacianLayout.from_operators(
        laplacian,
        boundary_operator,
        n_interior=n_interior,
        n_boundary=n_boundary,
        power=power,
        stage_scale=stage_scale,
    )
    matrix = layout.system(lap_scale, power_scale)
    rhs = layout.rhs(rhs_physical, rhs_boundary, boundary_traces, power_scale)
    return AugmentedLaplacianSystem(
        matrix=matrix,
        rhs=rhs,
        n_full=layout.blocks.n_full,
        n_interior=layout.blocks.n_interior,
        n_boundary=layout.blocks.n_boundary,
        power=layout.power,
    )
