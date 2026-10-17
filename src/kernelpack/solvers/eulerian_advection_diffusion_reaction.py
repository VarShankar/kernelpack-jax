from __future__ import annotations
from ._cudss import prepare_transient_cudss, close_transient_cudss, solve_transient_cudss, get_transient_cudss_factor

from copy import copy
from dataclasses import dataclass, field
from math import comb
from typing import Callable

from jax import debug, lax
import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import eigvals
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigs

from kernelpack.domain import DomainDescriptor
from kernelpack.geometry import distance_matrix
from kernelpack.rbffd import OpProperties, StencilProperties, build_frozen_stencil_graph
from ._common import (
    FixedDiscretization,
    FrozenILUPreconditioner,
    SparseCOOMatrix,
    assemble_operator,
    assemble_operator_sparse,
    build_domain_state,
    build_boundary_paired_ghost_stencil_graph,
    build_fixed_discretization,
    build_fixed_ilu_preconditioner,
    build_implicit_rhs,
    build_implicit_system,
    build_implicit_system_sparse,
    build_stencil_properties,
    evaluate_boundary_coefficient,
    evaluate_forcing_callback,
    evaluate_transient_boundary_values,
    is_fixed_boundary_callback,
    solve_dense_system,
    solve_sparse_system_gmres,
    sparse_matvec,
    sparse_matrix_to_dense,
    _ensure_unbatched_operator_input,
)
from .reaction_diffusion import _evaluate_reaction_callback, _solve_lineax_system


def _evaluate_velocity_callback(
    velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
    t: float,
    x: jnp.ndarray,
    dim: int,
) -> jnp.ndarray:
    """Normalize velocity callbacks to one vector value per physical node."""

    values = velocity(t, x) if callable(velocity) else velocity
    arr = jnp.asarray(values, dtype=float)
    if arr.ndim == 1:
        if dim != 1:
            raise ValueError("one-dimensional velocity output is only valid in 1D")
        arr = arr[:, None]
    if arr.ndim != 2 or int(arr.shape[0]) != int(x.shape[0]) or int(arr.shape[1]) != dim:
        raise ValueError("velocity must return an array with shape (num_physical_nodes, dim)")
    return arr


def _laplacian_power_phs(r: jnp.ndarray, degree: int, dim: int, power: int) -> jnp.ndarray:
    """Evaluate Delta^power r^degree for odd PHS powers."""

    coeff = 1.0
    for s in range(power):
        coeff *= (degree - 2 * s) * (degree + dim - 2 - 2 * s)
    values = coeff * r ** (degree - 2 * power)
    return jnp.where(jnp.isfinite(values), values, 0.0)


def make_hyperviscosity_operator(power: int) -> Callable[..., jnp.ndarray]:
    """Return a direct RBF-FD operator for Delta^power.

    Use PHS r^(2k+1) and reproduce every polynomial below derivative order
    2k. This keeps its action on smooth data bounded under stable scaled
    weights, so the coefficient's h power can preserve the spatial order.
    The polynomial derivative RHS is zero for ell=2k-1.
    """

    def op(stencil, sp, _op, r_rhs, _x_subset, _x, _x_at_origin_subset, _x_at_origin):
        radial = _laplacian_power_phs(r_rhs, sp.spline_degree, stencil.s_dim, power).T
        poly = jnp.zeros((stencil.npoly, radial.shape[1]), dtype=radial.dtype)
        return jnp.vstack([radial, poly])

    # Explicit mathematical contract for the common batched RBF-FD assembler.
    op._kernelpack_laplacian_power = power
    return op


def build_hyperviscosity_stencil_properties(domain: DomainDescriptor, power: int) -> StencilProperties:
    """RBF-FD filter annihilating all polynomials below derivative order 2k."""

    dim = domain.get_dim()
    ell = 2 * power - 1
    npoly = comb(dim + ell, dim)
    return StencilProperties(
        n=2 * npoly + 1,
        dim=dim,
        ell=ell,
        spline_degree=2 * power + 1,
        npoly=npoly,
        tree_mode="all",
        point_set="interior_boundary",
    )


def paper_gamma_component(
    h: float,
    velocity_scale: float,
    tau: float,
    q: float,
    power: int,
    dimension: int = 1,
) -> float:
    """One coordinate contribution to the Euclidean hyperviscosity scale.

    The original three-dimensional formula contains ``3**(-power)``; in
    dimension ``d`` this is ``d**(-power)``.  ``dimension=1`` preserves the
    historical one-dimensional primitive for callers that omit the argument.
    """
    if int(dimension) < 1:
        raise ValueError("dimension must be a positive integer")
    return (
        ((-1.0) ** (1 - power))
        * (float(dimension) ** (-power))
        * (2.0 ** (q - 2 * power))
        * (h ** (2 * power - q))
        * velocity_scale
        * tau
    )


@dataclass(frozen=True)
class HyperviscosityEstimate:
    tau: jnp.ndarray
    q: jnp.ndarray
    gamma_components: jnp.ndarray
    h: float
    wavenumber: float
    power: int


def _scaled_sparse_sum(terms: list[tuple[jnp.ndarray | float, SparseCOOMatrix]]) -> SparseCOOMatrix:
    """Combine sparse matrices, keeping duplicate COO entries additive."""

    matrices = [matrix for _scale, matrix in terms]
    shape = matrices[0].shape
    indices = jnp.concatenate([matrix.indices for _scale, matrix in terms], axis=0)
    values = jnp.concatenate([jnp.asarray(scale, dtype=float) * matrix.values for scale, matrix in terms], axis=0)
    return SparseCOOMatrix(indices=indices, values=values, shape=shape)


def _to_scipy_coo(matrix: SparseCOOMatrix) -> coo_matrix:
    indices = np.asarray(matrix.indices, dtype=np.int32)
    values = np.asarray(matrix.values, dtype=np.float64)
    return coo_matrix((values, (indices[:, 0], indices[:, 1])), shape=matrix.shape)


def _validate_full_state(state: jnp.ndarray, nf: int) -> jnp.ndarray:
    arr = jnp.asarray(state, dtype=float)
    if arr.ndim == 1:
        if arr.size != nf:
            raise ValueError(f"expected a full state of length {nf}")
        return arr
    if arr.ndim == 2 and arr.shape[0] == nf:
        return arr
    raise ValueError(f"expected a full state with leading dimension {nf}")


def _sparse_relative_residual(system: SparseCOOMatrix, solution: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    residual = sparse_matvec(system.indices, system.values, system.shape[0], solution) - rhs
    return jnp.linalg.norm(residual) / jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(rhs.dtype).tiny)


@dataclass
class EulerianAdvectionDiffusionReactionSolver:
    """Eulerian RBF-FD advection-diffusion-reaction solver with ghost unknowns.

    The time history stores the full vector `[physical; ghost]`. PDE rows are
    enforced on interior+boundary nodes, and boundary rows close the ghost-node
    system at the new time.
    """

    lap_assembler: str = "fd"
    bc_assembler: str = "fd"
    lap_stencil: str = "rbf"
    bc_stencil: str = "rbf"
    domain: DomainDescriptor = field(default_factory=DomainDescriptor)
    xi: int = 0
    boundary_xi: int | None = None
    dt: float = jnp.nan
    nu: float = jnp.nan
    hyperviscosity_power: int = 2
    hyperviscosity_method: str = "direct_rbf"
    hyperviscosity_treatment: str = "implicit"
    laplacian_power_solver: str = "sparse"
    explicit_power_evaluator: object | None = None
    hyperviscosity_boundary_traces: Callable | None = None
    power_layout: object | None = None
    power_layout_key: tuple | None = None
    num_omp_threads: int = 1
    linear_solver: str = "gmres_ilu"
    x: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xall: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    xb: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    nr: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    n: int = 0
    nf: int = 0
    dim: int = 0
    lap: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    hyper: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    bc: jnp.ndarray = field(default_factory=lambda: jnp.zeros((0, 0)))
    grad: list[jnp.ndarray] = field(default_factory=list)
    lap_sparse: SparseCOOMatrix | None = None
    full_lap_sparse: SparseCOOMatrix | None = None
    hyper_sparse: SparseCOOMatrix | None = None
    bc_sparse: SparseCOOMatrix | None = None
    grad_sparse: list[SparseCOOMatrix] = field(default_factory=list)
    lap_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    bc_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    hyper_stencil_properties: StencilProperties = field(default_factory=StencilProperties)
    op_properties: OpProperties = field(default_factory=lambda: OpProperties(decompose=False, store_weights=True, record_stencils=False))
    fixed_discretization: FixedDiscretization | None = None
    hyper_graph: object | None = None
    cfnm2: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cfnm1: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cfn: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    tnm2: float = 0.0
    tnm1: float = 0.0
    tn: float = 0.0
    completed_steps_: int = 0
    gamma_components: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    fixed_bc_operator_ready_: bool = False
    fixed_bc_coefficients_ready_: bool = False
    cached_neu_coeff_: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    cached_dir_coeff_: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    fixed_preconditioner: FrozenILUPreconditioner | None = None
    assemble_dense_operators: bool = True
    gmres_tol: float = 1e-8
    gmres_atol: float = 1e-10
    gmres_restart: int = 40
    gmres_maxiter: int | None = None
    linear_solve_diagnostics: bool = False
    linear_solve_diagnostics_threshold: float = 1e-6
    hyperviscosity_row_mask: str = "none"
    hyperviscosity_mask_distance_factor: float = 2.0
    sparse_assembly_chunk_size: int | None = None
    force_boundary_paired_ghost_stencil: bool = False

    def prepare_cudss(self, *, orders=(1, 2, 3)):
        self._prepare_explicit_hyperviscosity()
        prepare_transient_cudss(self, orders=orders)

    def close_cudss(self):
        close_transient_cudss(self)
        if self.explicit_power_evaluator is not None:
            self.explicit_power_evaluator.close()
            self.explicit_power_evaluator = None

    def _prepare_explicit_hyperviscosity(self):
        if (self.hyperviscosity_treatment == "explicit"
                and self.hyperviscosity_method == "prescribed_trace_power"
                and self.explicit_power_evaluator is None):
            from ._explicit_laplacian_power import PreparedExplicitLaplacianPower
            if self.hyperviscosity_boundary_traces is None:
                raise ValueError("Explicit Laplacian powers require prescribed boundary traces")
            self.explicit_power_evaluator = PreparedExplicitLaplacianPower.prepare(
                self.lap_sparse, n_interior=self.domain.get_num_interior_nodes(),
                n_boundary=self.domain.get_num_bdry_nodes(), power=self.hyperviscosity_power,
            )

    def _explicit_hyperviscosity_rhs(self, t, full_state, gamma_components=None):
        gamma = self.gamma_components if gamma_components is None else gamma_components
        if self.hyperviscosity_method == "prescribed_trace_power":
            if self.explicit_power_evaluator is None:
                raise ValueError("Prepare explicit hyperviscosity before tracing the rollout")
            value = self.explicit_power_evaluator.apply(
                full_state, t, self.xb, self.hyperviscosity_boundary_traces)
        else:
            value = self._apply_physical_operator(self.hyper, self.hyper_sparse, full_state)
        return jnp.sum(gamma) * value

    def spawn_rollout_copy(self) -> "EulerianAdvectionDiffusionReactionSolver":
        clone = copy(self)
        clone.cfnm2 = jnp.zeros((self.nf,), dtype=float)
        clone.cfnm1 = jnp.zeros((self.nf,), dtype=float)
        clone.cfn = jnp.zeros((self.nf,), dtype=float)
        clone.tnm2 = 0.0
        clone.tnm1 = 0.0
        clone.tn = 0.0
        clone.completed_steps_ = 0
        clone.fixed_preconditioner = self.fixed_preconditioner
        return clone

    def init(
        self,
        domain: DomainDescriptor,
        xi: int,
        dlt: float,
        d_coeff: float,
        *,
        hyperviscosity_power: int = 2,
        hyperviscosity_method: str | None = None,
        hyperviscosity_treatment: str | None = None,
        laplacian_power_solver: str | None = None,
        hyperviscosity_boundary_traces: Callable | None = None,
        gamma_components: jnp.ndarray | None = None,
        num_omp_threads: int = 1,
        assemble_dense_operators: bool | None = None,
        hyperviscosity_row_mask: str | None = None,
        hyperviscosity_mask_distance_factor: float | None = None,
        sparse_assembly_chunk_size: int | None = None,
        force_boundary_paired_ghost_stencil: bool | None = None,
        boundary_xi: int | None = None,
    ) -> None:
        self.domain = domain
        self.xi = xi
        self.boundary_xi = None if boundary_xi is None else int(boundary_xi)
        self.dt = dlt
        self.nu = d_coeff
        self.hyperviscosity_power = int(hyperviscosity_power)
        if hyperviscosity_method is not None:
            self.hyperviscosity_method = str(hyperviscosity_method)
        if self.hyperviscosity_method not in ("laplacian_power", "prescribed_trace_power", "direct_rbf"):
            raise ValueError("hyperviscosity_method must be laplacian_power, direct_rbf, or legacy prescribed_trace_power")
        if hyperviscosity_treatment is not None:
            self.hyperviscosity_treatment = str(hyperviscosity_treatment)
        if self.hyperviscosity_treatment not in ("implicit", "explicit"):
            raise ValueError("hyperviscosity_treatment must be implicit or explicit")
        if laplacian_power_solver is not None:
            self.laplacian_power_solver = str(laplacian_power_solver)
        if self.laplacian_power_solver not in ("sparse", "augmented", "factorized"):
            raise ValueError("laplacian_power_solver must be sparse, augmented, or factorized")
        if self.hyperviscosity_method == "laplacian_power" and self.laplacian_power_solver != "sparse":
            raise ValueError("laplacian_power uses the full square sparse matrix power; select laplacian_power_solver='sparse'")
        self.explicit_power_evaluator = None
        self.hyperviscosity_boundary_traces = hyperviscosity_boundary_traces
        self.power_layout = None
        self.power_layout_key = None
        self.num_omp_threads = num_omp_threads
        if assemble_dense_operators is not None:
            self.assemble_dense_operators = bool(assemble_dense_operators)
        elif self.linear_solver == "cudss":
            self.assemble_dense_operators = False
        if hyperviscosity_row_mask is not None:
            self.hyperviscosity_row_mask = self._validate_hyperviscosity_row_mask(hyperviscosity_row_mask)
        if hyperviscosity_mask_distance_factor is not None:
            self.hyperviscosity_mask_distance_factor = float(hyperviscosity_mask_distance_factor)
        if sparse_assembly_chunk_size is not None:
            self.sparse_assembly_chunk_size = int(sparse_assembly_chunk_size)
        if force_boundary_paired_ghost_stencil is not None:
            self.force_boundary_paired_ghost_stencil = bool(force_boundary_paired_ghost_stencil)
        state = build_domain_state(self.domain)
        self.x = state.x
        self.xall = self.domain.get_all_nodes()
        self.xb = state.xb
        self.nr = state.nr
        self.n = state.n
        self.nf = state.nf
        self.dim = self.domain.get_dim()
        self.gamma_components = jnp.zeros((self.dim,), dtype=float) if gamma_components is None else self._validate_gamma(gamma_components)

        self.lap_stencil_properties = build_stencil_properties(self.domain, self.xi, 2, "interior_boundary")
        bc_xi = self.xi if self.boundary_xi is None else self.boundary_xi
        self.bc_stencil_properties = build_stencil_properties(self.domain, bc_xi, 1, "boundary")
        self.hyper_stencil_properties = build_hyperviscosity_stencil_properties(self.domain, self.hyperviscosity_power)
        # Sparse chunked setup must not precompute full stencil graphs. The
        # sparse assembler builds each graph slice from its active rows.
        use_incremental_sparse_graphs = (
            self.sparse_assembly_chunk_size is not None
            and self.sparse_assembly_chunk_size > 0
            and not self.assemble_dense_operators
        )
        self.fixed_discretization = (
            None
            if use_incremental_sparse_graphs
            else build_fixed_discretization(self.domain, self.lap_stencil_properties, self.bc_stencil_properties)
        )
        self.hyper_graph = (None if use_incremental_sparse_graphs or self.hyperviscosity_method in ("laplacian_power", "prescribed_trace_power")
                            else build_frozen_stencil_graph(self.domain, self.hyper_stencil_properties))

        if self.assemble_dense_operators and self.hyperviscosity_method != "laplacian_power":
            self.lap = assemble_operator(
                self.domain,
                self.lap_assembler,
                self.lap_stencil,
                "lap",
                self.lap_stencil_properties,
                self.op_properties,
                stencil_graph=self.fixed_discretization.laplacian_graph,
            )
        else:
            self.lap = jnp.zeros((0, 0), dtype=float)
        if self.hyperviscosity_method == "laplacian_power":
            from dataclasses import replace
            full_properties = replace(self.lap_stencil_properties, point_set="all", tree_mode="all")
            self.full_lap_sparse = assemble_operator_sparse(
                self.domain, self.lap_assembler, self.lap_stencil, "lap",
                full_properties, self.op_properties,
                chunk_size=self.sparse_assembly_chunk_size,
            )
            physical_rows = self.full_lap_sparse.indices[:, 0] < self.n
            self.lap_sparse = SparseCOOMatrix(
                self.full_lap_sparse.indices[physical_rows],
                self.full_lap_sparse.values[physical_rows], (self.n, self.nf),
            )
            if self.assemble_dense_operators:
                self.lap = sparse_matrix_to_dense(self.lap_sparse)
        else:
            self.full_lap_sparse = None
            self.lap_sparse = assemble_operator_sparse(
                self.domain, self.lap_assembler, self.lap_stencil, "lap",
                self.lap_stencil_properties, self.op_properties,
                stencil_graph=None if self.fixed_discretization is None else self.fixed_discretization.laplacian_graph,
                chunk_size=self.sparse_assembly_chunk_size,
            )
        self.grad = []
        self.grad_sparse = []
        for d in range(self.dim):
            props = OpProperties(selectdim=d, decompose=False, store_weights=True, record_stencils=False)
            if self.assemble_dense_operators:
                self.grad.append(
                    assemble_operator(
                        self.domain,
                        self.lap_assembler,
                        self.lap_stencil,
                        "grad",
                        self.lap_stencil_properties,
                        props,
                        stencil_graph=None if self.fixed_discretization is None else self.fixed_discretization.laplacian_graph,
                    )
                )
            else:
                self.grad.append(jnp.zeros((0, 0), dtype=float))
            self.grad_sparse.append(
                assemble_operator_sparse(
                    self.domain,
                    self.lap_assembler,
                    self.lap_stencil,
                    "grad",
                    self.lap_stencil_properties,
                    props,
                    stencil_graph=None if self.fixed_discretization is None else self.fixed_discretization.laplacian_graph,
                    chunk_size=self.sparse_assembly_chunk_size,
                )
            )

        hyper_op = make_hyperviscosity_operator(self.hyperviscosity_power)
        if self.assemble_dense_operators and self.hyperviscosity_method == "direct_rbf":
            self.hyper = assemble_operator(
                self.domain,
                self.lap_assembler,
                self.lap_stencil,
                hyper_op,
                self.hyper_stencil_properties,
                self.op_properties,
                stencil_graph=self.hyper_graph,
            )
        else:
            self.hyper = jnp.zeros((0, 0), dtype=float)
        if self.hyperviscosity_method == "direct_rbf":
            self.hyper_sparse = assemble_operator_sparse(
                self.domain, self.lap_assembler, self.lap_stencil, hyper_op,
                self.hyper_stencil_properties, self.op_properties,
                stencil_graph=self.hyper_graph, chunk_size=self.sparse_assembly_chunk_size,
            )
            self._apply_hyperviscosity_row_mask()
        elif self.hyperviscosity_method == "laplacian_power":
            if self.hyperviscosity_row_mask != "none":
                raise ValueError("Row masks would change the requested sparse L^k")
            # Assemble rows at every node, including ghosts. Crop only after
            # composing the full square operator, retaining every column.
            self._rebuild_sparse_laplacian_power()
        else:
            if self.hyperviscosity_row_mask != "none":
                raise ValueError("Row masks would change L^k and are not supported for laplacian_power")
            self.hyper_sparse = None
        self.bc = jnp.zeros((0, self.nf))
        self.bc_sparse = None
        self.set_initial_state(jnp.zeros((self.nf,), dtype=float), t0=0.0)
        self.fixed_bc_operator_ready_ = False
        self.fixed_bc_coefficients_ready_ = False
        self.cached_neu_coeff_ = jnp.zeros(0)
        self.cached_dir_coeff_ = jnp.zeros(0)
        self.fixed_preconditioner = None

    def set_initial_state(self, c0_full: jnp.ndarray, *, t0: float = 0.0) -> None:
        arr = jnp.asarray(c0_full, dtype=float).reshape(-1)
        if int(arr.shape[0]) != self.nf:
            raise ValueError(f"Eulerian ADR initial state must include ghost nodes and have length {self.nf}")
        self.cfnm2 = arr
        self.cfnm1 = jnp.zeros_like(arr)
        self.cfn = jnp.zeros_like(arr)
        self.tnm2 = float(t0)
        self.tnm1 = float(t0)
        self.tn = float(t0)
        self.completed_steps_ = 0

    def current_full_state(self) -> jnp.ndarray:
        if self.completed_steps_ <= 0:
            return self.cfnm2
        if self.completed_steps_ == 1:
            return self.cfnm1
        return self.cfn

    def current_physical_state(self) -> jnp.ndarray:
        return self.current_full_state()[: self.n]

    def set_gamma_components(self, gamma_components: jnp.ndarray) -> None:
        self.gamma_components = self._validate_gamma(gamma_components)
        self.fixed_preconditioner = None

    def set_laplacian_power(self, power: int) -> None:
        """Change a composed power without rebuilding Laplacian or gradients."""
        if self.hyperviscosity_method not in ("laplacian_power", "prescribed_trace_power"):
            raise ValueError("set_laplacian_power requires laplacian_power construction")
        if int(power) < 1:
            raise ValueError("Laplacian power must be positive")
        self.close_cudss()
        self.hyperviscosity_power = int(power)
        if self.hyperviscosity_method == "laplacian_power":
            self._rebuild_sparse_laplacian_power()
            self.fixed_preconditioner = None
            return
        self.power_layout = None
        self.power_layout_key = None
        self.fixed_preconditioner = None
        if self.bc_sparse is not None and self.hyperviscosity_treatment == "implicit":
            from ._composed_adr import prepare_power_layout
            prepare_power_layout(self)

    def _rebuild_sparse_laplacian_power(self) -> None:
        from ._sparse_laplacian_power import power_sparse_laplacian_rows
        if self.full_lap_sparse is None or self.full_lap_sparse.shape != (self.nf, self.nf):
            raise ValueError("Sparse L^k requires the full square all-node Laplacian")
        self.hyper_sparse = power_sparse_laplacian_rows(
            self.full_lap_sparse, self.hyperviscosity_power, self.n,
            backend="cupy" if jax.default_backend() == "gpu" else "cpu",
        )
        self.hyperviscosity_boundary_protocol = "full_square_sparse_power_then_physical_row_crop"
        self.hyper = (sparse_matrix_to_dense(self.hyper_sparse)
                      if self.assemble_dense_operators else jnp.zeros((0, 0), dtype=float))

    @staticmethod
    def _validate_hyperviscosity_row_mask(row_mask: str) -> str:
        mask = str(row_mask).strip().lower()
        if mask not in ("none", "boundary", "near_boundary"):
            raise ValueError("hyperviscosity row mask must be one of: none, boundary, near_boundary")
        return mask

    def _hyperviscosity_physical_row_weights(self) -> jnp.ndarray:
        mask = self._validate_hyperviscosity_row_mask(self.hyperviscosity_row_mask)
        weights = jnp.ones((self.n,), dtype=float)
        if mask == "none":
            return weights
        n_interior = int(self.domain.get_num_interior_nodes())
        n_boundary = int(self.domain.get_num_bdry_nodes())
        if n_boundary > 0:
            weights = weights.at[n_interior : n_interior + n_boundary].set(0.0)
        if mask == "boundary" or n_interior == 0 or n_boundary == 0:
            return weights
        x_interior = jnp.asarray(self.x[:n_interior], dtype=float)
        xb = jnp.asarray(self.xb, dtype=float)
        nearest = jnp.min(distance_matrix(x_interior, xb), axis=1)
        near = nearest <= float(self.hyperviscosity_mask_distance_factor) * float(self.domain.get_sep_rad())
        return weights.at[:n_interior].set(jnp.where(near, 0.0, 1.0))

    def _apply_hyperviscosity_row_mask(self) -> None:
        weights = self._hyperviscosity_physical_row_weights()
        if self.hyper.size:
            self.hyper = weights[:, None] * self.hyper
        if self.hyper_sparse is not None:
            row_weights = weights[self.hyper_sparse.indices[:, 0]]
            self.hyper_sparse = SparseCOOMatrix(
                indices=self.hyper_sparse.indices,
                values=self.hyper_sparse.values * row_weights,
                shape=self.hyper_sparse.shape,
            )

    def prepare_boundary_operator(
        self,
        t: float,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> None:
        neu_coeff, dir_coeff = self._get_boundary_coefficients(t, neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)

    def prepare_preconditioner(
        self,
        t: float,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        *,
        nu_ref: float | None = None,
        gamma_components_ref: jnp.ndarray | None = None,
        bdf_order: int = 1,
        application: str = "dense",
    ) -> FrozenILUPreconditioner:
        neu_coeff, dir_coeff = self._get_boundary_coefficients(t, neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)
        alpha = self._bdf_alpha(bdf_order)
        nu = self.nu if nu_ref is None else float(nu_ref)
        gamma = self.gamma_components if gamma_components_ref is None else self._validate_gamma(gamma_components_ref)
        if self.hyperviscosity_method == "prescribed_trace_power" and self.hyperviscosity_treatment == "implicit":
            sparse_system = self.power_layout.system(-alpha * self.dt * nu, -alpha * self.dt * jnp.sum(gamma))
        else:
            operator_sparse = self._implicit_operator_sparse(nu, gamma)
            sparse_system = build_implicit_system_sparse(operator_sparse, self.bc_sparse, self.n, -alpha * self.dt)
        self.fixed_preconditioner = build_fixed_ilu_preconditioner(
            sparse_system,
            application=self._ilu_application(sparse_system, requested=application),
        )
        return self.fixed_preconditioner

    def bdf1_step(
        self,
        t: float,
        velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
        gamma_components: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        self._prepare_explicit_hyperviscosity()
        prev = self.current_full_state()
        rhs_explicit = self._explicit_rhs(self._current_time(), prev, velocity, reaction, forcing, gamma_components)
        rhs_physical = prev[: self.n] + self.dt * rhs_explicit
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, 1.0, gamma_components)

    def bdf2_step(
        self,
        t: float,
        velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
        gamma_components: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if self.completed_steps_ < 1:
            raise ValueError("bdf2_step requires one prior step")
        self._prepare_explicit_hyperviscosity()
        e0 = self._explicit_rhs(self.tnm2, self.cfnm2, velocity, reaction, forcing, gamma_components)
        e1 = self._explicit_rhs(self.tnm1, self.cfnm1, velocity, reaction, forcing, gamma_components)
        rhs_physical = (4.0 / 3.0) * self.cfnm1[: self.n] - (1.0 / 3.0) * self.cfnm2[: self.n] + (2.0 / 3.0) * self.dt * (2.0 * e1 - e0)
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, 2.0 / 3.0, gamma_components)

    def bdf3_step(
        self,
        t: float,
        velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
        gamma_components: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if self.completed_steps_ < 2:
            raise ValueError("bdf3_step requires two prior steps")
        self._prepare_explicit_hyperviscosity()
        e0 = self._explicit_rhs(self.tnm2, self.cfnm2, velocity, reaction, forcing, gamma_components)
        e1 = self._explicit_rhs(self.tnm1, self.cfnm1, velocity, reaction, forcing, gamma_components)
        e2 = self._explicit_rhs(self.tn, self.cfn, velocity, reaction, forcing, gamma_components)
        rhs_physical = (
            (18.0 / 11.0) * self.cfn[: self.n]
            - (9.0 / 11.0) * self.cfnm1[: self.n]
            + (2.0 / 11.0) * self.cfnm2[: self.n]
            + (6.0 / 11.0) * self.dt * (3.0 * e2 - 3.0 * e1 + e0)
        )
        return self._take_step(rhs_physical, t, neu_coeff_func, dir_coeff_func, bc, 6.0 / 11.0, gamma_components)

    def estimate_hyperviscosity_from_paper(
        self,
        *,
        velocity_scale: float,
        h: float | None = None,
        wavenumber: float | None = None,
        arpack_ncv: int | None = None,
        arpack_maxiter: int | None = None,
    ) -> HyperviscosityEstimate:
        """Estimate tau, q, and gamma_j using the componentwise paper workflow."""

        h_value = float(self.domain.get_sep_rad() if h is None else h)
        khat = float(2.0 / h_value if wavenumber is None else wavenumber)
        phase_all = khat * np.asarray(jnp.sum(self.xall, axis=1))
        f_all = np.exp(1j * phase_all)
        f_phys = f_all[: self.n]
        full_sp = StencilProperties(
            n=self.lap_stencil_properties.n,
            dim=self.dim,
            ell=self.lap_stencil_properties.ell,
            spline_degree=self.lap_stencil_properties.spline_degree,
            npoly=self.lap_stencil_properties.npoly,
            tree_mode="all",
            point_set="all",
        )
        tau_values: list[float] = []
        q_values: list[float] = []
        gamma_values: list[float] = []
        print(
            "[gamma] paper hyperviscosity estimate start: "
            f"dim={self.dim}, h={h_value:.6e}, khat={khat:.6e}, "
            f"power={self.hyperviscosity_power}, chunk_size={self.sparse_assembly_chunk_size}",
            flush=True,
        )
        for d in range(self.dim):
            props = OpProperties(selectdim=d, decompose=False, store_weights=True, record_stencils=False)
            print(f"[gamma] component {d} full-gradient assembly start", flush=True)
            full_grad = assemble_operator_sparse(
                self.domain,
                self.lap_assembler,
                self.lap_stencil,
                "grad",
                full_sp,
                props,
                chunk_size=self.sparse_assembly_chunk_size,
            )
            # Rightmost eigenvalue of the full differentiation matrix by GPU
            # Krylov-Schur Arnoldi (sparse device matvec); ARPACK only as fallback.
            from kernelpack.solvers.spectral import sparse_rightmost_eigenvalue
            lr, converged, krylov, ritz_residual = sparse_rightmost_eigenvalue(full_grad)
            if converged:
                print(f"[gamma] component {d} GPU Arnoldi: tau={lr.real:.6e} (m={krylov}, residual={ritz_residual:.1e})", flush=True)
            else:
                print(f"[gamma] component {d} GPU Arnoldi did not converge; ARPACK fallback", flush=True)
                scipy_full = _to_scipy_coo(full_grad).tocsc()
                ncv = arpack_ncv
                if ncv is not None:
                    ncv = min(max(int(ncv), 3), max(int(scipy_full.shape[0]) - 1, 3))
                maxiter = None if arpack_maxiter is None else int(arpack_maxiter)
                v0 = np.linspace(1.0, 2.0, scipy_full.shape[0], dtype=float)
                try:
                    lr = eigs(scipy_full, k=1, which="LR", ncv=ncv, maxiter=maxiter, v0=v0, return_eigenvectors=False)[0]
                except ArpackNoConvergence:
                    print(f"[gamma] component {d} ARPACK did not converge; dense fallback start", flush=True)
                    dense_full = scipy_full.toarray()
                    eig_values = eigvals(dense_full, overwrite_a=True, check_finite=False)
                    lr = eig_values[int(np.argmax(np.real(eig_values)))]
            print(f"[gamma] component {d} q/gamma calculation start", flush=True)
            tau = float(np.real(lr))
            tau_safe = max(abs(tau), np.finfo(float).tiny)
            approx = _to_scipy_coo(self.grad_sparse[d]).dot(f_all)
            exact = 1j * khat * f_phys
            residual = max(float(np.linalg.norm(exact - approx)), np.finfo(float).tiny)
            scale = max(tau_safe * float(np.linalg.norm(f_phys)), np.finfo(float).tiny)
            q = float((np.log(residual) - np.log(scale)) / np.log(khat))
            gamma = paper_gamma_component(
                h_value,
                velocity_scale,
                tau_safe,
                q,
                self.hyperviscosity_power,
                dimension=self.dim,
            )
            tau_values.append(tau)
            q_values.append(q)
            gamma_values.append(gamma)
            print(
                f"[gamma] component {d} done: tau={tau:.6e}, q={q:.6e}, gamma={gamma:.6e}",
                flush=True,
            )
        estimate = HyperviscosityEstimate(
            tau=jnp.asarray(tau_values, dtype=float),
            q=jnp.asarray(q_values, dtype=float),
            gamma_components=jnp.asarray(gamma_values, dtype=float),
            h=h_value,
            wavenumber=khat,
            power=self.hyperviscosity_power,
        )
        self.set_gamma_components(estimate.gamma_components)
        print(
            "[gamma] paper hyperviscosity estimate done: "
            f"gamma_sum={float(jnp.sum(estimate.gamma_components)):.6e}",
            flush=True,
        )
        return estimate

    def _explicit_rhs(
        self,
        t: float,
        full_state: jnp.ndarray,
        velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None,
        gamma_components: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        physical = full_state[: self.n]
        vel = _evaluate_velocity_callback(velocity, t, self.x, self.dim)
        adv = jnp.zeros((self.n,), dtype=float)
        for d in range(self.dim):
            adv = adv - vel[:, d] * self._apply_physical_operator(self.grad[d], self.grad_sparse[d], full_state)
        reaction_term = _evaluate_reaction_callback(reaction, t, physical, self.x)
        forcing_term = jnp.zeros((self.n,), dtype=float) if forcing is None else evaluate_forcing_callback(forcing, self.nu, t, self.x)
        explicit = adv + reaction_term + forcing_term
        if self.hyperviscosity_treatment == "explicit":
            explicit = explicit + self._explicit_hyperviscosity_rhs(t, full_state, gamma_components)
        return explicit

    def _take_step(
        self,
        rhs_physical: jnp.ndarray,
        t: float,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        alpha: float,
        gamma_components: jnp.ndarray | None,
    ) -> jnp.ndarray:
        gamma = self.gamma_components if gamma_components is None else self._validate_gamma(gamma_components)
        neu_coeff, dir_coeff = self._get_boundary_coefficients(t, neu_coeff_func, dir_coeff_func)
        self._ensure_boundary_operator(neu_coeff, dir_coeff)
        rhs_boundary = evaluate_transient_boundary_values(bc, neu_coeff, dir_coeff, self.nr, t, self.xb)
        if self.hyperviscosity_method == "prescribed_trace_power" and self.hyperviscosity_treatment == "implicit":
            from ._composed_adr import solve_power_step
            sol = solve_power_step(self, rhs_physical, rhs_boundary, t, alpha, gamma)
            self._push_completed_step(sol, t)
            return self.current_physical_state()
        rhs = build_implicit_rhs(rhs_physical, rhs_boundary)
        if self.linear_solver == "cudss":
            order = 1 if alpha == 1. else 2 if alpha == 2./3. else 3
            factor = get_transient_cudss_factor(self, order, gamma)
            if factor is not None:
                sol = factor.solve(rhs)
            else:
                operator = self._implicit_operator_sparse(self.nu, gamma)
                system = build_implicit_system_sparse(operator, self.bc_sparse, self.n, -alpha * self.dt)
                sol = solve_transient_cudss(self, system, rhs, order, gamma)
        elif self.linear_solver == "dense":
            implicit_operator = self._implicit_operator_dense(self.nu, gamma)
            system = build_implicit_system(implicit_operator, self.bc, self.n, -alpha * self.dt)
            sol = solve_dense_system(system, rhs)
            self.fixed_preconditioner = None
        elif self.linear_solver == "lineax":
            if self.lap_sparse is not None and (self.hyper_sparse is not None or self.hyperviscosity_treatment == "explicit") and self.bc_sparse is not None:
                sparse_operator = self._implicit_operator_sparse(self.nu, gamma)
                sparse_system = build_implicit_system_sparse(sparse_operator, self.bc_sparse, self.n, -alpha * self.dt)
                dense_system = sparse_matrix_to_dense(sparse_system)
            else:
                implicit_operator = self._implicit_operator_dense(self.nu, gamma)
                dense_system = build_implicit_system(implicit_operator, self.bc, self.n, -alpha * self.dt)
            sol = _solve_lineax_system(dense_system, rhs)
            self.fixed_preconditioner = None
        elif self.linear_solver in ("gmres", "gmres_ilu"):
            if self.lap_sparse is None or (self.hyper_sparse is None and self.hyperviscosity_treatment != "explicit") or self.bc_sparse is None:
                raise ValueError(f"{self.linear_solver} requires sparse operators")
            sparse_operator = self._implicit_operator_sparse(self.nu, gamma)
            sparse_system = build_implicit_system_sparse(sparse_operator, self.bc_sparse, self.n, -alpha * self.dt)
            if self.linear_solver == "gmres_ilu" and self.fixed_preconditioner is None:
                self.fixed_preconditioner = build_fixed_ilu_preconditioner(
                    sparse_system,
                    application=self._ilu_application(sparse_system),
                )
            sol = solve_sparse_system_gmres(
                sparse_system,
                rhs,
                preconditioner=self.fixed_preconditioner,
                tol=self.gmres_tol,
                atol=self.gmres_atol,
                restart=self.gmres_restart,
                maxiter=self.gmres_maxiter,
            )
            if self.linear_solve_diagnostics:
                relres = _sparse_relative_residual(sparse_system, sol, rhs)
                finite = jnp.all(jnp.isfinite(sol))
                should_print = jnp.logical_or(relres >= self.linear_solve_diagnostics_threshold, jnp.logical_not(finite))
                message = (
                    f"[linear] solver={self.linear_solver} "
                    "t={t:.6e} alpha={alpha:.6e} relres={relres:.6e} "
                    "rhs_norm={rhs_norm:.6e} sol_norm={sol_norm:.6e} finite={finite}"
                )

                def print_diagnostic(_):
                    debug.print(
                        message,
                        t=t,
                        alpha=alpha,
                        relres=relres,
                        rhs_norm=jnp.linalg.norm(rhs),
                        sol_norm=jnp.linalg.norm(sol),
                        finite=finite,
                        ordered=True,
                    )
                    return jnp.array(0)

                lax.cond(should_print, print_diagnostic, lambda _: jnp.array(0), operand=None)
        else:
            raise ValueError(f"unknown linear solver {self.linear_solver}")
        next_full = jnp.asarray(sol[: self.nf], dtype=float)
        self._push_completed_step(next_full, t)
        return self.current_physical_state()

    def _implicit_operator_dense(self, nu, gamma_components):
        operator = nu * self.lap
        if self.hyperviscosity_treatment == "implicit":
            operator = operator + jnp.sum(gamma_components) * self.hyper
        return operator

    def _implicit_operator_sparse(self, nu: float, gamma_components: jnp.ndarray) -> SparseCOOMatrix:
        if self.hyperviscosity_treatment == "explicit":
            if self.lap_sparse is None:
                raise ValueError("Prepare the sparse Laplacian first")
            return _scaled_sparse_sum([(nu, self.lap_sparse)])
        if self.hyperviscosity_method == "prescribed_trace_power":
            raise ValueError("L^k uses the prepared auxiliary system; it is not a separate COO hyperviscosity matrix")
        if self.lap_sparse is None or self.hyper_sparse is None:
            raise ValueError("sparse implicit operator requested before sparse assembly")
        return _scaled_sparse_sum([(nu, self.lap_sparse), (jnp.sum(gamma_components), self.hyper_sparse)])

    @staticmethod
    def _apply_physical_operator(dense: jnp.ndarray, sparse: SparseCOOMatrix | None, full_state: jnp.ndarray) -> jnp.ndarray:
        if sparse is None:
            return dense @ full_state
        return sparse_matvec(sparse.indices, sparse.values, sparse.shape[0], full_state)

    def _get_boundary_coefficients(
        self,
        t: float,
        neu_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
        dir_coeff_func: Callable[..., jnp.ndarray] | jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if is_fixed_boundary_callback(neu_coeff_func) and is_fixed_boundary_callback(dir_coeff_func):
            if not self.fixed_bc_coefficients_ready_:
                self.cached_neu_coeff_ = _ensure_unbatched_operator_input(evaluate_boundary_coefficient(neu_coeff_func, self.xb), "boundary coefficient")
                self.cached_dir_coeff_ = _ensure_unbatched_operator_input(evaluate_boundary_coefficient(dir_coeff_func, self.xb), "boundary coefficient")
                self.fixed_bc_coefficients_ready_ = True
            return self.cached_neu_coeff_, self.cached_dir_coeff_
        return (
            _ensure_unbatched_operator_input(evaluate_boundary_coefficient(neu_coeff_func, self.xb, t), "boundary coefficient"),
            _ensure_unbatched_operator_input(evaluate_boundary_coefficient(dir_coeff_func, self.xb, t), "boundary coefficient"),
        )

    def _ensure_boundary_operator(self, neu_coeff: jnp.ndarray, dir_coeff: jnp.ndarray) -> None:
        if self.fixed_bc_operator_ready_ and (self.bc.shape[0] > 0 or self.bc_sparse is not None):
            return
        if self.force_boundary_paired_ghost_stencil:
            boundary_graph = build_boundary_paired_ghost_stencil_graph(
                self.domain,
                self.bc_stencil_properties,
            )
        else:
            boundary_graph = None if self.fixed_discretization is None else self.fixed_discretization.boundary_graph
        if self.assemble_dense_operators:
            self.bc = assemble_operator(
                self.domain,
                self.bc_assembler,
                self.bc_stencil,
                "bc",
                self.bc_stencil_properties,
                self.op_properties,
                neu_coeff=neu_coeff,
                dir_coeff=dir_coeff,
                stencil_graph=boundary_graph,
            )
        else:
            self.bc = jnp.zeros((0, self.nf), dtype=float)
        self.bc_sparse = assemble_operator_sparse(
            self.domain,
            self.bc_assembler,
            self.bc_stencil,
            "bc",
            self.bc_stencil_properties,
            self.op_properties,
            neu_coeff=neu_coeff,
            dir_coeff=dir_coeff,
            stencil_graph=boundary_graph,
            chunk_size=self.sparse_assembly_chunk_size,
        )
        if self.fixed_bc_coefficients_ready_:
            self.fixed_bc_operator_ready_ = True
        if self.hyperviscosity_method == "prescribed_trace_power" and self.hyperviscosity_treatment == "implicit":
            from ._composed_adr import prepare_power_layout
            prepare_power_layout(self)
        self._prepare_explicit_hyperviscosity()
        self.fixed_preconditioner = None

    def _push_completed_step(self, next_full: jnp.ndarray, t: float) -> None:
        if self.completed_steps_ <= 0:
            self.cfnm1 = next_full
            self.tnm1 = float(t)
            self.completed_steps_ = 1
        elif self.completed_steps_ == 1:
            self.cfn = next_full
            self.tn = float(t)
            self.completed_steps_ = 2
        else:
            self.cfnm2, self.cfnm1, self.cfn = self.cfnm1, self.cfn, next_full
            self.tnm2, self.tnm1, self.tn = self.tnm1, self.tn, float(t)
            self.completed_steps_ += 1

    def _current_time(self) -> float:
        if self.completed_steps_ <= 0:
            return self.tnm2
        if self.completed_steps_ == 1:
            return self.tnm1
        return self.tn

    def _validate_gamma(self, gamma_components: jnp.ndarray) -> jnp.ndarray:
        arr = jnp.asarray(gamma_components, dtype=float).reshape(-1)
        if int(arr.shape[0]) != self.dim:
            raise ValueError(f"gamma_components must have one entry per spatial dimension ({self.dim})")
        return arr

    @staticmethod
    def _ilu_application(sparse_system: SparseCOOMatrix, *, requested: str = "dense") -> str:
        """Avoid a dense approximate inverse for large sparse systems."""

        if requested not in ("dense", "host_callback"):
            raise ValueError("ILU application must be 'dense' or 'host_callback'")
        if requested == "dense" and max(sparse_system.shape) >= 8192:
            print(
                f"[ilu] overriding dense application with host_callback for sparse system shape={sparse_system.shape}",
                flush=True,
            )
            return "host_callback"
        return requested

    @staticmethod
    def _bdf_alpha(order: int) -> float:
        if order == 1:
            return 1.0
        if order == 2:
            return 2.0 / 3.0
        if order == 3:
            return 6.0 / 11.0
        raise ValueError("bdf_order must be 1, 2, or 3")


@dataclass(frozen=True)
class StatelessEulerianAdvectionDiffusionReactionSolver:
    """Functional ADR marcher over an initialized full-state Eulerian solver.

    The base solver owns fixed RBF-FD operators, boundary operators, and the
    optional frozen ILU. This wrapper keeps BDF history in function arguments,
    which is the path to use for JIT-compiled inverse problems.
    """

    base: EulerianAdvectionDiffusionReactionSolver

    @classmethod
    def from_solver(cls, solver: EulerianAdvectionDiffusionReactionSolver, *, prepare_orders=(1, 2, 3)) -> "StatelessEulerianAdvectionDiffusionReactionSolver":
        if solver.n <= 0 or solver.nf <= 0:
            raise ValueError("StatelessEulerianAdvectionDiffusionReactionSolver requires an initialized solver")
        if solver.bc_sparse is None or (solver.assemble_dense_operators and solver.bc.shape[0] == 0):
            raise ValueError("Call prepare_boundary_operator(...) on the base solver before stateless marching")
        if solver.hyperviscosity_method == "prescribed_trace_power" and solver.hyperviscosity_treatment == "implicit":
            from ._composed_adr import prepare_power_layout
            prepare_power_layout(solver)
        solver._prepare_explicit_hyperviscosity()
        if solver.linear_solver == "gmres_ilu" and solver.fixed_preconditioner is None:
            raise ValueError("Call prepare_preconditioner(...) on the base solver before stateless gmres_ilu marching")
        if solver.linear_solver == "cudss":
            solver.prepare_cudss(orders=prepare_orders)
            return cls(copy(solver))
        return cls(solver)

    def march_to_final_full_state(
        self,
        initial_full_state: jnp.ndarray,
        *,
        steps: int,
        velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
        gamma_components: jnp.ndarray | None = None,
        t0: float = 0.0,
    ) -> jnp.ndarray:
        """Advance ``steps`` times and return the final full state."""

        if steps < 0:
            raise ValueError("steps must be nonnegative")
        u0 = _validate_full_state(initial_full_state, self.base.nf)
        gamma = self.base.gamma_components if gamma_components is None else self.base._validate_gamma(gamma_components)
        if steps == 0:
            return u0
        u1 = self.bdf1_full_state(u0, t0, t0 + self.base.dt, velocity=velocity, reaction=reaction, bc=bc, forcing=forcing, gamma_components=gamma)
        if steps == 1:
            return u1
        u2 = self.bdf2_full_state(
            u0,
            u1,
            t0,
            t0 + self.base.dt,
            t0 + 2.0 * self.base.dt,
            velocity=velocity,
            reaction=reaction,
            bc=bc,
            forcing=forcing,
            gamma_components=gamma,
        )
        if steps == 2:
            return u2

        def body(k: int, carry: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            u_nm2, u_nm1, u_n = carry
            t_nm2 = t0 + (k - 2) * self.base.dt
            t_nm1 = t0 + (k - 1) * self.base.dt
            t_n = t0 + k * self.base.dt
            t_np1 = t0 + (k + 1) * self.base.dt
            u_np1 = self.bdf3_full_state(
                u_nm2,
                u_nm1,
                u_n,
                t_nm2,
                t_nm1,
                t_n,
                t_np1,
                velocity=velocity,
                reaction=reaction,
                bc=bc,
                forcing=forcing,
                gamma_components=gamma,
            )
            return u_nm1, u_n, u_np1

        _, _, final_state = lax.fori_loop(2, steps, body, (u0, u1, u2))
        return final_state

    def march_to_final_physical_state(
        self,
        initial_full_state: jnp.ndarray,
        *,
        steps: int,
        velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
        gamma_components: jnp.ndarray | None = None,
        t0: float = 0.0,
    ) -> jnp.ndarray:
        return self.march_to_final_full_state(
            initial_full_state,
            steps=steps,
            velocity=velocity,
            reaction=reaction,
            bc=bc,
            forcing=forcing,
            gamma_components=gamma_components,
            t0=t0,
        )[: self.base.n]

    def bdf1_full_state(
        self,
        previous: jnp.ndarray,
        t_previous: float,
        t_next: float,
        *,
        velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
        gamma_components: jnp.ndarray,
    ) -> jnp.ndarray:
        explicit = self._explicit_rhs(t_previous, previous, velocity, reaction, forcing, gamma_components)
        rhs_physical = previous[: self.base.n] + self.base.dt * explicit
        return self._solve_full_state(rhs_physical, t_next, bc, 1.0, gamma_components)

    def bdf2_full_state(
        self,
        u_nm2: jnp.ndarray,
        u_nm1: jnp.ndarray,
        t_nm2: float,
        t_nm1: float,
        t_next: float,
        *,
        velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
        gamma_components: jnp.ndarray,
    ) -> jnp.ndarray:
        e0 = self._explicit_rhs(t_nm2, u_nm2, velocity, reaction, forcing, gamma_components)
        e1 = self._explicit_rhs(t_nm1, u_nm1, velocity, reaction, forcing, gamma_components)
        rhs_physical = (4.0 / 3.0) * u_nm1[: self.base.n] - (1.0 / 3.0) * u_nm2[: self.base.n] + (2.0 / 3.0) * self.base.dt * (2.0 * e1 - e0)
        return self._solve_full_state(rhs_physical, t_next, bc, 2.0 / 3.0, gamma_components)

    def bdf3_full_state(
        self,
        u_nm2: jnp.ndarray,
        u_nm1: jnp.ndarray,
        u_n: jnp.ndarray,
        t_nm2: float,
        t_nm1: float,
        t_n: float,
        t_next: float,
        *,
        velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None = None,
        gamma_components: jnp.ndarray,
    ) -> jnp.ndarray:
        e0 = self._explicit_rhs(t_nm2, u_nm2, velocity, reaction, forcing, gamma_components)
        e1 = self._explicit_rhs(t_nm1, u_nm1, velocity, reaction, forcing, gamma_components)
        e2 = self._explicit_rhs(t_n, u_n, velocity, reaction, forcing, gamma_components)
        return self._bdf3_full_state_from_explicit(
            u_nm2, u_nm1, u_n, e0, e1, e2, t_next, bc, gamma_components
        )

    def explicit_rhs(self, t, full_state, *, velocity, reaction, forcing=None, gamma_components=None):
        """Evaluate the pure explicit BDF RHS as differentiable JAX state."""
        return self._explicit_rhs(t, full_state, velocity, reaction, forcing, gamma_components)

    def bdf3_full_state_with_explicit_history(
        self, u_nm2, u_nm1, u_n, e_nm2, e_nm1, t_n, t_next, *,
        velocity, reaction, bc, forcing=None, gamma_components,
    ):
        """Advance from cached explicit history and return (u_next, e_n)."""
        e_n = self._explicit_rhs(t_n, u_n, velocity, reaction, forcing, gamma_components)
        u_next = self._bdf3_full_state_from_explicit(
            u_nm2, u_nm1, u_n, e_nm2, e_nm1, e_n, t_next, bc, gamma_components
        )
        return u_next, e_n

    def _bdf3_full_state_from_explicit(
        self, u_nm2, u_nm1, u_n, e_nm2, e_nm1, e_n, t_next, bc, gamma_components,
    ):
        rhs_physical = (
            (18.0 / 11.0) * u_n[: self.base.n]
            - (9.0 / 11.0) * u_nm1[: self.base.n]
            + (2.0 / 11.0) * u_nm2[: self.base.n]
            + (6.0 / 11.0) * self.base.dt * (3.0 * e_n - 3.0 * e_nm1 + e_nm2)
        )
        return self._solve_full_state(rhs_physical, t_next, bc, 6.0 / 11.0, gamma_components)

    def _explicit_rhs(
        self,
        t: float,
        full_state: jnp.ndarray,
        velocity: Callable[..., jnp.ndarray] | jnp.ndarray,
        reaction: Callable[..., jnp.ndarray] | jnp.ndarray,
        forcing: Callable[..., jnp.ndarray] | jnp.ndarray | None,
        gamma_components: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        physical = full_state[: self.base.n]
        vel = _evaluate_velocity_callback(velocity, t, self.base.x, self.base.dim)
        adv = jnp.zeros((self.base.n,), dtype=float)
        for d in range(self.base.dim):
            adv = adv - vel[:, d] * self.base._apply_physical_operator(self.base.grad[d], self.base.grad_sparse[d], full_state)
        reaction_term = _evaluate_reaction_callback(reaction, t, physical, self.base.x)
        forcing_term = jnp.zeros((self.base.n,), dtype=float) if forcing is None else evaluate_forcing_callback(forcing, self.base.nu, t, self.base.x)
        explicit = adv + reaction_term + forcing_term
        if self.base.hyperviscosity_treatment == "explicit":
            explicit = explicit + self.base._explicit_hyperviscosity_rhs(t, full_state, gamma_components)
        return explicit

    def _solve_full_state(
        self,
        rhs_physical: jnp.ndarray,
        t: float,
        bc: Callable[..., jnp.ndarray] | jnp.ndarray,
        alpha: float,
        gamma_components: jnp.ndarray,
    ) -> jnp.ndarray:
        rhs_boundary = evaluate_transient_boundary_values(
            bc,
            self.base.cached_neu_coeff_,
            self.base.cached_dir_coeff_,
            self.base.nr,
            t,
            self.base.xb,
        )
        if self.base.hyperviscosity_method == "prescribed_trace_power" and self.base.hyperviscosity_treatment == "implicit":
            from ._composed_adr import solve_power_step
            return solve_power_step(self.base, rhs_physical, rhs_boundary, t, alpha, gamma_components)
        rhs = build_implicit_rhs(rhs_physical, rhs_boundary)
        if self.base.linear_solver == "cudss":
            order = 1 if alpha == 1. else 2 if alpha == 2./3. else 3
            factor = get_transient_cudss_factor(self.base, order, gamma_components)
            if factor is not None:
                sol = factor.solve(rhs)
            else:
                operator = self.base._implicit_operator_sparse(self.base.nu, gamma_components)
                system = build_implicit_system_sparse(operator, self.base.bc_sparse, self.base.n, -alpha * self.base.dt)
                sol = solve_transient_cudss(self.base, system, rhs, order, gamma_components)
        elif self.base.linear_solver == "dense":
            implicit_operator = self.base._implicit_operator_dense(self.base.nu, gamma_components)
            system = build_implicit_system(implicit_operator, self.base.bc, self.base.n, -alpha * self.base.dt)
            sol = solve_dense_system(system, rhs)
        elif self.base.linear_solver == "lineax":
            if self.base.lap_sparse is not None and (self.base.hyper_sparse is not None or self.base.hyperviscosity_treatment == "explicit") and self.base.bc_sparse is not None:
                sparse_operator = self.base._implicit_operator_sparse(self.base.nu, gamma_components)
                sparse_system = build_implicit_system_sparse(sparse_operator, self.base.bc_sparse, self.base.n, -alpha * self.base.dt)
                dense_system = sparse_matrix_to_dense(sparse_system)
            else:
                implicit_operator = self.base._implicit_operator_dense(self.base.nu, gamma_components)
                dense_system = build_implicit_system(implicit_operator, self.base.bc, self.base.n, -alpha * self.base.dt)
            sol = _solve_lineax_system(dense_system, rhs)
        elif self.base.linear_solver in ("gmres", "gmres_ilu"):
            sparse_operator = self.base._implicit_operator_sparse(self.base.nu, gamma_components)
            sparse_system = build_implicit_system_sparse(sparse_operator, self.base.bc_sparse, self.base.n, -alpha * self.base.dt)
            sol = solve_sparse_system_gmres(
                sparse_system,
                rhs,
                preconditioner=self.base.fixed_preconditioner,
                tol=self.base.gmres_tol,
                atol=self.base.gmres_atol,
                restart=self.base.gmres_restart,
                maxiter=self.base.gmres_maxiter,
            )
            if self.base.linear_solve_diagnostics:
                relres = _sparse_relative_residual(sparse_system, sol, rhs)
                finite = jnp.all(jnp.isfinite(sol))
                should_print = jnp.logical_or(relres >= self.base.linear_solve_diagnostics_threshold, jnp.logical_not(finite))
                message = (
                    f"[linear] solver={self.base.linear_solver} "
                    "t={t:.6e} alpha={alpha:.6e} relres={relres:.6e} "
                    "rhs_norm={rhs_norm:.6e} sol_norm={sol_norm:.6e} finite={finite}"
                )

                def print_diagnostic(_):
                    debug.print(
                        message,
                        t=t,
                        alpha=alpha,
                        relres=relres,
                        rhs_norm=jnp.linalg.norm(rhs),
                        sol_norm=jnp.linalg.norm(sol),
                        finite=finite,
                        ordered=True,
                    )
                    return jnp.array(0)

                lax.cond(should_print, print_diagnostic, lambda _: jnp.array(0), operand=None)
        else:
            raise ValueError(f"unknown linear solver {self.base.linear_solver}")
        return jnp.asarray(sol[: self.base.nf], dtype=float)


EulerianAdvectionDiffusionReactionSolver_stateless = StatelessEulerianAdvectionDiffusionReactionSolver
