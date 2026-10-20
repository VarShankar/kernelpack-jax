"""GPU-only sparse GP fitting, validation and operator Lagrange prediction.

Numerical covariance work uses the shared cuDSS sparse Cholesky backend.
The full residual-field energy is retained, in bounded output blocks. Exact
REML derivatives stream inverse actions and retain only CSR support entries.
"""
from dataclasses import dataclass, replace
from functools import partial
import math
import logging
import time
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
from kernelpack.kernels import squared_distances, direct_squared_distances, squared_norm, safe_sqrt, wendland_c4_squared, matern52_squared
from kernelpack.domain.support import support_counts, compact_support
from kernelpack.sparse.cudss import CudssFactor, _check_solution
from kernelpack.sparse.products import csr_edge_inner_products
from .tensor_operator import TensorWendlandOperator, _normalized_svd, _training_coordinates, _mean_values, _gate
from .fastol import FastOL, TrainingNormalizer

_logger=logging.getLogger(__name__)


@dataclass(frozen=True)
class KernelTrainingConfig:
    """GPU kernel fitting options.

    ``mean_dimension=0`` omits every polynomial, including the intercept.
    Positive values include a constant plus up to that many training PCA
    coordinates. Geometry and covariance construction do not depend on this
    choice, so fixed and trainable models can share the same mean space.
    """
    mean_dimension: int = 15
    parallel_dimension: int = 3
    penalties: tuple = (0., 1.)
    training_steps: int = 60
    validation_steps: int = 35
    learning_rate: float = .03
    density_fractions: tuple = (.10, .25, .50, .75, .90)
    improvement_margin: float = 1e-6
    mode_energy_floor: float = 1e-14  # archived API; full-field path does not truncate modes
    output_block_size: int = 128
    rhs_block_size: int = 128
    assembly_bytes: int = 32*1024**2
    geometry_cache_bytes: int = 1024**3
    response_scaling: str = 'none'
    padded_training_workspace: bool = True
    compress_full_output_energy: bool = True
    output_factor_setup_bytes: int = 8*1024**3
    projected_output_bytes: int = 1024**3
    cudss_reordering: str = 'default'
    validation_start_policy: str = 'training'

    def __post_init__(self):
        object.__setattr__(self, 'penalties', tuple(self.penalties))
        object.__setattr__(self, 'density_fractions', tuple(self.density_fractions))
        if int(self.mean_dimension) != self.mean_dimension or self.mean_dimension < 0:
            raise ValueError('mean_dimension must be a nonnegative integer')
        object.__setattr__(self, 'mean_dimension', int(self.mean_dimension))
        for value in (self.parallel_dimension, self.training_steps,
                      self.validation_steps, self.output_block_size, self.rhs_block_size, self.assembly_bytes,
                      self.geometry_cache_bytes,self.output_factor_setup_bytes):
            if int(value) != value or value < 1:
                raise ValueError('dimensions, steps and block sizes must be positive integers')
        if int(self.projected_output_bytes) != self.projected_output_bytes or self.projected_output_bytes < 0:
            raise ValueError('projected_output_bytes must be a nonnegative integer')
        object.__setattr__(self, 'projected_output_bytes', int(self.projected_output_bytes))
        if self.parallel_dimension > 3 or not self.penalties or not self.density_fractions:
            raise ValueError('parallel_dimension must be <=3; candidate lists must be nonempty')
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not 0 < self.mode_energy_floor < 1 or not 0 <= self.improvement_margin < 1
                or any(not math.isfinite(x) or x < 0 for x in self.penalties)
                or any(not math.isfinite(x) or not 0 < x < 1 for x in self.density_fractions)):
            raise ValueError('invalid optimizer, penalty or density budget')
        if self.response_scaling not in ('none', 'taylor_green_ray'):
            raise ValueError('response_scaling must be none or taylor_green_ray')
        if self.cudss_reordering not in ('default', 'none', 'amd'):
            raise ValueError('cudss_reordering must be default, none or amd')
        if self.validation_start_policy not in ('training', 'common_budget'):
            raise ValueError('validation_start_policy must be training or common_budget')


@dataclass(frozen=True)
class TrainingFit:
    inputs: jax.Array
    targets: jax.Array
    training_ids: jax.Array
    normalizer: TrainingNormalizer
    center: jax.Array
    basis: jax.Array
    mean_scale: jax.Array
    P: jax.Array
    parallel_basis: jax.Array
    z: jax.Array
    perp: jax.Array
    gate: jax.Array
    delta: jax.Array
    parameters: tuple
    histories: tuple
    initial: jax.Array
    has_perp: bool
    affine_rank: int
    config: KernelTrainingConfig
    output_q: jax.Array
    residual_energy: jax.Array
    response_scale: jax.Array
    ray_direction: jax.Array | None
    ray_zero: jax.Array | None
    ray_median: jax.Array | None
    perpendicular_distances: jax.Array | None = None
    supplied_modes: jax.Array | None = None
    timings: tuple = ()
    output_factor_metadata: tuple = ()


@dataclass(frozen=True)
class KernelSelection:
    operator: object
    parameters: jax.Array
    validation_error: float
    density: float
    meets_reference: bool | None
    training_ids: jax.Array
    validation_ids: jax.Array
    candidates: tuple
    config: KernelTrainingConfig
    training_histories: tuple
    timings: tuple = ()

    def close(self):
        self.operator.close()


def _gpu_float64(value, device=None):
    if not jax.config.x64_enabled:
        raise ValueError('Trainable compact kernels require jax_enable_x64=True')
    array = jnp.asarray(value, dtype=jnp.float64, device=device)
    if array.device.platform != 'gpu':
        raise ValueError('Trainable compact kernels require GPU arrays')
    if not bool(jnp.all(jnp.isfinite(array))):
        raise ValueError('Inputs and targets must be finite')
    return array


def _ids(values, n, device):
    ids = jnp.asarray(values, device=device)
    if ids.shape != (n,) or not jnp.issubdtype(ids.dtype, jnp.integer):
        raise ValueError('Example IDs must match examples')
    ordered = jnp.sort(ids)
    if not bool(jnp.all(ordered[1:] != ordered[:-1])):
        raise ValueError('Example IDs must be unique within each split')
    return ids


def _decode(theta, q, has_perp):
    raw = jnp.concatenate([theta[2:2+q-1], jnp.zeros(1, dtype=theta.dtype)])
    a = jnp.log(2.)*jnp.tanh(raw)
    a = a-jnp.mean(a)
    return jnp.exp(theta[0]), jnp.exp(theta[1])+1e-12, a, (jnp.exp(theta[-1]) if has_perp else jnp.array(1., dtype=theta.dtype))


@partial(jax.jit, static_argnames=('has_perp', 'chunk'))
def _kernel_values(theta, z, perp2, gate, delta, rows, cols, valid_nnz, *, has_perp, chunk):
    radius, noise, a, ratio = _decode(theta, z.shape[1], has_perp)
    scaled = z*jnp.exp(a)
    size = len(rows)
    width = min(size, chunk)
    offsets = jnp.arange(width, dtype=jnp.int32)
    def step(output, nominal_start):
        # Overlap the last block rather than padding complete CSR arrays.
        # Repeated entries have identical values and are simply overwritten.
        start = jnp.minimum(nominal_start, size-width)
        positions = start+offsets
        i, j, distance = rows[positions], cols[positions], perp2[positions]
        values = (wendland_c4_squared(squared_norm(scaled[i]-scaled[j])/radius**2)
                  * matern52_squared(distance*(ratio/delta)**2)*gate[i]*gate[j])
        values = jnp.where(positions < valid_nnz, values+noise*(i == j), 0.)
        return jax.lax.dynamic_update_slice(output, values, (start,)), None
    return jax.lax.scan(step, jnp.zeros_like(perp2),
                        jnp.arange(0, size, width, dtype=jnp.int32))[0]


@partial(jax.jit, static_argnames=('has_perp', 'chunk'))
def _values_vjp(theta, z, perp2, gate, delta, rows, cols, valid_nnz, weights, *, has_perp, chunk):
    # Differentiate one support chunk before advancing the forward scan.
    # Reversing the complete kernel-values scan would retain its intermediates
    # for every CSR entry. The carry here contains only the theta gradient.
    size = len(rows)
    offsets = jnp.arange(chunk, dtype=jnp.int32)
    def step(total, start):
        remaining = size-start
        positions = start+jnp.minimum(offsets, remaining)
        safe = jnp.minimum(positions, size-1)
        i, j = rows[safe], cols[safe]
        left, right = z[i], z[j]
        distance = perp2[safe]
        left_gate, right_gate = gate[i], gate[j]
        local_weights = weights[safe]
        valid = (offsets < remaining) & (positions < valid_nnz)
        def objective(t):
            radius, noise, a, ratio = _decode(t, z.shape[1], has_perp)
            scale = jnp.exp(a)
            values = (wendland_c4_squared(squared_norm(left*scale-right*scale)/radius**2)
                      * matern52_squared(distance*(ratio/delta)**2)*left_gate*right_gate
                      + noise*(i == j))
            return jnp.sum(jnp.where(valid, local_weights*values, 0.))
        return total+jax.grad(objective)(theta), None
    return jax.lax.scan(step, jnp.zeros_like(theta),
                        jnp.arange(0, size, chunk, dtype=jnp.int32))[0]


@partial(jax.jit, static_argnames=('chunk',))
def _edge_products(left, right, rows, cols, *, chunk):
    """Retained gather oracle for the independent derivative checker.

    Production training/validation use the shared CSR-aware GPU utility.
    """
    size = len(rows)
    rr = jnp.pad(rows, (0, (-size)%chunk)).reshape(-1, chunk)
    cc = jnp.pad(cols, (0, (-size)%chunk)).reshape(-1, chunk)
    def step(_, ij):
        i, j = ij
        return None, jnp.sum(left[i]*right[j], axis=1)
    return jax.lax.scan(step, None, (rr, cc))[1].ravel()[:size]


@partial(jax.jit, static_argnames=('chunk',))
def _edge_distances(points, rows, cols, *, chunk):
    size=len(rows)
    rr=jnp.pad(rows,(0,(-size)%chunk)).reshape(-1,chunk)
    cc=jnp.pad(cols,(0,(-size)%chunk)).reshape(-1,chunk)
    return jax.lax.scan(lambda _,ij:(None,squared_norm(points[ij[0]]-points[ij[1]])),None,(rr,cc))[1].ravel()[:size]


@partial(jax.jit, static_argnames=('has_perp',))
def _soft_density(theta, z, *, has_perp):
    radius, _, a, _ = _decode(theta, z.shape[1], has_perp)
    z = z*jnp.exp(a)
    n = len(z); width = min(64, n)
    def step(total, start):
        ids = start+jnp.arange(width)
        distances = safe_sqrt(squared_distances(z[jnp.minimum(ids, n-1)], z))
        mask = (ids[:, None] < n) & (ids[:, None] != jnp.arange(n)[None, :])
        return total+jnp.sum(jnp.where(mask, jax.nn.sigmoid((1-distances/radius)/.03), 0)), None
    return jax.lax.scan(jax.checkpoint(step), jnp.asarray(0., dtype=z.dtype), jnp.arange(0, n, width))[0]/(n*(n-1))


_soft_value_grad = jax.jit(jax.value_and_grad(_soft_density), static_argnames=('has_perp',))


@jax.jit
def _residual_energy(y, response_scale, q):
    # No covariance or inverse is formed. Output columns may be dense data.
    y = y.reshape(len(y), -1)/response_scale[:, None]
    width = min(128, y.shape[1])
    blocks = jnp.pad(y, ((0, 0), (0, (-y.shape[1])%width))).reshape(len(y), -1, width)
    def step(total, block):
        residual = block-q@(q.T@block)
        return total+jnp.sum(residual**2), None
    return jax.lax.scan(step, jnp.asarray(0., dtype=y.dtype), jnp.swapaxes(blocks, 0, 1))[0]


@jax.jit
def _residual_block(y, scale, q, energy):
    block = y/scale[:, None]
    return (block-q@(q.T@block))*jnp.sqrt(len(y)/energy)


@jax.jit
def _full_energy_factor(y,scale,q,energy):
    """Exact thin QR representation of the complete output statistic.

    If E.T=Q R, then E E.T=R.T R. Retain every R column, including
    small directions; this is neither rank truncation nor mode equalization.
    Only the residual output data are QR-factored, never the covariance.
    """
    response=y.reshape(len(y),-1)/scale[:,None]
    residual=response-q@(q.T@response)
    return jnp.linalg.qr(residual.T,mode='r').T*jnp.sqrt(len(y)/energy)


@jax.jit
def _check_full_energy_factor(y,scale,q,energy,factor):
    response=y.reshape(len(y),-1)/scale[:,None]
    residual=response-q@(q.T@response)
    probes=jnp.cos(jnp.arange(len(y),dtype=y.dtype)[:,None]*
        jnp.asarray([.127,.713,1.319],dtype=y.dtype)[None,:])
    exact=(residual@(residual.T@probes))*(len(y)/energy)
    actual=factor@(factor.T@probes)
    return (jnp.linalg.norm(actual-exact)/jnp.maximum(jnp.linalg.norm(exact),jnp.finfo(y.dtype).tiny),
            jnp.abs(jnp.sum(factor**2)/len(y)-1))


@jax.jit
def _check_projected_action(result,rhs,p,correction,operator):
    # Current original-array CSR action is a dynamic pytree. Its unique zero
    # padding lives in dummy rows; native cuDSS still sees the exact prefix.
    effective_rhs=rhs-p@(correction@rhs)
    return _check_solution(result,effective_rhs,operator.native_matmul,
                           anorm=jnp.linalg.norm(operator.data))


@jax.jit
def _adam_step(theta, gradient, first, second, iteration, rate, lower, upper):
    first = .9*first+.1*gradient
    second = .999*second+.001*gradient**2
    step = rate*(first/(1-.9**iteration))/(jnp.sqrt(second/(1-.999**iteration))+1e-8)
    return jnp.clip(theta-step, lower, upper), first, second


class _InvalidCandidate(RuntimeError):
    """An explicit numerical admissibility check rejected a parameter trial."""


def _minimize_sparse(objective, initial, lower, upper, steps, rate, *, label='optimization'):
    """Host orchestration only; objectives, gradients and Adam state stay GPU."""
    current = jnp.clip(initial, lower, upper)
    first = jnp.zeros_like(current); second = jnp.zeros_like(current)
    best = current; best_value = jnp.asarray(jnp.inf, dtype=current.dtype)
    history = []
    for iteration in range(1, steps+1):
        try:
            value, gradient = objective(current)
        except _InvalidCandidate:
            value=jnp.asarray(jnp.inf,dtype=current.dtype)
            gradient=jnp.zeros_like(current)
        finite = bool(jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient)))
        history.append(jnp.stack([value, jnp.asarray(finite, dtype=current.dtype)]))
        if _logger.isEnabledFor(logging.INFO) and (iteration==1 or iteration%10==0 or iteration==steps):
            _logger.info('%s step=%d/%d objective=%.9g finite=%s',label,iteration,steps,float(value),finite)
        if not finite:
            current = best
            rate *= .5
            continue
        if bool(value < best_value):
            best, best_value = current, value
        current, first, second = _adam_step(current, gradient, first, second,
            jnp.asarray(iteration, dtype=current.dtype), jnp.asarray(rate, dtype=current.dtype), lower, upper)
    if not bool(jnp.isfinite(best_value)):
        raise RuntimeError('No finite sparse GP/validation iterate; no fallback applied')
    return best, best_value, jnp.stack(history)


class _SparseWorkspace:
    """A single checked sparse factor, updated before each eager evaluation."""
    def __init__(self, fit):
        self.fit = fit
        self.factor = None
        self.rows = self.cols = self.indptr = self.values = None
        self.chunk = max(1, fit.config.assembly_bytes//(16*fit.inputs.dtype.itemsize))
        self.perp2=None
        self._projected_panel_columns = None
        self.projected_output_metadata = None
        self.projected_output_plans = []
        self.timings = {'support_and_factor_seconds': 0., 'objective_seconds': 0., 'evaluations': 0}

    def prepare(self, theta):
        start = time.perf_counter()
        fit = self.fit
        radius, _, a, _ = _decode(theta, fit.z.shape[1], fit.has_perp)
        z = fit.z*jnp.exp(a)
        self.indptr = support_counts(z, radius)
        count = int(self.indptr[-1])
        if count < len(z) or count >= 2**31:
            raise ValueError('Invalid or overflowing compact support')
        capacity=(min(len(z)**2,1<<(count-1).bit_length()) if fit.config.padded_training_workspace else count)
        rows, cols = compact_support(z, radius, self.indptr,
            jnp.full(capacity,len(z)-1,dtype=jnp.int32,device=z.device),
            jnp.zeros(capacity, dtype=jnp.int32, device=z.device))
        self.nnz=count
        same=(self.rows is not None and rows.shape==self.rows.shape and
            bool(jnp.all(rows==self.rows)&jnp.all(cols==self.cols)))
        self.rows,self.cols=rows,cols
        if not same:
            if fit.perpendicular_distances is not None:
                self.perp2=fit.perpendicular_distances[rows,cols]
            else:
                chunk=max(1,fit.config.assembly_bytes//max(1,3*fit.perp.shape[1]*fit.inputs.dtype.itemsize))
                self.perp2=_edge_distances(fit.perp,rows,cols,chunk=min(capacity,chunk))
        self.values = _kernel_values(theta, fit.z, self.perp2, fit.gate, fit.delta, self.rows, self.cols,
            self.indptr[-1],has_perp=fit.has_perp, chunk=min(capacity, self.chunk))
        try:
            if self.factor is None:
                self.factor = CudssFactor(self.indptr, self.cols, self.values, (len(z), len(z)),
                    matrix_type='spd', mutable_values=True, row_indices=self.rows,
                    rhs_block_size=fit.config.rhs_block_size,padded_capacity=fit.config.padded_training_workspace,
                    padded_action_scratch_bytes=fit.config.assembly_bytes,
                    reordering=fit.config.cudss_reordering)
                self.factor.enable_cuda_graph()
            else:
                self.factor.update_matrix(self.indptr, self.cols, self.values, row_indices=self.rows)
        except RuntimeError as error:
            if str(error).startswith(('cuDSS factorization failed:', 'cuDSS refactor failed:')):
                raise _InvalidCandidate(str(error)) from error
            raise
        if fit.P.shape[1]:
            whitened_p = self.factor.whiten_current(fit.P)
            self.Q, self.R = jnp.linalg.qr(whitened_p, mode='reduced')
            self.correction = self.factor.whiten_current(jl.solve_triangular(self.R, self.Q.T, lower=False).T,
                backward=True).T
            reproduction = self.correction@fit.P
            if not bool(jnp.all(jnp.isfinite(reproduction)) &
                    (jnp.max(jnp.abs(reproduction-jnp.eye(fit.P.shape[1], dtype=fit.P.dtype))) <= 1e-8)):
                raise _InvalidCandidate('Sparse Cholesky-QR failed affine reproduction')
        else:
            # A zero-dimensional mean has no Schur system or native RHS.
            self.Q=jnp.empty_like(fit.P)
            self.R=jnp.empty((0,0),dtype=fit.P.dtype)
            self.correction=jnp.empty((0,len(fit.P)),dtype=fit.P.dtype)
        self.timings['support_and_factor_seconds'] += time.perf_counter()-start
        self.timings['evaluations'] += 1

    def projection(self, rhs):
        white = self.factor.whiten_current(rhs)
        residual = white-self.Q@(self.Q.T@white)
        projected=self.factor.whiten_current(residual, backward=True)
        return _check_projected_action(projected,rhs,self.fit.P,self.correction,
            self.factor.original_matrix_operator())

    def _projected_panel_width(self, columns):
        """Choose temporary checked-output storage using scalar GPU metadata.

        A panel is retained as checked blocks and then concatenated, so its
        peak estimate includes both copies. No projected numerical result is
        cached between parameter trials. A missing memory limit, disabled
        budget, or insufficient headroom retains the original block path.
        """
        fit = self.fit
        rows = len(fit.inputs)
        scalar = fit.inputs.dtype.itemsize
        block_columns = min(columns, fit.config.output_block_size)
        block_bytes = rows*block_columns*scalar
        stats = fit.inputs.device.memory_stats() or {}
        limit, used = stats.get('bytes_limit'), stats.get('bytes_in_use')
        available = (max(0, int(limit)-int(used))
                     if limit is not None and used is not None else 0)
        native = self.factor.storage_statistics()
        native_headroom = sum(int(native[name]) for name in
            ('peak_device_bytes', 'owned_csr_bytes', 'graph_dense_buffer_bytes'))
        # Reserve all native factor estimates outside JAX's allocator, two
        # supported-output buffers, the shared contraction scratch, six live
        # RHS/check blocks, and an additional 256 MiB for vendor workspace.
        # Keeping a further quarter of reported available memory unassigned
        # makes this a conservative allocation guard, not a peak measurement.
        # The current prepared map is already charged to bytes_in_use. Allow
        # another map during support replacement and the dummy-row native
        # SpMM output, in addition to the existing contraction/check reserve.
        padded_index_reserve = int(native.get('padded_action_index_bytes', 0))
        extra_action_rows = int(native.get('padded_action_extra_rows', 0))
        padded_output_reserve = extra_action_rows*block_columns*scalar
        action_headroom = (2*self.values.nbytes + fit.config.assembly_bytes
                           + padded_index_reserve + padded_output_reserve
                           + 6*block_bytes + 256*1024**2)
        peak_budget = max(0, available*3//4-native_headroom-action_headroom)
        retained_budget = min(fit.config.projected_output_bytes, peak_budget//2)
        capacity_columns = retained_budget//(rows*scalar)
        if capacity_columns >= columns:
            panel_columns = columns
        elif capacity_columns >= 2*block_columns:
            # Power-of-two counts of complete solve blocks limit new JIT
            # shapes when available headroom decreases during optimization.
            blocks = capacity_columns//block_columns
            panel_columns = block_columns*(1 << (blocks.bit_length()-1))
        else:
            panel_columns = block_columns
        if self._projected_panel_columns is not None:
            panel_columns = min(panel_columns, self._projected_panel_columns)
        self._projected_panel_columns = panel_columns
        self.projected_output_metadata = dict(
            configured_retained_bytes=fit.config.projected_output_bytes,
            reported_available_bytes=available,
            native_headroom_bytes=native_headroom,
            action_headroom_bytes=action_headroom,
            padded_action_live_index_bytes=padded_index_reserve,
            padded_action_replacement_reserve_bytes=padded_index_reserve,
            padded_action_output_reserve_bytes=padded_output_reserve,
            effective_peak_budget_bytes=peak_budget,
            panel_columns=panel_columns,
            panel_count=(columns+panel_columns-1)//panel_columns,
            retained_panel_bytes=rows*panel_columns*scalar,
            estimated_panel_peak_bytes=2*rows*panel_columns*scalar,
            estimated_peak_temporary_bytes=(2*rows*panel_columns*scalar+action_headroom),
            aggregates_blocks=panel_columns > block_columns)
        if (not self.projected_output_plans or
                panel_columns != self.projected_output_plans[-1]['panel_columns']):
            self.projected_output_plans.append(dict(self.projected_output_metadata,
                first_evaluation=self.timings['evaluations']))
        return panel_columns

    def reml(self, theta, penalty):
        start = time.perf_counter()
        self.prepare(theta)
        fit = self.fit
        dof = len(fit.inputs)-fit.P.shape[1]
        energy = jnp.asarray(0., dtype=fit.inputs.dtype)
        edge_energy = jnp.zeros_like(self.values)
        fields = fit.targets.reshape(len(fit.inputs), -1)
        source = fields if fit.supplied_modes is None else fit.supplied_modes
        operator = self.factor.original_matrix_operator()
        panel_columns = self._projected_panel_width(source.shape[1])
        for panel_start in range(0, source.shape[1], panel_columns):
            panel_stop = min(panel_start+panel_columns, source.shape[1])
            checked_blocks = []
            for column in range(panel_start, panel_stop, fit.config.output_block_size):
                raw = source[:, column:min(column+fit.config.output_block_size, panel_stop)]
                z = (_residual_block(raw, fit.response_scale, fit.output_q, fit.residual_energy)
                    if fit.supplied_modes is None else raw)
                white = self.factor.whiten_current(z)
                residual = white-self.Q@(self.Q.T@white)
                energy = energy+jnp.sum(residual**2)
                projected = self.factor.whiten_current(residual, backward=True)
                projected = _check_projected_action(projected,z,fit.P,self.correction,
                    operator)
                checked_blocks.append(projected)
            projected_panel = (checked_blocks[0] if len(checked_blocks) == 1
                               else jnp.concatenate(checked_blocks, axis=1))
            # Summing block products equals this concatenated product. Keep
            # every original block's solve check and energy contribution.
            # The native factor has already validated these same CSR buffers.
            edge_energy = edge_energy+csr_edge_inner_products(
                projected_panel,projected_panel,self.indptr,self.rows,self.cols,
                scratch_bytes=fit.config.assembly_bytes,validate=False)
            del projected_panel, checked_blocks
        inverse,factor_logdet = self.factor.inverse_on_support_and_logdet(block_size=fit.config.rhs_block_size)
        if fit.P.shape[1]:
            u = self.factor.whiten_current(self.Q, backward=True)
            edge_projection = inverse-csr_edge_inner_products(u,u,self.indptr,self.rows,self.cols,
                scratch_bytes=fit.config.assembly_bytes,validate=False)
        else:
            edge_projection = inverse
        weights = .5*(edge_projection/dof-edge_energy/energy)
        gradient = _values_vjp(theta, fit.z, self.perp2, fit.gate, fit.delta, self.rows, self.cols,
            self.indptr[-1], weights, has_perp=fit.has_perp, chunk=min(len(self.rows), self.chunk))
        logdet = factor_logdet+2*jnp.sum(jnp.log(jnp.abs(jnp.diag(self.R))))
        value = .5*(logdet/dof+jnp.log(energy/dof))
        if penalty:
            density, density_gradient = _soft_value_grad(theta, fit.z, has_perp=fit.has_perp)
            value = value+penalty*density
            gradient = gradient+penalty*density_gradient
        value.block_until_ready(); gradient.block_until_ready()
        self.timings['objective_seconds'] += time.perf_counter()-start
        return value, gradient

    def close(self):
        try:
            if self.factor is not None:
                self.factor.close()
        finally:
            self.factor = None
            self.rows = self.cols = self.indptr = self.values = self.perp2 = None
            self.Q = self.R = self.correction = None


def _ray_geometry(x, normalizer, center, parallel):
    zero = normalizer.transform(jnp.zeros((1, x.shape[1]), dtype=x.dtype, device=x.device))[0]
    normalized = normalizer.transform(x)
    direction = parallel[:, 0]
    amplitude = (normalized-zero)@direction
    sign = jnp.where(jnp.median(amplitude) < 0, -1., 1.)
    direction = direction*sign; amplitude = amplitude*sign
    median = jnp.median(amplitude)
    gap = jnp.linalg.norm(normalized-zero-amplitude[:, None]*direction)
    limit = 1e-10*jnp.maximum(jnp.linalg.norm(normalized-zero), jnp.finfo(x.dtype).tiny)
    if not bool((gap <= limit) & jnp.all(amplitude > 0) & (median > 0)):
        raise ValueError('Taylor-Green response scaling requires verified positive input-amplitude ray')
    return amplitude/median, direction, zero, median


def _query_scale(fit, queries):
    if fit.ray_direction is None:
        return jnp.ones(len(queries), dtype=fit.inputs.dtype)
    normalized = fit.normalizer.transform(queries)
    amplitude = (normalized-fit.ray_zero)@fit.ray_direction
    gap = safe_sqrt(squared_norm(normalized-fit.ray_zero-amplitude[:, None]*fit.ray_direction))
    limit = 1e-10*jnp.maximum(safe_sqrt(squared_norm(normalized-fit.ray_zero)), 1.)
    return jnp.where((gap <= limit) & (amplitude >= 0), amplitude/fit.ray_median, jnp.nan)


def train_compact_kernel(inputs, targets, *, training_ids, radius, nugget, config=None, training_modes=None,
                         optimize=True, geometry=None):
    """Prepare fixed geometry and optionally fit full-energy sparse REML.

    ``optimize=False`` returns the isotropic, unit perpendicular-ratio model
    without computing REML or output-energy factors. Validation may still
    choose radius and nugget. ``radius=None`` chooses a training-only radius
    with at most 25 percent support. A previous ``TrainingFit`` can supply
    identical ordered-input geometry; it must retain every requested mean
    coordinate (a positive-mean anchor can also supply the no-mean variant).
    """
    start = time.perf_counter()
    config = KernelTrainingConfig() if config is None else config
    x = _gpu_float64(inputs); y = _gpu_float64(targets, x.device)
    if x.ndim != 2 or len(x) < 3 or not x.shape[1] or y.ndim not in (2, 3, 4) or len(y) != len(x) or not y.size:
        raise ValueError('Nonempty training inputs/targets must share example axis')
    if (radius is not None and (not math.isfinite(radius) or radius <= 0)
            or not math.isfinite(nugget) or nugget <= 0):
        raise ValueError('Initial radius, when supplied, and nugget must be finite and positive')
    if not isinstance(optimize,bool):
        raise ValueError('optimize must be a boolean')
    ids = _ids(training_ids, len(x), x.device)
    if geometry is None:
        mean, safe, zero, normalized, center, _, vh, rank = _normalized_svd(x)
        rank = int(rank)
        if rank < 1:
            raise ValueError('Covariance shape unidentified for constant training functions')
        basis, parallel, scale, p, z, perp, delta, gate = _training_coordinates(normalized, center, vh,
            mean_dimension=min(rank, config.mean_dimension), parallel_dimension=min(rank, config.parallel_dimension),
            include_constant=config.mean_dimension>0)
        has_perp = bool(jnp.linalg.norm(perp) > 1e-10*jnp.maximum(jnp.linalg.norm(normalized-center), 1.))
        normalizer = TrainingNormalizer(mean, safe, zero)
        # Do not retain the full normalized inputs or economy right singular
        # vectors through training; the selected coordinate bases own data.
        del normalized, vh
    else:
        if not isinstance(geometry,TrainingFit):
            raise ValueError('geometry must be a previous TrainingFit')
        if (x.shape!=geometry.inputs.shape or x.dtype!=geometry.inputs.dtype or x.device!=geometry.inputs.device
                or config.parallel_dimension!=geometry.config.parallel_dimension
                or not bool(jnp.all(x==geometry.inputs)&jnp.all(ids==geometry.training_ids))):
            raise ValueError('Shared geometry requires identical ordered inputs and parallel dimension')
        rank=geometry.affine_rank
        mean_count=min(rank,config.mean_dimension)
        if mean_count>geometry.basis.shape[1]:
            raise ValueError('Shared geometry does not retain the requested mean coordinates')
        normalizer,center=geometry.normalizer,geometry.center
        basis,scale=geometry.basis[:,:mean_count],geometry.mean_scale[:mean_count]
        p=(geometry.P[:,:mean_count+1] if config.mean_dimension else
           jnp.empty((len(x),0),dtype=x.dtype,device=x.device))
        parallel,z,perp,delta,gate=(geometry.parallel_basis,geometry.z,geometry.perp,geometry.delta,geometry.gate)
        has_perp=geometry.has_perp
    if radius is None:
        budget=max(len(x),math.floor(.25*len(x)**2))
        radius=float(_radius_cap(z,jnp.asarray(budget,dtype=jnp.int32,device=x.device)))
        if not math.isfinite(radius) or radius<=0:
            raise ValueError('Training support budget cannot provide a positive initial radius')
    if p.shape[1] >= len(x):
        raise ValueError('REML requires residual degrees of freedom')
    response_scale = jnp.ones(len(x), dtype=x.dtype, device=x.device)
    ray_direction = ray_zero = ray_median = None
    if config.response_scaling == 'taylor_green_ray':
        response_scale, ray_direction, ray_zero, ray_median = _ray_geometry(x, normalizer, center, parallel)
    # Fixed-shape validation uses none of the GP output statistic. A marker
    # scalar keeps the fit schema stable without doing discarded QR/energy work.
    output_q=(jnp.linalg.qr(p,mode='reduced')[0] if optimize and p.shape[1]
              else jnp.empty((len(x),0),dtype=x.dtype,device=x.device))
    energy=(_residual_energy(y,response_scale,output_q) if optimize
            else jnp.asarray(1.,dtype=x.dtype,device=x.device))
    if optimize and not bool(jnp.isfinite(energy) & (energy > jnp.finfo(y.dtype).tiny)):
        raise ValueError('No residual output energy; covariance is unidentified')
    modes = None
    factor_metadata={'kind':'physical_output_blocks','estimated_qr_setup_bytes':0,
        'qr_setup_budget_bytes':config.output_factor_setup_bytes}
    if not optimize:
        if training_modes is not None:
            raise ValueError('A fixed-shape model does not use training_modes')
        factor_metadata['kind']='not_computed_fixed_shape'
    elif training_modes is not None:
        modes = _gpu_float64(training_modes, x.device)
        if modes.ndim != 2 or len(modes) != len(x) or not modes.shape[1]:
            raise ValueError('Supplied Z factor must be a nonempty training-only matrix')
        factor_metadata['kind']='supplied_full_energy_factor'
    elif config.compress_full_output_energy and y.size//len(y)>2*len(y):
        # Reserve space for responses/residuals, GEQRF input/work and the full
        # factor. This is an allocation estimate, not a measured peak claim.
        estimate=3*y.nbytes+2*len(y)**2*y.dtype.itemsize
        stats=x.device.memory_stats() or {}
        available=max(0,(stats.get('bytes_limit') or 2**63)-(stats.get('bytes_in_use') or 0))
        budget=min(config.output_factor_setup_bytes,available*3//4)
        factor_metadata.update(estimated_qr_setup_bytes=estimate,effective_setup_budget_bytes=budget)
        if estimate<=budget:
            modes=_full_energy_factor(y,response_scale,output_q,energy)
            gram_gap,energy_gap=_check_full_energy_factor(y,response_scale,output_q,energy,modes)
            if not bool(jnp.all(jnp.isfinite(modes))&(gram_gap<=5e-12)&(energy_gap<=5e-12)):
                raise RuntimeError('Full residual-energy QR failed its Gram/energy identity checks')
            factor_metadata.update(kind='full_energy_qr',rhs_columns=modes.shape[1],
                gram_action_relative_gap=float(gram_gap),trace_relative_gap=float(energy_gap))
            _logger.info('full-energy QR outputs=%d rhs=%d Gram_gap=%.3g energy_gap=%.3g',
                y.size//len(y),modes.shape[1],float(gram_gap),float(energy_gap))
        else:
            factor_metadata['reason']='QR setup estimate exceeds explicit/available GPU budget; exact GPU output blocks retained'
            _logger.info('full-energy QR exceeds GPU setup budget; retaining exact GPU output blocks')
    initial = jnp.concatenate([jnp.log(jnp.array([radius, max(nugget, 1e-12)], device=x.device)),
        jnp.zeros(z.shape[1]-1+int(has_perp), dtype=x.dtype, device=x.device)])
    lower = jnp.concatenate([jnp.log(jnp.array([radius/8, 1e-12], device=x.device)),
        jnp.full(z.shape[1]-1, -2., device=x.device),
        jnp.full(1, -math.log(4.), device=x.device) if has_perp else jnp.zeros(0, device=x.device)])
    upper = jnp.concatenate([jnp.log(jnp.array([radius*4, .1], device=x.device)),
        jnp.full(z.shape[1]-1, 2., device=x.device),
        jnp.full(1, math.log(4.), device=x.device) if has_perp else jnp.zeros(0, device=x.device)])
    # A geometry-only cache saves repeated high-dimensional differences. It is
    # capped independently of sparse covariance storage and is released after
    # selection. Above budget, compute only current support edges instead.
    distances=((geometry.perpendicular_distances
                if geometry is not None and geometry.perpendicular_distances is not None
                else direct_squared_distances(perp,perp))
        if len(x)**2*x.dtype.itemsize<=config.geometry_cache_bytes else None)
    fit = TrainingFit(x, y, ids, normalizer, center, basis, scale, p, parallel, z, perp, gate, delta,
        (), (), initial, has_perp, rank, config, output_q, energy, response_scale,
        ray_direction, ray_zero, ray_median, distances, modes)
    if not optimize:
        return replace(fit,parameters=(initial,),histories=(),
            timings=(('total_training_seconds',time.perf_counter()-start),('training_steps_executed',0)),
            output_factor_metadata=tuple(factor_metadata.items()))
    workspace = _SparseWorkspace(fit)
    parameters = []; histories = []
    try:
        for penalty in config.penalties:
            theta, _, history = _minimize_sparse(lambda t: workspace.reml(t, penalty), initial, lower, upper,
                config.training_steps, config.learning_rate,label=f'training penalty={penalty:g}')
            parameters.append(theta); histories.append(history)
        timings = tuple(workspace.timings.items())+(
            ('total_training_seconds', time.perf_counter()-start),
            ('training_steps_executed',config.training_steps*len(config.penalties)),
            ('projected_output_panels', tuple(tuple(plan.items())
                                            for plan in workspace.projected_output_plans)))
    finally:
        workspace.close()
    return replace(fit, parameters=tuple(parameters), histories=tuple(histories), timings=timings,
        output_factor_metadata=tuple(factor_metadata.items()))


def _relative_field_error(prediction, truth):
    axes = (1, 3) if truth.ndim == 4 else tuple(range(1, truth.ndim))
    denominator = safe_sqrt(jnp.sum(truth**2, axis=axes))
    numerator = safe_sqrt(jnp.sum((prediction-truth)**2, axis=axes))
    return jnp.mean(numerator/jnp.where(denominator == 0, 1., denominator))


@partial(jax.jit, static_argnames=('has_perp',))
def _cross_covariance(theta, z, zq, perp2, gate, gateq, delta, *, has_perp):
    radius, _, a, ratio = _decode(theta, z.shape[1], has_perp)
    return (wendland_c4_squared(squared_distances(zq*jnp.exp(a), z*jnp.exp(a))/radius**2)
        * matern52_squared(perp2*(ratio/delta)**2)*gateq[:, None]*gate[None, :])


@partial(jax.jit, static_argnames=('has_perp',))
def _cross_vjp(theta, z, zq, perp2, gate, gateq, delta, cotangent, *, has_perp):
    return jax.grad(lambda t: jnp.vdot(cotangent, _cross_covariance(t, z, zq, perp2, gate,
        gateq, delta, has_perp=has_perp)))(theta)


@partial(jax.jit, static_argnames=('iterations',))
def _radius_cap(z, budget, *, iterations=48):
    # Monotone bisection of exact support counts; no N x N distance allocation.
    upper = 2*jnp.max(jnp.linalg.norm(z, axis=1))+1.
    def step(bounds, _):
        low, high = bounds; middle = (low+high)/2
        count = support_counts(z, middle)[-1]
        feasible = (count >= 0) & (count <= budget)
        return (jnp.where(feasible, middle, low), jnp.where(feasible, high, middle)), None
    bounds = jax.lax.scan(step, (jnp.asarray(0., dtype=z.dtype), upper), None, length=iterations)[0]
    return bounds[0]*(1-1e-12)


class _ScaledOperator:
    """Physical Lagrange map for the verified input-amplitude response model."""
    def __init__(self, operator, fit):
        self.operator = operator
        self.response_scale = fit.response_scale
        self.ray_direction = fit.ray_direction
        self.ray_zero = fit.ray_zero
        self.ray_median = fit.ray_median
    def __getattr__(self, name):
        return getattr(self.operator, name)
    def device_state(self):
        return dict(operator=self.operator.device_state(),response_scale=self.response_scale,
            ray_direction=self.ray_direction,ray_zero=self.ray_zero,ray_median=self.ray_median)
    def lagrange_map(self, queries, *, state=None):
        state=self.device_state() if state is None else state
        q = jnp.asarray(queries, dtype=self.inputs.dtype, device=self.inputs.device)
        leading = q.shape[:-1]; flat = q.reshape(-1, q.shape[-1])
        base=state['operator']
        normalized=jnp.where(base['zero'],0.,(flat-base['mean'])/base['std_safe'])
        amplitude = (normalized-state['ray_zero'])@state['ray_direction']
        gap = safe_sqrt(squared_norm(normalized-state['ray_zero']-amplitude[:, None]*state['ray_direction']))
        valid = (gap <= 1e-10*jnp.maximum(safe_sqrt(squared_norm(normalized-state['ray_zero'])), 1.)) & (amplitude >= 0)
        scale = jnp.where(valid, amplitude/state['ray_median'], jnp.nan)
        return (scale[:, None]*self.operator.lagrange_map(flat,state=base)/state['response_scale'][None, :]).reshape(leading+(len(self.inputs),))
    def prepare_coefficients(self, targets):
        raise ValueError('Response-scaled operator requires Lagrange evaluation')
    def close(self):
        self.operator.close()


def freeze_compact_operator(fit,theta,*,rhs_block_size=None,profile_setup=True):
    """Build one selected sparse Lagrange operator from its exact fit geometry.

    This performs no parameter selection and does not inspect validation or
    test outputs. It also retains the explicitly configured response scaling.
    Callers can close selection-time factors before rebuilding several final
    models one at a time, avoiding overlapping temporary factor storage.
    """
    theta=_gpu_float64(theta,fit.inputs.device)
    if theta.shape!=fit.initial.shape:
        raise ValueError('Frozen parameters must match the fit parameter shape')
    radius,noise,a,ratio=_decode(theta,fit.z.shape[1],fit.has_perp)
    operator=TensorWendlandOperator(fit.inputs,radius=float(radius),nugget=float(noise),
        mean_dimension=fit.config.mean_dimension,parallel_dimension=fit.config.parallel_dimension,
        perpendicular_ratio=float(ratio),log_parallel_scales=a,backend='cudss',
        rhs_block_size=fit.config.rhs_block_size if rhs_block_size is None else rhs_block_size,
        cudss_reordering=fit.config.cudss_reordering,profile_setup=profile_setup,_geometry=fit)
    return _ScaledOperator(operator,fit) if fit.ray_direction is not None else operator


def calibrate_compact_kernel(fit, validation_inputs, validation_targets, *, validation_ids,
        reference_density, reference_error=None, policy='accuracy', reconstruct=None, backend='cudss', rhs_block_size=128):
    """Validate physical fields with sparse Lagrange maps; tune radius/nugget only."""
    start = time.perf_counter()
    if backend != 'cudss':
        raise ValueError('Training/validation require sparse cuDSS; no dense backend')
    if (not math.isfinite(reference_density) or not 0 < reference_density <= 1 or
            reference_error is not None and (not math.isfinite(reference_error) or reference_error <= 0)):
        raise ValueError('Invalid training density or validation error reference')
    if policy not in ('accuracy', 'compression') or policy == 'compression' and reference_error is None:
        raise ValueError('Compression selection requires a validation reference error')
    x = _gpu_float64(validation_inputs, fit.inputs.device)
    truth = _gpu_float64(validation_targets, fit.inputs.device)
    if x.ndim != 2 or x.shape[1] != fit.inputs.shape[1] or not len(x) or len(truth) != len(x):
        raise ValueError('Validation example axes must match')
    ids = _ids(validation_ids, len(x), x.device)
    ordered = jnp.sort(fit.training_ids); positions = jnp.searchsorted(ordered, ids)
    if bool(jnp.any((positions < len(ordered)) & (ordered[jnp.minimum(positions, len(ordered)-1)] == ids))):
        raise ValueError('Training and validation IDs overlap')
    centered = fit.normalizer.transform(x)-fit.center
    zq = centered@fit.parallel_basis; perpq = centered-zq@fit.parallel_basis.T
    gateq = _gate(safe_sqrt(squared_norm(perpq))/fit.delta)
    perpendicular_cross=direct_squared_distances(perpq,fit.perp)
    pq = _mean_values(centered,fit.basis,fit.mean_scale,include_constant=bool(fit.P.shape[1]))
    query_scale = _query_scale(fit, x)
    if not bool(jnp.all(jnp.isfinite(query_scale))):
        raise ValueError('Validation inputs violate the response scaling model')
    flat = fit.targets.reshape(len(fit.inputs), -1)
    shape = fit.targets.shape[1:]
    reconstruction_buffers=() if reconstruct is None else reconstruct.buffers
    def predict_fields(h,flat,query_scale,response_scale,reconstruction_buffers):
        physical = query_scale[:, None]*h/response_scale[None, :]
        fields = (physical@flat).reshape((len(x),)+shape)
        fields = reconstruct.compiled(fields,reconstruction_buffers) if reconstruct is not None else fields
        if fields.shape != truth.shape:
            raise ValueError('Reconstructed validation fields must match truth')
        return fields
    def field_loss(h,flat,query_scale,response_scale,truth,reconstruction_buffers):
        return _relative_field_error(predict_fields(h,flat,query_scale,response_scale,reconstruction_buffers), truth)
    field_value_grad = jax.jit(jax.value_and_grad(field_loss))
    field_value = jax.jit(field_loss)
    workspace = _SparseWorkspace(fit)
    records = []; eligible = []
    def evaluate(theta, gradient=False, return_fields=False):
        workspace.prepare(theta)
        cross = _cross_covariance(theta, fit.z, zq, perpendicular_cross, fit.gate, gateq, fit.delta, has_perp=fit.has_perp)
        ordinary = workspace.factor.solve_current(cross.T).T
        if fit.P.shape[1]:
            h = ordinary+(pq-ordinary@fit.P)@workspace.correction
            residual = h@fit.P-pq
            h = jnp.where((jnp.max(jnp.abs(residual), axis=1) <= 1e-8*(1+jnp.max(jnp.abs(pq), axis=1)))[:, None], h, jnp.nan)
        else:
            h = ordinary
        if return_fields:
            return predict_fields(h,flat,query_scale,fit.response_scale,reconstruction_buffers)
        if not gradient:
            return field_value(h,flat,query_scale,fit.response_scale,truth,reconstruction_buffers)
        value, cotangent = field_value_grad(h,flat,query_scale,fit.response_scale,truth,reconstruction_buffers)
        v = workspace.projection(cotangent.T).T
        edge = -csr_edge_inner_products(h.T,v.T,workspace.indptr,workspace.rows,workspace.cols,
            scratch_bytes=fit.config.assembly_bytes,validate=False)
        derivative = _values_vjp(theta, fit.z, workspace.perp2, fit.gate, fit.delta, workspace.rows, workspace.cols,
            workspace.indptr[-1], edge, has_perp=fit.has_perp, chunk=min(len(workspace.rows), workspace.chunk))
        derivative = derivative+_cross_vjp(theta, fit.z, zq, perpendicular_cross, fit.gate, gateq,
            fit.delta, v, has_perp=fit.has_perp)
        value.block_until_ready(); derivative.block_until_ready()
        return value, derivative
    max_density = max(fit.config.density_fractions)*reference_density
    def record(theta, origin, penalty, history=None):
        try:
            score = evaluate(theta)
        except _InvalidCandidate:
            score=jnp.asarray(jnp.inf,dtype=x.dtype)
        density = workspace.nnz/len(fit.inputs)**2
        row = dict(parameters=theta, origin=origin, training_penalty=penalty, validation_error=score,
            density=density, history=history)
        records.append(row)
        if bool(jnp.isfinite(score)) and density <= max_density and density < reference_density*(1-fit.config.improvement_margin):
            eligible.append(row)
    try:
        for penalty, theta in zip(fit.config.penalties, fit.parameters):
            _, _, a, _ = _decode(theta, fit.z.shape[1], fit.has_perp)
            caps = []
            for fraction in fit.config.density_fractions:
                budget = math.floor(fraction*reference_density*len(fit.inputs)**2)
                if budget < len(fit.inputs):
                    continue
                cap = _radius_cap(fit.z*jnp.exp(a), jnp.asarray(budget, dtype=jnp.int32))
                if not bool(jnp.isfinite(cap) & (cap > 0)):
                    continue
                caps.append(cap)
                record(theta.at[0].set(jnp.log(cap)), 'training_distance_budget', penalty)
            if not caps:
                continue
            cap = jnp.max(jnp.stack(caps))
            lower = jnp.log(jnp.stack([cap/8, jnp.asarray(1e-12, dtype=x.dtype)]))
            upper = jnp.log(jnp.stack([cap, jnp.asarray(.1, dtype=x.dtype)]))
            radius_starts=((jnp.log(cap/2),jnp.log(cap/4))
                if fit.config.validation_start_policy=='common_budget' else (theta[0],fit.initial[0]))
            noise_starts=((jnp.log(jnp.asarray(1e-6,dtype=x.dtype,device=x.device)),
                           jnp.log(jnp.asarray(1e-3,dtype=x.dtype,device=x.device)))
                if fit.config.validation_start_policy=='common_budget' else (theta[1],fit.initial[1]))
            for radius_start in radius_starts:
                for noise_start in noise_starts:
                    initial = jnp.stack([radius_start, noise_start])
                    def objective(values):
                        value, gradient = evaluate(theta.at[:2].set(values), gradient=True)
                        return value, gradient[:2]
                    best, _, history = _minimize_sparse(objective, initial, lower, upper,
                        fit.config.validation_steps, fit.config.learning_rate,
                        label=f'validation penalty={penalty:g}')
                    record(theta.at[:2].set(best), 'validation_polish', penalty, history)
        if policy == 'compression':
            eligible = [row for row in eligible if bool(row['validation_error'] < reference_error*(1-fit.config.improvement_margin))]
        if not eligible:
            raise RuntimeError('No finite candidate meets validation/density criteria')
        key = ((lambda row: (float(row['density']), float(row['validation_error']))) if policy == 'compression'
            else (lambda row: (float(row['validation_error']), float(row['density']))))
        winner = min(eligible, key=key)
        theta = winner['parameters']
        # Retain only the independently computed physical fields across the
        # handoff. The calibration factor/CSR must not overlap frozen setup.
        expected_fields = evaluate(theta,return_fields=True).block_until_ready()
        workspace.close()
        final_setup_started=time.perf_counter()
        operator = freeze_compact_operator(fit,theta,rhs_block_size=rhs_block_size)
        final_setup_seconds=time.perf_counter()-final_setup_started
        try:
            h = operator.lagrange_map(x)
            fields = (h@flat).reshape((len(x),)+shape)
            fields = reconstruct(fields) if reconstruct is not None else fields
            actual = _relative_field_error(fields, truth)
            field_gap=jnp.linalg.norm(fields-expected_fields)
            if not bool(jnp.isfinite(actual) & jnp.all(jnp.isfinite(expected_fields)) &
                    (field_gap <= 1e-8*(1+jnp.linalg.norm(expected_fields)))):
                raise RuntimeError('Frozen sparse operator disagrees with selected validation fields')
            density = operator.kernel_nnz/len(fit.inputs)**2
            if density > max_density or density >= reference_density*(1-fit.config.improvement_margin):
                raise RuntimeError('Frozen kernel violates training support budget')
            meets = None if reference_error is None else bool(actual < reference_error*(1-fit.config.improvement_margin))
            if policy == 'compression' and not meets:
                raise RuntimeError('Frozen operator fails validation accuracy criterion')
            timings = tuple(workspace.timings.items())+(('total_validation_seconds', time.perf_counter()-start),
                ('frozen_operator_setup_seconds',final_setup_seconds),('frozen_operator_setup_components',tuple(operator.setup_timings.items())))
            return KernelSelection(operator, theta, float(actual), density, meets, fit.training_ids, ids,
                tuple(records), fit.config, fit.histories, timings)
        except BaseException:
            operator.close()
            raise
    finally:
        workspace.close()


def train_validate_fastol(inputs, points, velocity, validation_inputs, validation_truth, *,
        training_ids, validation_ids, radius, nugget, reference_density, reference_error=None,
        config=None, spatial_queries=None, spatial_center_indices=None, policy='accuracy',
        backend='cudss', rhs_block_size=128, training_modes=None, spatial_reconstructor=None,
        optimize=True,geometry=None,**spatial_options):
    """Entire numerical workflow runs on GPU; no test targets enter this API."""
    if backend != 'cudss':
        raise ValueError('FastOL training requires sparse cuDSS')
    fit = train_compact_kernel(inputs, velocity, training_ids=training_ids, radius=radius, nugget=nugget,
        config=config, training_modes=training_modes,optimize=optimize,geometry=geometry)
    # Full-energy factors accelerate REML only. Validation uses the original
    # physical targets; retain the factor's checked metadata without its buffer.
    fit = replace(fit, supplied_modes=None)
    reconstruction = FastOL.prepare_spatial_reconstructor(points, fit.targets,
        spatial_queries=spatial_queries, spatial_center_indices=spatial_center_indices,
        spatial_reconstructor=spatial_reconstructor, **spatial_options)
    selection = calibrate_compact_kernel(fit, validation_inputs, validation_truth, validation_ids=validation_ids,
        reference_density=reference_density, reference_error=reference_error, policy=policy,
        reconstruct=reconstruction, backend=backend, rhs_block_size=rhs_block_size)
    selection=replace(selection,timings=(('training',fit.timings),('validation',selection.timings),
        ('output_energy_factor',fit.output_factor_metadata)))
    try:
        model = FastOL(fit.inputs, points, fit.targets, epsilon=1., operator_model=selection.operator,
            backend='cudss', spatial_reconstructor=reconstruction, **spatial_options)
    except BaseException:
        selection.close()
        raise
    return model, selection
