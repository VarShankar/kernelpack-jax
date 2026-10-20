"""Full local DF-PHS/polynomial + Wendland operator-learning workflow."""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from functools import partial
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jl
from .local_dfphs import LocalDFPHSDecoder
from kernelpack.kernels import squared_distances, squared_norm, wendland_c4_squared
from kernelpack.domain.support import support_counts, compact_support


@jax.jit
def _wendland_pairs(x, rows, columns, epsilon):
    return 3 * wendland_c4_squared(epsilon**2 * squared_norm(x[rows]-x[columns]))


@partial(jax.jit,static_argnames=("chunk",))
def _assemble_wendland_support(x,rows,columns,epsilon,nugget,*,chunk):
    count=len(rows)
    padding=(-count)%chunk
    rr=jnp.pad(rows,(0,padding)).reshape(-1,chunk)
    cc=jnp.pad(columns,(0,padding)).reshape(-1,chunk)
    def step(_,pair):
        i,j=pair
        values=_wendland_pairs(x,i,j,epsilon)+nugget*(i==j).astype(x.dtype)
        return None,values
    return jax.lax.scan(step,None,(rr,cc))[1].ravel()[:count]


def _sparse_training_kernel(x, epsilon, nugget, assembly_batch_bytes):
    """Device radius topology and values; only the allocation count is read."""
    radius=jnp.asarray(1./epsilon,dtype=x.dtype)
    indptr=support_counts(x,radius)
    count=int(indptr[-1])
    if count < 1 or count >= 2**31 or len(x) >= 2**31:
        raise ValueError("Sparse operator exceeds cuDSS int32 CSR limits")
    rows,columns=compact_support(x,radius,indptr,jnp.zeros(count,dtype=jnp.int32,device=x.device),
                                jnp.zeros(count,dtype=jnp.int32,device=x.device))
    width = min(len(columns), max(1, int(assembly_batch_bytes) // (3 * x.shape[1] * x.dtype.itemsize)))
    values=_assemble_wendland_support(x,rows,columns,epsilon,nugget,chunk=width)
    return indptr, columns, values


@jax.jit
def wendland_c4_matrix(x, y, epsilon):
    """Ram's phi_3,2; PD guarantee requires affine input dimension <=3.

    Direct differences use bounded storage and analytic coincidence limits.
    """
    return 3 * wendland_c4_squared(epsilon**2 * squared_distances(x,y))


@dataclass
class TrainingNormalizer:
    mean: jax.Array
    std_safe: jax.Array
    zero: jax.Array

    @classmethod
    def fit(cls, x):
        x = jnp.asarray(x)
        mean = jnp.mean(x, axis=0)
        sigma = jnp.std(x, axis=0, ddof=0)
        return cls(mean, jnp.where(sigma == 0, 1.0, sigma), sigma == 0)

    def transform(self, x):
        return jnp.where(self.zero, 0.0, (jnp.asarray(x) - self.mean) / self.std_safe)


@dataclass
class PreparedGPUCallable:
    """A compiled callable whose numerical GPU buffers are explicit inputs.

    Keeping buffers outside the compiled function's closure prevents JAX from
    treating resident device data as host constants during a later compilation.
    Nested compiled callers must pass ``buffers`` to ``compiled`` explicitly.
    """
    compiled: object
    buffers: object
    spatial_mode: str | None = None
    precomputed_bytes: int = 0
    operator_evaluation: str | None = None

    def __call__(self, values):
        return self.compiled(values, self.buffers)

    @property
    def buffers_sizes(self):
        return tuple(int(value.nbytes) for value in jax.tree.leaves(self.buffers)
                     if isinstance(value, jax.Array))


def _spatial_identity(value):
    if isinstance(value,(tuple,list)):
        return tuple(_spatial_identity(item) for item in value)
    if value is None:
        return None
    return (id(value),tuple(value.shape),str(value.dtype),str(value.device))


def _same_spatial_identity(left,right):
    if isinstance(left,(tuple,list)) and isinstance(right,(tuple,list)):
        return len(left)==len(right) and all(_same_spatial_identity(a,b) for a,b in zip(left,right))
    return left is right


def _spatial_source_token(source,snapshot,default):
    """Freeze parameter metadata; never inspect numerical device contents."""
    if source is None:
        return default
    if isinstance(source,jax.Array):
        return ("array",_spatial_identity(source if source.device.platform=="gpu" else snapshot))
    if isinstance(source,(bool,int,float,str)):
        return ("literal",source)
    if isinstance(source,(tuple,list)):
        def literal(value):
            if isinstance(value,(bool,int,float,str)) or value is None:
                return value
            if isinstance(value,(tuple,list)):
                return tuple(literal(item) for item in value)
            raise TypeError
        try:
            return ("literal",literal(source))
        except TypeError:
            pass
    # Mutable host arrays are represented by their private immutable GPU copy.
    # A second construction gets a different copy and cannot reuse this map.
    return ("snapshot",_spatial_identity(snapshot))


@dataclass
class PreparedSpatialReconstructor(PreparedGPUCallable):
    """Geometry-only spatial map, independent of operator factors and targets."""
    signature: tuple = ()
    keepalive: tuple = field(default_factory=tuple,repr=False)
    output_shape: tuple = ()
    setup_bytes: int = 0
    _precomputed_training: object = field(default=None,repr=False)

    @property
    def method(self):
        return self.signature[4]

    def precompute_fields(self,fields,*,budget,force=False):
        """Retain at most one exact R(Y), subject to the caller's existing cap."""
        if not isinstance(fields,jax.Array) or fields.device.platform!="gpu":
            raise ValueError("Shared spatial fields must be immutable GPU arrays")
        if (str(fields.device)!=self.signature[5] or str(fields.dtype)!=self.signature[6]
                or tuple(fields.shape[1:])!=self.signature[7]):
            raise ValueError("Shared spatial fields must match the prepared device, precision and saved-time layout")
        required=len(fields)*math.prod(self.output_shape)*jax.tree.leaves(self.buffers)[0].dtype.itemsize
        if not force and required>budget:
            return None
        previous=self._precomputed_training
        if previous is not None and previous[0] is fields:
            return previous[1]
        prepared=self(fields)
        prepared.block_until_ready()
        self._precomputed_training=(fields,prepared)
        return prepared

    def clear_precomputed(self):
        self._precomputed_training=None


class _DenseFactor:
    def __init__(self, k):
        self.lower = jnp.linalg.cholesky(k)
        if not bool(jnp.all(jnp.isfinite(self.lower))):
            raise RuntimeError(
                "Operator Cholesky failed; no hidden nugget or fallback was applied"
            )

    def solve(self, b):
        return jl.solve_triangular(
            self.lower.T, jl.solve_triangular(self.lower, b, lower=True), lower=False
        )

    def close(self):
        pass


class FastOL:
    """Fit the operator once; stream local spatial fits and reconstruction.

    Epsilon and nugget are explicit. Any selection must use validation data,
    never test targets. Inputs are (train,features), output velocity is
    (train,points,dim) or (train,points,time,dim). Prediction returns
    (query,spatial,dim) or (query,spatial,time,dim).

    Spatial queries are evaluated with their own assigned nearest stencil;
    this preserves divergence within that stencil, not across stencil changes.
    Geometry setup is eager; all numerical fits/solves run through JAX/cuDSS.
    """

    def __init__(
        self,
        inputs,
        points,
        velocity,
        *,
        epsilon,
        nugget=0.0,
        backend="cudss",
        spatial_batch_size=16,
        rhs_block_size=128,
        cudss_reordering=None,
        sparse_assembly_batch_bytes=64 * 1024 * 1024,
        spatial_setup_bytes=64 * 1024 * 1024,
        max_inflight_spatial_blocks=4,
        operator_model=None,
        spatial_reconstructor=None,
        **decoder_options,
    ):
        x = jnp.asarray(inputs)
        if x.device.platform!="gpu":
            raise ValueError("FastOL requires GPU arrays; no CPU fallback is enabled")
        v = jnp.asarray(velocity, dtype=x.dtype,device=x.device)
        if (
            x.ndim != 2
            or not len(x)
            or x.dtype not in (jnp.float32, jnp.float64)
            or not bool(jnp.all(jnp.isfinite(x)))
        ):
            raise ValueError("inputs must be a nonempty finite floating matrix")
        if v.ndim not in (3, 4) or v.shape[0] != len(x):
            raise ValueError("velocity training axis must match inputs")
        if (
            not math.isfinite(epsilon)
            or epsilon <= 0
            or not math.isfinite(nugget)
            or nugget < 0
        ):
            raise ValueError("epsilon must be positive and nugget nonnegative")
        if backend not in ("dense", "cudss"):
            raise ValueError("backend must be 'dense' or 'cudss'")
        if cudss_reordering is not None and cudss_reordering not in ("default", "none", "amd"):
            raise ValueError("cudss_reordering must be default, none or amd")
        inherited_reordering = getattr(operator_model, "cudss_reordering", "default")
        if operator_model is not None and cudss_reordering is not None and cudss_reordering != inherited_reordering:
            raise ValueError("cudss_reordering must match operator_model.cudss_reordering")
        self.cudss_reordering = inherited_reordering if cudss_reordering is None else cudss_reordering
        if int(spatial_batch_size) != spatial_batch_size or spatial_batch_size < 1:
            raise ValueError("spatial_batch_size must be a positive integer")
        self.normalizer = operator_model.normalizer if operator_model is not None else TrainingNormalizer.fit(x)
        for label, value in (("rhs_block_size", rhs_block_size),
                             ("sparse_assembly_batch_bytes", sparse_assembly_batch_bytes),
                             ("spatial_setup_bytes", spatial_setup_bytes),
                             ("max_inflight_spatial_blocks", max_inflight_spatial_blocks)):
            if int(value) != value or value < 1:
                raise ValueError(label + " must be a positive integer")
        self.inputs = operator_model.inputs if operator_model is not None else self.normalizer.transform(x)
        self.velocity = v
        self.epsilon = float(epsilon)
        self.nugget = float(nugget)
        self.backend = backend
        self.decoder = LocalDFPHSDecoder(jnp.asarray(points,device=x.device), **decoder_options)
        self._configure_spatial_sources(points,decoder_options)
        if (
            v.shape[1] != len(self.decoder.points)
            or v.shape[-1] != self.decoder.dimension
        ):
            raise ValueError("velocity spatial shape must match points")
        self.spatial_batch_size = int(spatial_batch_size)
        self.spatial_setup_bytes = int(spatial_setup_bytes)
        self.max_inflight_spatial_blocks = int(max_inflight_spatial_blocks)
        self._shared_spatial_reconstructor=spatial_reconstructor
        if spatial_reconstructor is not None and not isinstance(spatial_reconstructor,PreparedSpatialReconstructor):
            raise TypeError("spatial_reconstructor must be a prepared geometry-only spatial map")
        if spatial_reconstructor is not None:
            self._validate_spatial_reconstructor(spatial_reconstructor,method="weights",geometry_only=True)
        self.operator_model = operator_model
        if operator_model is not None:
            if backend != operator_model.backend:
                raise ValueError("backend must match operator_model.backend")
            if operator_model.training_inputs.shape != x.shape or not bool(jnp.all(operator_model.training_inputs == x)):
                raise ValueError("operator_model must use the same ordered training inputs")
            self.factor = operator_model
            self.nugget = operator_model.nugget
        elif backend == "cudss":
            from kernelpack.sparse import CudssFactor

            indptr, indices, data = _sparse_training_kernel(
                self.inputs, self.epsilon, self.nugget,
                int(sparse_assembly_batch_bytes),
            )
            self.factor = CudssFactor(
                indptr, indices, data, (len(x), len(x)),
                rhs_block_size=int(rhs_block_size),
                reordering=self.cudss_reordering,
            )
            self.factor.enable_cuda_graph()
        else:
            k = wendland_c4_matrix(
                self.inputs, self.inputs, self.epsilon
            ) + self.nugget * jnp.eye(len(x), dtype=x.dtype)
            self.factor = _DenseFactor(k)
        self._closed = False
        self._prepared_prediction = None

    @classmethod
    def prepare_spatial_reconstructor(cls, points, velocity, *, spatial_queries=None,
            spatial_center_indices=None, spatial_batch_size=16,
            spatial_setup_bytes=64*1024**2, max_inflight_spatial_blocks=4,
            spatial_reconstructor=None,
            **decoder_options):
        """Prepare GPU spatial maps without constructing an operator factor.

        Training/validation can reuse the spatial method independently of the
        covariance. This avoids factoring a disposable seed operator.
        """
        fields=jnp.asarray(velocity)
        if fields.device.platform!="gpu" or fields.ndim not in (3,4):
            raise ValueError("Spatial preparation requires GPU fields with node/component axes")
        for value in (spatial_batch_size,spatial_setup_bytes,max_inflight_spatial_blocks):
            if int(value)!=value or value<1:
                raise ValueError("Spatial block and memory budgets must be positive integers")
        preparation=cls.__new__(cls)
        preparation.velocity=fields
        preparation.decoder=LocalDFPHSDecoder(jnp.asarray(points,device=fields.device),**decoder_options)
        preparation._configure_spatial_sources(points,decoder_options)
        if fields.shape[1]!=len(preparation.decoder.points) or fields.shape[-1]!=preparation.decoder.dimension:
            raise ValueError("Spatial fields must match nodes and vector dimension")
        preparation.spatial_batch_size=int(spatial_batch_size)
        preparation.spatial_setup_bytes=int(spatial_setup_bytes)
        preparation.max_inflight_spatial_blocks=int(max_inflight_spatial_blocks)
        preparation._closed=False
        return preparation.prepare_reconstructor(spatial_queries=spatial_queries,
            spatial_center_indices=spatial_center_indices,spatial_reconstructor=spatial_reconstructor)

    def _configure_spatial_sources(self,points,options):
        decoder=self.decoder
        self._spatial_source_keepalive=(points,options.get("domain_bounds"),
            options.get("periodic_dims"),options.get("time_points"))
        point_source=(points if isinstance(points,jax.Array) and points.device.platform=="gpu" else decoder.points)
        point_token=("array",_spatial_identity(point_source))
        self._spatial_sources={
            "points":(decoder.points,point_token),
            "bounds":((decoder.minimum,decoder.maximum),_spatial_source_token(
                options.get("domain_bounds"),(decoder.minimum,decoder.maximum),("point_bounds",point_token))),
            "periodic":(decoder.periodic,_spatial_source_token(options.get("periodic_dims"),
                decoder.periodic,("all_axes",))),
            "time":(decoder.time_points,_spatial_source_token(options.get("time_points"),
                decoder.time_points,("no_times",))),
        }

    def _spatial_specification(self,spatial_queries=None,spatial_center_indices=None,*,method):
        decoder=self.decoder
        current={"points":decoder.points,"bounds":(decoder.minimum,decoder.maximum),
                 "periodic":decoder.periodic,"time":decoder.time_points}
        tokens=tuple(token if _same_spatial_identity(original,current[name])
                     else ("changed",_spatial_identity(current[name]))
                     for name,(original,token) in self._spatial_sources.items())
        nodal=spatial_queries is None or spatial_queries is decoder.points
        positions=jnp.asarray(decoder.points if spatial_queries is None else spatial_queries,
            dtype=decoder.points.dtype,device=self.velocity.device)
        centers=(None if spatial_center_indices is None else
                 self._spatial_centers(spatial_queries,positions,spatial_center_indices))
        query_source=(spatial_queries if isinstance(spatial_queries,jax.Array)
                      and spatial_queries.device.platform=="gpu" else positions)
        query_token=(("nodes",) if nodal else ("array",_spatial_identity(query_source)))
        center_token=(("original_node_ids",) if nodal and spatial_center_indices is None else
                      (("inferred",) if spatial_center_indices is None else ("array",_spatial_identity(
                          spatial_center_indices if isinstance(spatial_center_indices,jax.Array)
                          and spatial_center_indices.device.platform=="gpu" else centers))))
        signature=("FastOLSpatialV1",tokens,query_token,center_token,method,
            str(self.velocity.device),str(self.velocity.dtype),tuple(self.velocity.shape[1:]),
            str(decoder.points.dtype),self.spatial_batch_size,
            decoder.order,decoder.exponent,decoder.stencil_size,decoder.boundary_fraction,
            decoder.use_periodic,decoder.temporal_interpolation,decoder.use_time_kernel,
            decoder.spatial_kernel,decoder.matern_base_length_factor,decoder.matern_num_modes,decoder.matern_gamma)
        keepalive=(self._spatial_sources,self._spatial_source_keepalive,current,positions,centers,
                   spatial_queries,spatial_center_indices)
        return signature,keepalive

    def _validate_spatial_reconstructor(self,reconstructor,spatial_queries=None,
                                      spatial_center_indices=None,*,method,geometry_only=False):
        if not isinstance(reconstructor,PreparedSpatialReconstructor):
            raise TypeError("spatial_reconstructor must be a prepared geometry-only spatial map")
        signature,_=self._spatial_specification(spatial_queries,spatial_center_indices,method=method)
        expected=reconstructor.signature
        if geometry_only:
            signature=signature[:2]+signature[5:]
            expected=expected[:2]+expected[5:]
        if signature!=expected:
            raise ValueError("Prepared spatial map is incompatible with geometry, node identities, method, precision, device, or saved-time layout")

    def device_state(self):
        """Numerical buffers required by a prepared operator Lagrange map."""
        if self._closed:
            raise RuntimeError("FastOL model is closed")
        if self.operator_model is not None:
            return self.operator_model.device_state()
        factor_state=(self.factor.device_state() if self.backend=="cudss"
                      else self.factor.lower)
        return dict(mean=self.normalizer.mean,std_safe=self.normalizer.std_safe,
                    zero=self.normalizer.zero,inputs=self.inputs,factor=factor_state)

    def lagrange_map(self, queries, *, state=None):
        """H=K(query,train) Kreg^-1, with factors reused."""
        if self._closed:
            raise RuntimeError("FastOL model is closed")
        if state is None:
            state=self.device_state()
        if self.operator_model is not None:
            return self.operator_model.lagrange_map(queries,state=state)
        inputs=state["inputs"]
        q = jnp.asarray(queries, dtype=inputs.dtype)
        if q.ndim < 2 or not q.size or q.shape[-1] != inputs.shape[1]:
            raise ValueError("query feature dimension must match training")
        query_shape = q.shape[:-1]
        q = q.reshape(-1, q.shape[-1])
        normalized=jnp.where(state["zero"],0.,(q-state["mean"])/state["std_safe"])
        kq = wendland_c4_matrix(normalized, inputs, self.epsilon)
        if self.backend=="cudss":
            solved=self.factor.solve_with_state(kq.T,state=state["factor"])
        else:
            lower=state["factor"]
            solved=jl.solve_triangular(lower.T,jl.solve_triangular(lower,kq.T,lower=True),lower=False)
        return solved.T.reshape(query_shape + (len(inputs),))

    def prepare_reconstructor(self, *, spatial_queries=None, method="weights", spatial_center_indices=None,
                              spatial_reconstructor=None):
        """Prepare local factors and return a compiled field-reconstruction function.

        Its input contains query fields at the original spatial nodes, shaped
        (queries, nodes, components), or (queries, nodes, times, components).
        Geometry preparation uses shared GPU stencil selection. Numerical local
        systems, factors, and reconstruction execute on GPU when inputs do.
        spatial_center_indices supplies zero-based original node identities for
        explicit spatial queries; default nodal queries carry their own indices.
        Selection retains the decoder's geometry precision, independent of fields.
        Prepared factors remain resident outside the decoder's LRU cache;
        keep the model open until all dependent computations have completed.
        """
        if self._closed:
            raise RuntimeError("FastOL model is closed")
        if method not in ("weights","solve"):
            raise ValueError("method must be weights or solve")
        positions = jnp.asarray(self.decoder.points if spatial_queries is None
                               else spatial_queries,dtype=self.decoder.points.dtype,device=self.velocity.device)
        centers=self._spatial_centers(spatial_queries,positions,spatial_center_indices)
        d = self.decoder.dimension
        if positions.ndim != 2 or positions.shape[1] != d or not len(positions) or not bool(jnp.all(jnp.isfinite(positions))):
            raise ValueError("spatial_queries must be a nonempty finite spatial matrix")
        signature,keepalive=self._spatial_specification(spatial_queries,spatial_center_indices,method=method)
        if spatial_reconstructor is not None:
            self._validate_spatial_reconstructor(spatial_reconstructor,spatial_queries,
                spatial_center_indices,method=method)
            return spatial_reconstructor
        size = min(self.spatial_batch_size, len(positions))
        # Factor many stencils together during setup, independently of the
        # smaller runtime streaming block. Bound matrix scratch near 64 MiB.
        rank=d*math.comb(self.decoder.order+d,d)-math.comb(self.decoder.order+d-1,d)
        width=d*self.decoder.stencil_size+rank
        itemsize=self.decoder.points.dtype.itemsize
        # Account for float64 local systems AND candidate/image/sort storage.
        # The selector also independently bounds its own query batches.
        raw_columns=(1 if d==2 else 3)*math.comb(self.decoder.order+1+d,d)
        # The dense stencil search needs one distance row per node and query; the cell-blocked
        # search used for large non-periodic node sets does not, so its blocks do not shrink with m.
        selector=0 if getattr(self.decoder,"uses_grid_knn",False) else len(self.decoder.points)*((6*d+16)*itemsize+16)
        per_stencil=itemsize*(4*width*width+4*d*self.decoder.stencil_size*raw_columns
            +raw_columns*raw_columns)+selector
        setup_size=max(1,getattr(self,"spatial_setup_bytes",64*1024**2)//max(1,per_stencil))
        padded=jnp.pad(positions,((0,(-len(positions))%size),(0,0)),mode="edge")
        padded_centers=None if centers is None else jnp.pad(centers,(0,(-len(positions))%size),mode="edge")
        entries=[]
        weight_chunks=[]
        index_chunks=[]
        from .local_dfphs import _prepared_coefficients, _local_evaluation_weights
        for start in range(0,len(padded),setup_size):
            entry=self.decoder._prepare_block(padded[start:start+setup_size],self.velocity,
                center_indices=None if padded_centers is None else padded_centers[start:start+setup_size])
            if method=="weights":
                _,ids,a,e,factor,pivot=entry
                # Adjoint solves and their pseudoinverse fallback obey the same
                # setup bound as factorization. Retain only compact weights.
                block_weights=_local_evaluation_weights(a,factor,pivot,e)
                if not bool(jnp.all(jnp.isfinite(block_weights))):
                    raise RuntimeError("Nonfinite prepared spatial weights")
                count=ids.shape[-1]
                weight_chunks.append(block_weights[...,:d*count].reshape(len(ids),d,d,count))
                index_chunks.append(ids)
            else:
                entries.append(entry)
        groups=len(padded)//size
        nt = self.velocity.shape[2] if self.velocity.ndim == 4 else 1
        npositions=len(positions)
        spacetime=self.velocity.ndim==4
        temporal = self.decoder.prepare_temporal_transform(nt, self.decoder.points.dtype)
        if method == "weights":
            weights=jnp.concatenate(weight_chunks,axis=0).reshape(groups,size,d,d,-1)
            indices=jnp.concatenate(index_chunks,axis=0).reshape(groups,size,-1)
            def reconstruct_weights(fields,buffers):
                weights,indices,temporal=buffers
                values=fields if fields.ndim==4 else fields[:,:,None,:]
                def step(_,data):
                    w,ids=data
                    selected=values[:,ids]
                    return None,jnp.einsum("sdcn,qsntc->qstd",w,selected,precision=jax.lax.Precision.HIGHEST)
                _,blocks=jax.lax.scan(step,None,(weights,indices))
                result=jnp.transpose(blocks,(1,0,2,3,4)).reshape(len(fields),-1,nt,d)[:,:npositions]
                if nt>1:
                    result=jnp.einsum("tu,qsud->qstd",temporal,result,precision=jax.lax.Precision.HIGHEST)
                return result if fields.ndim==4 else result[:,:,0,:]
            return PreparedSpatialReconstructor(jax.jit(reconstruct_weights),(weights,indices,temporal),
                signature=signature,keepalive=keepalive,
                output_shape=(npositions,nt,d) if spacetime else (npositions,d),
                setup_bytes=getattr(self,"spatial_setup_bytes",64*1024**2))

        systems,evaluation,lu,pivots=(
            jnp.concatenate([entry[k] for entry in entries],axis=0).reshape(
                (groups,size)+entries[0][k].shape[1:]) for k in (2,3,4,5))
        indices=jnp.concatenate([entry[1] for entry in entries],axis=0).reshape(groups,size,-1)
        def reconstruct_fields(fields,buffers):
            systems,evaluation,lu,pivots,indices,temporal=buffers
            def reconstruct(_, data):
                a, e, factor, pivot, ids = data
                coefficients = _prepared_coefficients(fields, ids, a, factor, pivot, temporal)
                # No explicit Kronecker evaluation matrix is needed.
                values = jnp.einsum("sqtc,sdc->qstd", coefficients, e,
                                     precision=jax.lax.Precision.HIGHEST)
                return None, values

            _, blocks = jax.lax.scan(reconstruct, None,
                                     (systems, evaluation, lu, pivots, indices))
            result = jnp.transpose(blocks, (1, 0, 2, 3, 4)).reshape(
                len(fields), -1, nt, d)[:, :npositions]
            if not spacetime:
                result = result[:, :, 0, :]
            return result
        return PreparedSpatialReconstructor(jax.jit(reconstruct_fields),
            (systems,evaluation,lu,pivots,indices,temporal),signature=signature,keepalive=keepalive,
            output_shape=(npositions,nt,d) if spacetime else (npositions,d),
            setup_bytes=getattr(self,"spatial_setup_bytes",64*1024**2))

    def prepare_predictor(self, *, spatial_queries=None, spatial_mode="auto",
                          precompute_bytes=128*1024**2, operator_evaluation="lagrange", spatial_center_indices=None,
                          spatial_reconstructor=None):
        """JIT Lagrange prediction with fixed spatial work moved into preparation.

        auto precomputes reconstructed training fields within the explicit memory
        budget; weights uses compact local evaluation maps; solve is a reference.
        Every option retains the same operator Lagrange solve and its check.
        """
        if operator_evaluation!="lagrange":
            raise ValueError("FastOL prepared prediction requires operator_evaluation='lagrange'")
        if spatial_mode not in ("auto","precomputed","weights","solve"):
            raise ValueError("Unknown spatial_mode")
        if int(precompute_bytes)!=precompute_bytes or precompute_bytes<0:
            raise ValueError("precompute_bytes must be a nonnegative integer")
        positions=self.decoder.points if spatial_queries is None else spatial_queries
        nt=self.velocity.shape[2] if self.velocity.ndim==4 else 1
        required=len(self.inputs)*len(positions)*nt*self.decoder.dimension*self.decoder.points.dtype.itemsize
        use_precomputed=(spatial_mode=="precomputed" or (spatial_mode=="auto" and required<=precompute_bytes))
        method="solve" if spatial_mode=="solve" else "weights"
        shared=spatial_reconstructor if spatial_reconstructor is not None else self._shared_spatial_reconstructor
        if shared is not None:
            try:
                self._validate_spatial_reconstructor(shared,spatial_queries,spatial_center_indices,method=method)
            except (TypeError,ValueError):
                if spatial_reconstructor is not None:
                    raise
                # An implicit cached map may have been prepared for earlier
                # geometry or a different reference method. Rebuild that map.
                self._shared_spatial_reconstructor=None
                self._prepared_prediction=None
                shared=None
        reconstruct=self.prepare_reconstructor(spatial_queries=spatial_queries,
            method=method,spatial_center_indices=spatial_center_indices,spatial_reconstructor=shared)
        prepared=(reconstruct.precompute_fields(self.velocity,budget=precompute_bytes,
            force=spatial_mode=="precomputed") if use_precomputed else None)
        dtype=self.inputs.dtype
        features=self.inputs.shape[1]
        lagrange=self.lagrange_map
        reconstruct_compiled=reconstruct.compiled
        operator_state=self.device_state()
        fields=prepared if use_precomputed else self.velocity
        def predict_prepared(queries,buffers):
            state,training_fields,reconstruction_buffers=buffers
            q=jnp.asarray(queries,dtype=dtype)
            if q.ndim < 2 or not q.size or q.shape[-1] != features:
                raise ValueError("queries must have shape (..., query, training_features)")
            query_shape=q.shape[:-1];q=q.reshape(-1,q.shape[-1])
            if use_precomputed:
                h=lagrange(q,state=state)
                result=jnp.tensordot(h,training_fields,axes=((1,),(0,)),precision=jax.lax.Precision.HIGHEST)
            else:
                h=lagrange(q,state=state)
                fields=jnp.tensordot(h,training_fields,axes=((1,),(0,)),precision=jax.lax.Precision.HIGHEST)
                result=reconstruct_compiled(fields,reconstruction_buffers)
            finite=jnp.all(jnp.isfinite(q),axis=-1)
            result=jnp.where(finite.reshape((-1,)+(1,)*(result.ndim-1)),result,jnp.nan)
            return result.reshape(query_shape+result.shape[1:])
        reconstruction_buffers=() if use_precomputed else reconstruct.buffers
        return PreparedGPUCallable(jax.jit(predict_prepared),(operator_state,fields,reconstruction_buffers),
            spatial_mode="precomputed" if use_precomputed else ("solve" if spatial_mode=="solve" else "weights"),
            precomputed_bytes=required if use_precomputed else 0,operator_evaluation=operator_evaluation)

    def _spatial_centers(self,spatial_queries,positions,provided):
        """Carry nodal identity separately from floating-point coordinates."""
        if provided is None:
            return (jnp.arange(len(positions),dtype=jnp.int32,device=self.decoder.points.device)
                if spatial_queries is None or spatial_queries is self.decoder.points else None)
        centers=jnp.asarray(provided,device=self.decoder.points.device)
        if centers.shape!=(len(positions),) or not jnp.issubdtype(centers.dtype,jnp.integer):
            raise ValueError("spatial_center_indices must be an integer vector, one per spatial query")
        return centers

    def predict(
        self, queries, *, spatial_queries=None, lagrange=True, query_batch_size=None, spatial_center_indices=None
    ):
        if self._closed:
            raise RuntimeError("FastOL model is closed")
        q = jnp.asarray(queries, dtype=self.inputs.dtype,device=self.inputs.device)
        if q.ndim > 2:
            if not q.size or q.shape[-1] != self.inputs.shape[1]:
                raise ValueError("query feature dimension must match training")
            prediction = self.predict(
                q.reshape(-1, q.shape[-1]), spatial_queries=spatial_queries,
                lagrange=lagrange, query_batch_size=query_batch_size,
                spatial_center_indices=spatial_center_indices,
            )
            return prediction.reshape(q.shape[:-1] + prediction.shape[1:])
        if (
            q.ndim != 2
            or not len(q)
            or q.shape[1] != self.inputs.shape[1]
            or not bool(jnp.all(jnp.isfinite(q)))
        ):
            raise ValueError(
                "queries must be a nonempty finite matrix matching training features"
            )
        if lagrange:
            # Prepare geometry once and stream all query blocks in a compiled
            # scan. Keep only the latest prepared request to bound retention.
            if query_batch_size is not None and (int(query_batch_size)!=query_batch_size or query_batch_size<1):
                raise ValueError("query_batch_size must be positive integer")
            factor=(getattr(self.operator_model,"sparse_factor",None)
                if self.operator_model is not None else self.factor)
            key=(id(spatial_queries),id(spatial_center_indices),id(factor),
                id(self._shared_spatial_reconstructor),self.spatial_batch_size,self.decoder.boundary_fraction,
                id(getattr(self.operator_model,"lower",None)),
                id(getattr(self.operator_model,"mean_correction",None)),
                id(self.velocity),id(self.inputs),id(self.decoder.points),
                id(getattr(self.decoder,"minimum",None)),id(getattr(self.decoder,"maximum",None)),
                id(getattr(self.decoder,"periodic",None)),id(getattr(self.decoder,"time_points",None)),
                getattr(self.decoder,"use_periodic",False),getattr(self.decoder,"use_time_kernel",False),
                getattr(self.decoder,"temporal_interpolation","none"),
                getattr(self.decoder,"spatial_kernel","phs"),
                getattr(self.decoder,"matern_base_length_factor",None),
                getattr(self.decoder,"matern_num_modes",None),getattr(self.decoder,"matern_gamma",None),
                self.decoder.order,getattr(self.decoder,"exponent",None),self.decoder.stencil_size)
            cached=getattr(self,"_prepared_prediction",None)
            # Only immutable device arrays have safe identity-based reuse.
            # Host arrays/lists can change in place; convert them afresh on GPU.
            immutable_queries=spatial_queries is None or isinstance(spatial_queries,jax.Array)
            immutable_centers=spatial_center_indices is None or isinstance(spatial_center_indices,jax.Array)
            if not (immutable_queries and immutable_centers) or cached is None or cached[0]!=key:
                prepared=self.prepare_predictor(spatial_queries=spatial_queries,
                    spatial_center_indices=spatial_center_indices,spatial_mode="weights")
                self._prepared_prediction=(key,prepared,spatial_queries,spatial_center_indices,{})
            else:
                prepared=cached[1]
            batch=len(q) if query_batch_size is None else min(len(q),int(query_batch_size))
            if batch==len(q):
                result=prepared(q)
            else:
                padded=jnp.pad(q,((0,(-len(q))%batch),(0,0)),mode="edge").reshape(-1,batch,q.shape[-1])
                runners=self._prepared_prediction[4]
                if batch not in runners:
                    prepared_compiled=prepared.compiled
                    @jax.jit
                    def evaluate_blocks(blocks,buffers):
                        return jax.lax.scan(lambda _,part:(None,prepared_compiled(part,buffers)),None,blocks)[1]
                    runners.clear()
                    runners[batch]=evaluate_blocks
                evaluate_blocks=runners[batch]
                result=evaluate_blocks(padded,prepared.buffers)
                result=result.reshape((-1,)+result.shape[2:])[:len(q)]
            if not bool(jnp.all(jnp.isfinite(result))):
                raise RuntimeError("Nonfinite operator or spatial prediction")
            return result
        if query_batch_size is not None:
            if int(query_batch_size) != query_batch_size or query_batch_size < 1:
                raise ValueError("query_batch_size must be positive integer")
            if len(q) > query_batch_size:
                return jnp.concatenate(
                    [
                        self.predict(
                            q[start : start + query_batch_size],
                            spatial_queries=spatial_queries,
                            lagrange=lagrange,
                            spatial_center_indices=spatial_center_indices,
                        )
                        for start in range(0, len(q), query_batch_size)
                    ],
                    axis=0,
                )
        positions = (
            self.decoder.points
            if spatial_queries is None
            else jnp.asarray(spatial_queries,dtype=self.decoder.points.dtype,device=self.velocity.device)
        )
        centers=self._spatial_centers(spatial_queries,positions,spatial_center_indices)
        if self.operator_model is not None and not lagrange:
            raise ValueError("Custom operator models require Lagrange evaluation")
        if (
            positions.ndim != 2
            or positions.shape[1] != self.decoder.dimension
            or not len(positions)
        ):
            raise ValueError("spatial_queries must have matching spatial dimension")
        if lagrange:
            h = self.lagrange_map(q)
            # Spatial coefficient solves are linear in the sampled fields.
            # Combine samples first so their RHS count is queries, not training
            # examples. Keep the full coefficient path for direct evaluation.
            predicted_velocity = jnp.tensordot(
                h, self.velocity, axes=((1,), (0,)),
                precision=jax.lax.Precision.HIGHEST,
            )
        else:
            kq = wendland_c4_matrix(
                self.normalizer.transform(q), self.inputs, self.epsilon
            )
        outputs = []
        for start in range(0, len(positions), self.spatial_batch_size):
            block = self.decoder.fit_block(
                predicted_velocity if lagrange else self.velocity,
                positions[start : start + self.spatial_batch_size],
                validate=False,
                center_indices=None if centers is None else centers[start:start+self.spatial_batch_size],
            )
            if lagrange:
                predicted = block.coefficients
            else:
                s, n, c = block.coefficients.shape
                fitted = self.factor.solve(
                    jnp.transpose(block.coefficients, (1, 0, 2)).reshape(n, s * c).astype(self.inputs.dtype)
                )
                predicted = jnp.transpose(
                    (kq @ fitted).reshape(len(q), s, c), (1, 0, 2)
                )
            outputs.append(block.reconstruct(predicted))
            # Bound live coefficient memory before creating the next block.
            if len(outputs) % self.max_inflight_spatial_blocks == 0:
                outputs[-1].block_until_ready()
        result = jnp.concatenate(outputs, axis=1)
        if self.velocity.ndim == 4:
            result = result.reshape(
                len(q), len(positions), self.velocity.shape[2], self.decoder.dimension
            )
        if not bool(jnp.all(jnp.isfinite(result))):
            raise RuntimeError("Nonfinite operator or spatial prediction")
        return result

    def close(self):
        if not self._closed:
            self._prepared_prediction = None
            self._shared_spatial_reconstructor = None
            self.factor.close()
            self.decoder.clear_geometry_cache()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
