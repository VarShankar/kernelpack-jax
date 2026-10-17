"""Arnoldi prechecks of the boundary-closed Dirichlet diffusion generator."""
import hashlib
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from ._common import SparseCOOMatrix,sparse_matvec
from ._cudss import CudssSystemCache


@partial(jax.jit,static_argnums=(1,))
def _arnoldi_basis(initial,steps,indices,values):
    """Arnoldi with two-pass Gram-Schmidt on the device for a sparse COO matrix."""
    size=initial.shape[0]
    def matvec(x):
        return jax.ops.segment_sum(values*x[indices[:,1]],indices[:,0],num_segments=size)
    basis=jnp.zeros((size,steps+1),initial.dtype).at[:,0].set(initial/jnp.linalg.norm(initial))
    hess=jnp.zeros((steps+1,steps),initial.dtype)
    def body(k,state):
        basis,hess=state
        w=matvec(basis[:,k])
        mask=(jnp.arange(steps+1)<=k).astype(w.dtype)
        c=(basis.T@w)*mask; w=w-basis@c
        d=(basis.T@w)*mask; w=w-basis@d
        beta=jnp.linalg.norm(w)
        basis=basis.at[:,k+1].set(w/jnp.maximum(beta,jnp.finfo(w.dtype).tiny))
        hess=hess.at[:,k].set(c+d).at[k+1,k].set(beta)
        return basis,hess
    _,hess=jax.lax.fori_loop(0,steps,body,(basis,hess))
    return hess


def sparse_rightmost_eigenvalue(matrix,*,dimensions=(160,320,640),relative_tolerance=1e-8,seed=1701):
    """Rightmost eigenvalue (largest real part) of a sparse COO matrix on the GPU.

    Non-restarted Arnoldi with full reorthogonalization; the Krylov dimension
    grows until the rightmost Ritz pair's residual |h_{m+1,m} y_m| is below
    ``relative_tolerance * max(1, |lambda|)``. Suitable for differentiation
    matrices whose spectra are dominated by complex-conjugate pairs. Returns
    (eigenvalue, converged, dimension, residual).
    """
    indices=jnp.asarray(matrix.indices); values=jnp.asarray(matrix.values)
    size=int(matrix.shape[0])
    initial=jnp.asarray(np.random.default_rng(seed).normal(size=size))
    value=np.nan; residual=np.inf; m=0
    for m in dimensions:
        m=int(min(m,size-1))
        hh=np.asarray(_arnoldi_basis(initial,m,indices,values))
        if not np.isfinite(hh).all(): break
        vals,vecs=np.linalg.eig(hh[:m,:m])
        i=int(np.argmax(vals.real)); value=vals[i]; y=vecs[:,i]
        residual=float(abs(hh[m,m-1]*y[-1])/np.linalg.norm(y))
        if residual<=relative_tolerance*max(1.,abs(value)):
            return complex(value),True,m,residual
        if m>=size-1: break
    return complex(value),False,m,residual


def arnoldi_rightmost(matvec,size,*,dimension=80,restarts=6,seeds=(1701,1702),relative_tolerance=1e-7,dtype=jnp.float64):
    """GPU Krylov construction; only the small projected eigensystem is on host.

    Thick restart from rightmost real Schur vectors. A nonconverged search never passes.
    Multiple starts reduce the chance of missing a mode; this is a numerical
    diagnostic, not a proof that every eigenvalue has been enclosed.
    """
    m=min(size,dimension)
    if size<1:raise ValueError("Arnoldi requires an evolving interior state")
    def block(v,h,start):
        def body(k,state):
            basis,hessenberg,count,active=state
            def advance(_):
                w=matvec(basis[:,k])
                scale=jnp.linalg.norm(w)
                c=basis.T@w;w=w-basis@c
                correction=basis.T@w;w=w-basis@correction
                coefficients=c+correction
                beta=jnp.linalg.norm(w)
                threshold=100*jnp.finfo(w.dtype).eps*jnp.maximum(scale,1.)
                good=jnp.isfinite(beta)
                next_active=good&(beta>threshold)
                next_column=jnp.where(next_active,w/jnp.maximum(beta,jnp.finfo(w.dtype).tiny),jnp.zeros_like(w))
                hh=hessenberg.at[:,k].set(coefficients).at[k+1,k].set(beta)
                return basis.at[:,k+1].set(next_column),hh,jnp.asarray(k+1,dtype=jnp.int32),next_active
            return jax.lax.cond(active,advance,lambda _:state,operand=None)
        return jax.lax.fori_loop(start,m,body,(v,h,start,jnp.bool_(True)))
    compiled=jax.jit(block);checks=[]
    for seed in seeds:
        initial=jnp.asarray(np.random.default_rng(seed).normal(size=size),dtype=dtype)
        vv=jnp.zeros((size,m+1),dtype=initial.dtype).at[:,0].set(initial/jnp.linalg.norm(initial))
        hh_initial=jnp.zeros((m+1,m),dtype=initial.dtype)
        start=jnp.int32(0)
        for cycle in range(restarts):
            v,h,count,_=compiled(vv,hh_initial,start)
            hh=np.asarray(h);count=int(count)
            if count==0 or not np.isfinite(hh).all():
                return dict(passed=False,reason="nonfinite boundary-closed matvec",checks=checks)
            values,vectors=np.linalg.eig(hh[:count,:count])
            index=int(np.argmax(values.real));value=values[index];y=vectors[:,index]
            residual=float(abs(hh[count,count-1]*y[-1])/np.linalg.norm(y))
            scale=max(1.,float(np.linalg.norm(hh[:count,:count],ord=np.inf)))
            epsilon=np.finfo(hh.dtype).eps
            roundoff=1000*epsilon*scale
            converged=residual<=max(relative_tolerance,100*epsilon)*max(1.,abs(value))
            check=dict(seed=seed,cycle=cycle,krylov_dimension=count,
                rightmost_real=float(value.real),rightmost_imag=float(value.imag),
                ritz_residual=residual,roundoff_tolerance=roundoff,converged=bool(converged))
            checks.append(check)
            if converged and value.real>0:
                reason="positive real Laplacian eigenvalue" if value.real>roundoff else "positive Ritz value within roundoff; sign precheck not passed"
                return dict(passed=False,reason=reason,checks=checks)
            if converged:break
            # Krylov-Schur/thick restart: retain several rightmost real Schur
            # vectors AND their residual coupling, rather than restarting
            # from one nearly converged eigenvector and forgetting other modes.
            from scipy.linalg import schur
            keep=max(1,min(6,count//3,count-2))
            sorted_real=np.sort(values.real)
            # The cutoff must separate the retained Ritz values from the rest by a gap that
            # survives the rounding of the Schur reordering; otherwise scipy raises
            # "Leading eigenvalues do not satisfy sort condition". Try nearby block sizes,
            # widest gap first.
            gap_tol=1e-8*scale
            candidates=sorted((k for k in range(1,count-1) if abs(k-keep)<=3),
                              key=lambda k:-(sorted_real[-k]-sorted_real[-k-1]))
            schur_ok=False
            for k in candidates:
                if sorted_real[-k]-sorted_real[-k-1]<=gap_tol:
                    continue
                cutoff=.5*(sorted_real[-k]+sorted_real[-k-1])
                try:
                    tt,uu,selected=schur(hh[:count,:count],output="real",sort=lambda real,imag:real>=cutoff)
                except np.linalg.LinAlgError:
                    continue
                schur_ok=True
                break
            if not schur_ok:
                return dict(passed=False,reason="Arnoldi restart: no separable block of rightmost Ritz values",checks=checks)
            keep=min(int(selected),count-2)
            if keep>1 and tt[keep,keep-1]!=0:keep-=1
            if keep<1:
                return dict(passed=False,reason="Arnoldi restart could not retain a real invariant block",checks=checks)
            vv=jnp.zeros_like(v).at[:,:keep].set(v[:,:count]@jnp.asarray(uu[:,:keep]))
            vv=vv.at[:,keep].set(v[:,count])
            hh_initial=jnp.zeros_like(h).at[:keep,:keep].set(jnp.asarray(tt[:keep,:keep]))
            coupling=hh[count,count-1]*uu[-1,:keep]
            hh_initial=hh_initial.at[keep,:keep].set(jnp.asarray(coupling))
            start=jnp.int32(keep)
        if not converged:
            return dict(passed=False,reason="rightmost Arnoldi search did not converge",checks=checks)
    return dict(passed=True,reason="converged rightmost Ritz values nonpositive",checks=checks)


def arnoldi_largest_magnitude(matvec, size, *, num_modes=8, dimension=80,
                              restarts=6, seeds=(1701, 1702),
                              relative_tolerance=1e-7, dtype=jnp.float64):
    """Estimate the largest-magnitude Ritz modes with a JAX Arnoldi basis.

    The Krylov basis and operator applications stay on the JAX device. Only
    the small projected eigensystem and Schur restart are handled on the host.
    The returned Ritz pairs are numerical samples, not a full-spectrum
    enclosure. Unlike a rightmost-real-part search, this targets large
    negative-real and complex modes that can constrain explicit stability.
    """
    size = int(size)
    num_modes = int(num_modes)
    if size < 2 or num_modes < 1 or num_modes >= size:
        raise ValueError("Arnoldi requires 1 <= num_modes < size")
    m = min(size, max(int(dimension), num_modes + 2))
    m = min(m, size)

    def block(v, h, start):
        def body(k, state):
            basis, hessenberg, active, completed = state

            def advance(_):
                w = matvec(basis[:, k])
                scale = jnp.linalg.norm(w)
                c = basis.T @ w
                w = w - basis @ c
                correction = basis.T @ w
                w = w - basis @ correction
                coefficients = c + correction
                beta = jnp.linalg.norm(w)
                threshold = 100 * jnp.finfo(w.dtype).eps * jnp.maximum(scale, 1.)
                good = jnp.isfinite(beta)
                next_active = good & (beta > threshold)
                next_column = jnp.where(
                    next_active,
                    w / jnp.maximum(beta, jnp.finfo(w.dtype).tiny),
                    jnp.zeros_like(w),
                )
                hh = hessenberg.at[:, k].set(coefficients).at[k + 1, k].set(beta)
                return basis.at[:, k + 1].set(next_column), hh, next_active, jnp.asarray(k + 1, dtype=jnp.int32)

            return jax.lax.cond(active, advance, lambda _: state, operand=None)

        return jax.lax.fori_loop(
            start, m, body, (v, h, jnp.bool_(True), start)
        )

    compiled = jax.jit(block)
    checks = []
    all_values = []
    all_residuals = []
    all_converged = True
    for seed in seeds:
        initial = jnp.asarray(
            np.random.default_rng(seed).normal(size=size), dtype=dtype
        )
        vv = jnp.zeros((size, m + 1), dtype=initial.dtype).at[:, 0].set(
            initial / jnp.linalg.norm(initial)
        )
        hh_initial = jnp.zeros((m + 1, m), dtype=initial.dtype)
        start = jnp.int32(0)
        converged = False
        best_values = np.asarray([], dtype=np.complex128)
        best_residuals = np.asarray([], dtype=np.float64)
        for cycle in range(restarts):
            v, h, _, completed = compiled(vv, hh_initial, start)
            hh = np.asarray(h)
            # The residual row is coupled only to the last completed Arnoldi vector.
            count = int(completed)
            if count == 0 or not np.isfinite(hh).all():
                checks.append(dict(seed=seed, cycle=cycle, converged=False,
                                   reason="nonfinite or empty Arnoldi projection"))
                all_converged = False
                break
            values, vectors = np.linalg.eig(hh[:count, :count])
            order = np.argsort(-np.abs(values))[:min(num_modes, count)]
            selected_values = values[order]
            selected_residuals = np.asarray([
                abs(hh[count, count - 1] * vectors[-1, index])
                / max(np.linalg.norm(vectors[:, index]), np.finfo(float).tiny)
                for index in order
            ], dtype=np.float64)
            scale = max(1., float(np.linalg.norm(hh[:count, :count], ord=np.inf)))
            epsilon = np.finfo(hh.dtype).eps
            converged_modes = (
                len(selected_values) == num_modes
                and np.all(np.isfinite(selected_values))
                and np.all(np.isfinite(selected_residuals))
                and np.all(selected_residuals <= max(
                    relative_tolerance, 100 * epsilon
                ) * np.maximum(1., np.abs(selected_values)))
            )
            best_values, best_residuals = selected_values, selected_residuals
            checks.append(dict(
                seed=seed, cycle=cycle, krylov_dimension=count,
                eigenvalues=np.asarray(selected_values, dtype=np.complex128),
                residuals=selected_residuals,
                roundoff_tolerance=1000 * epsilon * scale,
                converged=bool(converged_modes),
            ))
            if converged_modes:
                converged = True
                break
            if cycle + 1 >= restarts or count < num_modes + 2:
                break

            # Retain a modulus-ordered real Schur subspace, preserving any
            # conjugate 2x2 blocks, then continue with the Arnoldi residual.
            from scipy.linalg import schur
            keep_target = min(max(num_modes + 2, count // 3), count - 2)
            sorted_magnitudes = np.sort(np.abs(values))
            cutoff = 0.5 * (
                sorted_magnitudes[-keep_target]
                + sorted_magnitudes[-keep_target - 1]
            )
            tt, uu, selected = schur(
                hh[:count, :count], output="real",
                sort=lambda real, imag: np.hypot(real, imag) >= cutoff,
            )
            keep = min(int(selected), count - 2)
            if keep > 1 and tt[keep, keep - 1] != 0:
                keep -= 1
            if keep < num_modes:
                break
            vv = jnp.zeros_like(v).at[:, :keep].set(
                v[:, :count] @ jnp.asarray(uu[:, :keep])
            )
            vv = vv.at[:, keep].set(v[:, count])
            hh_initial = jnp.zeros_like(h).at[:keep, :keep].set(
                jnp.asarray(tt[:keep, :keep])
            )
            coupling = hh[count, count - 1] * uu[-1, :keep]
            hh_initial = hh_initial.at[keep, :keep].set(jnp.asarray(coupling))
            start = jnp.int32(keep)

        all_values.extend(best_values.tolist())
        all_residuals.extend(best_residuals.tolist())
        all_converged = all_converged and converged

    # Seeds can rediscover the same Ritz value. Keep one representative within
    # a tolerance-scaled distance while preserving descending magnitude order.
    unique_values = []
    unique_residuals = []
    for value, residual in sorted(
        zip(all_values, all_residuals), key=lambda item: -abs(item[0])
    ):
        if any(abs(value - existing) <= relative_tolerance * max(1., abs(value))
               for existing in unique_values):
            continue
        unique_values.append(value)
        unique_residuals.append(residual)
    return dict(
        converged=bool(all_converged and len(unique_values) >= num_modes),
        reason=("converged largest-magnitude Ritz modes" if all_converged
                else "largest-magnitude Arnoldi did not converge"),
        eigenvalues=np.asarray(unique_values, dtype=np.complex128),
        residuals=np.asarray(unique_residuals, dtype=np.float64),
        requested_modes=num_modes,
        returned_modes=len(unique_values),
        krylov_method="JAX Arnoldi basis; host projected eigensystem and Schur restart",
        checks=checks,
    )


def dirichlet_implicit_generator_precheck(
    laplacian, hyperviscosity, boundary, domain, *, nu, gamma,
    cache=None, dimension=80, restarts=6,
):
    """Check the actual direct-RBF diffusion/hyperviscosity generator.

    The boundary PDE rows of the combined operator determine its ghosts.
    A separately closed Laplacian's spectrum is insufficient for this check.
    """
    if laplacian.shape != hyperviscosity.shape:
        raise ValueError("Laplacian and hyperviscosity must have the same physical-row shape")
    operator = SparseCOOMatrix(
        jnp.concatenate((laplacian.indices, hyperviscosity.indices), axis=0),
        jnp.concatenate((nu * laplacian.values, jnp.sum(jnp.asarray(gamma)) * hyperviscosity.values)),
        laplacian.shape,
    )
    result = dict(dirichlet_laplacian_precheck(
        operator, boundary, domain, cache=cache, dimension=dimension, restarts=restarts,
    ))
    result["operator"] = "homogeneous-Dirichlet boundary-closed nu*L + sum(gamma)*H_direct"
    result["hyperviscosity_construction"] = "direct_rbf_physical_rows"
    if result["reason"] == "positive real Laplacian eigenvalue":
        result["reason"] = "positive real implicit generator eigenvalue"
    return result


def dirichlet_laplacian_precheck(laplacian,boundary,domain,*,cache=None,dimension=80,restarts=6):
    """Eliminate Dirichlet boundary values and ghost variables before Arnoldi.

    For u_B=0, the boundary PDE rows give
      L_BG u_G = -L_BI u_I,
    hence the evolving interior generator is
      L_II - L_IG (L_BG)^(-1) L_BI.
    No dense Schur complement is formed.
    """
    ni=domain.get_num_interior_nodes();nb=domain.get_num_bdry_nodes()
    n=ni+nb;nf=domain.get_num_total_nodes();ng=nf-n
    ij=np.asarray(laplacian.indices);values=np.asarray(laplacian.values)
    bij=np.asarray(boundary.indices);bval=np.asarray(boundary.values)
    if not np.isfinite(values).all() or not np.isfinite(bval).all():
        raise ValueError("Nonfinite stencil weights: polynomial rank/weight construction did not pass")
    digest=hashlib.sha256()
    for a in (ij,values,bij,bval,np.array([ni,nb,ng,dimension,restarts],dtype=np.int64)):digest.update(a.tobytes())
    fingerprint=digest.hexdigest()
    if cache is not None and fingerprint in cache:return dict(cache[fingerprint],cache_hit=True)
    expected=np.zeros(nb)
    # Measure rowwise difference from exact nodal homogeneous Dirichlet data.
    discrepancy=bval-np.where(bij[:,1]==ni+bij[:,0],1.,0.)
    np.add.at(expected,bij[:,0],np.abs(discrepancy))
    if expected.max(initial=0)>1e-10:
        raise ValueError("Dirichlet Arnoldi precheck requires nodal Dirichlet boundary rows")
    if laplacian.shape!=(n,nf) or boundary.shape!=(nb,nf):
        raise ValueError("Laplacian/boundary shapes do not match physical and ghost nodes")
    ghost_cache=None
    try:
        if ng:
            if ng!=nb:raise ValueError("Dirichlet ghost elimination requires one ghost unknown per boundary PDE row")
            mask=(ij[:,0]>=ni)&(ij[:,1]>=n)
            ghost=SparseCOOMatrix(jnp.asarray(ij[mask]-[ni,n]),jnp.asarray(values[mask]),(nb,nb))
            ghost_cache=CudssSystemCache();factor=ghost_cache.prepare(ghost)
            def matvec(v):
                physical=jnp.concatenate((v,jnp.zeros(nb+ng,dtype=v.dtype)))
                boundary_rhs=-sparse_matvec(laplacian.indices,laplacian.values,n,physical)[ni:]
                g=factor.solve(boundary_rhs)
                full=physical.at[n:].set(g)
                return sparse_matvec(laplacian.indices,laplacian.values,n,full)[:ni]
        else:
            def matvec(v):
                full=jnp.concatenate((v,jnp.zeros(nb,dtype=v.dtype)))
                return sparse_matvec(laplacian.indices,laplacian.values,n,full)[:ni]
        result=arnoldi_rightmost(matvec,ni,dimension=dimension,restarts=restarts,dtype=laplacian.values.dtype)
        # Increase the Krylov space under the same bounded policy for every
        # cloud. An unconverged small search is not evidence of instability.
        enlarged=dimension
        while result["reason"]=="rightmost Arnoldi search did not converge" and enlarged<min(ni,320):
            enlarged=min(ni,320,2*enlarged)
            result=arnoldi_rightmost(matvec,ni,dimension=enlarged,restarts=restarts,dtype=laplacian.values.dtype)
        result["final_krylov_dimension_limit"]=min(ni,enlarged)
    except Exception as e:
        result=dict(passed=False,reason="boundary-closed spectral setup failed",error=repr(e),checks=[])
    finally:
        if ghost_cache is not None:ghost_cache.close()
    result.update(fingerprint=fingerprint,cache_hit=False,interior_size=ni,
        boundary_size=nb,ghost_size=ng,operator="homogeneous-Dirichlet boundary-closed Laplacian",
        krylov_backend=jax.default_backend(),projected_eigensystem_backend="host NumPy, size at most krylov_dimension")
    if cache is not None:cache[fingerprint]=result
    return result


def neumann_operator_precheck(operator,boundary,domain,*,cache=None,dimension=80,restarts=6):
    """Rightmost eigenvalues of a ghost-closed operator with Neumann (or Robin) boundary rows.

    operator: (N, N_f) rows of the PDE operator at the physical nodes; boundary: (N_b, N_f) boundary rows
    B U = 0. The boundary rows determine the ghost values, B_G u_G = -B_P u_P, and the operator acting
    on the physical values is M = A_P - A_G B_G^{-1} B_P, of size N x N. With pure Neumann rows,
    constants are an exact null vector of M (RBF-FD reproduces constants), so we restrict M to the
    mean-zero vectors and map the constants to -1, u -> P M P u - (1^T u / N) 1 with P = I - 1 1^T / N,
    whose eigenvalues are those of M with the zero eigenvalue of the constants replaced by -1. Passing means no converged Ritz value with positive real part.
    """
    ni=domain.get_num_interior_nodes();nb=domain.get_num_bdry_nodes()
    n=ni+nb;nf=domain.get_num_total_nodes();ng=nf-n
    ij=np.asarray(operator.indices);values=np.asarray(operator.values)
    bij=np.asarray(boundary.indices);bval=np.asarray(boundary.values)
    if not np.isfinite(values).all() or not np.isfinite(bval).all():
        raise ValueError("Nonfinite stencil weights: polynomial rank/weight construction did not pass")
    if operator.shape!=(n,nf) or boundary.shape!=(nb,nf):
        raise ValueError("operator/boundary shapes do not match physical and ghost nodes")
    if ng!=nb:raise ValueError("Neumann ghost elimination requires one ghost unknown per boundary row")
    digest=hashlib.sha256()
    for a in (ij,values,bij,bval,np.array([ni,nb,ng,dimension,restarts,7],dtype=np.int64)):digest.update(a.tobytes())
    fingerprint=digest.hexdigest()
    if cache is not None and fingerprint in cache:return dict(cache[fingerprint],cache_hit=True)
    ghost_cache=None
    try:
        mask=bij[:,1]>=n
        ghost=SparseCOOMatrix(jnp.asarray(bij[mask]-[0,n]),jnp.asarray(bval[mask]),(nb,nb))
        ghost_cache=CudssSystemCache();factor=ghost_cache.prepare(ghost)
        def matvec(v):
            mean=jnp.mean(v); v=v-mean
            physical=jnp.concatenate((v,jnp.zeros(ng,dtype=v.dtype)))
            g=factor.solve(-sparse_matvec(boundary.indices,boundary.values,nb,physical))
            w=sparse_matvec(operator.indices,operator.values,n,physical.at[n:].set(g))
            return w-jnp.mean(w)-mean          # constants -> eigenvalue -1, the rest unchanged
        result=arnoldi_rightmost(matvec,n,dimension=dimension,restarts=restarts,dtype=operator.values.dtype)
        enlarged=dimension
        while result["reason"]=="rightmost Arnoldi search did not converge" and enlarged<min(n,320):
            enlarged=min(n,320,2*enlarged)
            result=arnoldi_rightmost(matvec,n,dimension=enlarged,restarts=restarts,dtype=operator.values.dtype)
        result["final_krylov_dimension_limit"]=min(n,enlarged)
    except Exception as e:
        result=dict(passed=False,reason="boundary-closed spectral setup failed",error=repr(e),checks=[])
    finally:
        if ghost_cache is not None:ghost_cache.close()
    result.update(fingerprint=fingerprint,cache_hit=False,interior_size=ni,boundary_size=nb,ghost_size=ng,
        operator="Neumann boundary-closed operator on mean-zero physical values",
        krylov_backend=jax.default_backend(),projected_eigensystem_backend="host NumPy, size at most krylov_dimension")
    if cache is not None:cache[fingerprint]=result
    return result
