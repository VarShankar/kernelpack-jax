"""Paper inverse problems on saved clouds: Darcy permeability, RD reaction, transport velocity.

For each cloud (geometry, h, seed), order p and parameter family we recover theta from
analytic observations and report the recovery error against h.

  observations  analytic solution at the physical nodes (Darcy: steady; RD, transport: t = T = .5)
  forcing       continuous manufactured forcing evaluated at the true parameters
  forward model the paper forward discretization of run_cudss_paper_forward.py: RBF-FD of order p,
                ghost-node Dirichlet closure, cuDSS; RD and transport use SBDF3 with the history
                u(0), u(dt), u(2dt) from the manufactured solution and dt = c min(h, h^(p/3));
                transport uses direct RBF-FD hyperviscosity on interior rows with k = p/2 + 1 and
                gamma estimated once at the true velocity, then frozen
  fit           Levenberg-Marquardt (Optimistix), forward-mode Jacobian, theta0 = 0, <= 60 steps

The residual is r(theta) = (u_h(theta) - u_exact) / ||u_exact||. At theta_true it equals the
forward discretization error, so the recovery error is driven only by discretization.
Results: one JSON line per case in <output>/results.jsonl, with wall-clock timings.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax
import numpy as np
import optimistix as optx

import run_cudss_paper_forward as fwd  # enables the persistent compile cache
from run_cudss_paper_forward import timestep, load_domain
from kernelpack.solvers.reaction_diffusion_stateless import StatelessReactionDiffusionSolver
from kernelpack.solvers.eulerian_advection_diffusion_reaction import StatelessEulerianAdvectionDiffusionReactionSolver
from kernelpack.solvers.spectral import (dirichlet_implicit_generator_precheck, dirichlet_laplacian_precheck,
                                         neumann_operator_precheck)
from advection_velocity_inverse_convergence_study import theta_reference
import darcy_forward_convergence_study as darcy
from darcy_inverse_convergence_study import manufactured_state_1
import reaction_diffusion_forward_convergence_study as rd
import reaction_diffusion_inverse_convergence_study as rdi
import advection_diffusion_forward_convergence_study as ad
import reference_velocity as rv

T = .5
MAX_LM_STEPS = 60
LM_BUDGET = {"max_steps": MAX_LM_STEPS}   # set from --max-lm-steps
TRANSPORT_NU = .05
TRANSPORT_FAMILIES = ("smooth", "c1")
HV_ROW_MASK = {"mask": "boundary"}     # set from --hyperviscosity-rows
SAVE_STATES = None                     # set from --save-states (ADR solution plots)


def run_lm(residual, theta0, max_steps=MAX_LM_STEPS):
    """Optimistix LM, forward-mode Jacobian. Returns theta, ||r||, steps, status, converged."""
    lm = optx.LevenbergMarquardt(atol=1e-12, rtol=1e-12, linear_solver=lineax.AutoLinearSolver(well_posed=False))
    options, tags = {"jac": "fwd"}, frozenset()
    fn = lambda theta, _: (residual(theta), None)
    r_struct, aux_struct = jax.eval_shape(fn, theta0, None)
    state = lm.init(fn, theta0, None, options, r_struct, aux_struct, tags)
    step = eqx.filter_jit(lambda y, s: lm.step(fn, y, None, options, s, tags))
    y, aux, result, steps, done = theta0, None, None, 0, False
    for _ in range(max_steps):
        y, state, aux = step(y, state)
        steps += 1
        done, result = lm.terminate(fn, y, None, options, state, tags)
        if bool(done):
            break
    y, _, _ = lm.postprocess(fn, y, aux, None, options, state, tags, result)
    # A run that exhausts max_steps is not converged even if Optimistix reports success.
    return y, float(jnp.linalg.norm(residual(y))), steps, str(result), bool(done) and bool(result == optx.RESULTS.successful)


def run_gn(residual, theta0, max_steps=30, step_tol=1e-10):
    """Gauss-Newton with step halving, for warm starts close to the minimizer.

    Each iteration forms r and the N x P Jacobian in one forward-mode pass and takes the least-squares
    step; a step that does not reduce ||r|| is halved (at most 6 times). Stops when the accepted step is
    below step_tol (1 + ||theta||) or no halved step reduces ||r||.
    Returns theta, ||r||, steps, status, converged.
    """
    def value_and_aux(th):
        v = residual(th)
        return v, v
    rj = jax.jit(jax.jacfwd(value_and_aux, has_aux=True))
    res = jax.jit(residual)
    theta = jnp.asarray(theta0)
    J, r = rj(theta); nr = float(jnp.linalg.norm(r))
    for k in range(1, max_steps + 1):
        Jn, rn = np.asarray(J, float), np.asarray(r, float)
        if not (np.isfinite(Jn).all() and np.isfinite(rn).all()):
            return theta, nr, k, "non-finite residual or Jacobian", False
        delta = jnp.asarray(np.linalg.lstsq(Jn, -rn, rcond=None)[0], dtype=theta.dtype)
        t, accepted = 1.0, False
        for _ in range(7):
            trial = theta + t * delta
            rt = res(trial); nt = float(jnp.linalg.norm(rt))
            if np.isfinite(nt) and nt <= nr:
                accepted = True
                break
            t *= .5
        if not accepted:
            return theta, nr, k, "no decrease along the Gauss-Newton step (minimum reached)", True
        theta = trial
        small = float(jnp.linalg.norm(t * delta)) <= step_tol * (1 + float(jnp.linalg.norm(theta)))
        if small:
            return theta, nt, k, "step below tolerance", True
        J, r = rj(theta); nr = float(jnp.linalg.norm(r))
    return theta, nr, max_steps, "max steps", False


def rel(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def jacobian_summary(residual, theta):
    # Host SVD: a non-finite Jacobian must not reach cuSOLVER, whose failure poisons the CUDA context.
    J = np.asarray(jax.jacfwd(residual)(theta), float)
    if not np.isfinite(J).all():
        return dict(jacobian_singular_values=None, jacobian_condition=None)
    s = np.linalg.svd(J, compute_uv=False)
    return dict(jacobian_singular_values=s.tolist(), jacobian_condition=float(s.max() / s.min()) if s.min() > 0 else None)


INVALID_RESIDUAL = 1e6


def dense_jvp(residual):
    """Same function, with a derivative rule that forms the N x P Jacobian by forward mode in one pass.

    Optimistix (jac="fwd") calls jax.linearize on the residual, which for a time-stepping residual stores
    the intermediates of every step. With this rule the Jacobian depends only on the primal theta, so
    linearize stores just the N x P matrix and applies it to tangents.
    """
    @jax.custom_jvp
    def r(theta):
        return residual(theta)

    @r.defjvp
    def r_jvp(primals, tangents):
        (theta,), (dtheta,) = primals, tangents
        def value_and_aux(th):
            v = residual(th)              # one rollout; its value is returned as the auxiliary output
            return v, v
        J, value = jax.jacfwd(value_and_aux, has_aux=True)(theta)
        return value, J @ dtheta
    return r


def guarded(residual):
    """Replace a non-finite residual (a trial theta whose rollout diverged) by a large constant with zero
    derivative, so that LM rejects the step and the Jacobian stays finite."""
    def safe(theta):
        ok_theta = jnp.all(jnp.isfinite(theta))
        r = residual(jnp.where(ok_theta, theta, 0.))
        ok = ok_theta & jnp.all(jnp.isfinite(r))
        return jnp.where(ok, r, jax.lax.stop_gradient(jnp.full_like(r, INVALID_RESIDUAL)))
    return safe


# ----------------------------------------------------------------------------- Darcy
def normal_derivative(fn, x, normals):
    """n . grad fn at the points x for a batched scalar function fn(x)."""
    g = jax.vmap(jax.grad(lambda z: fn(z[None, :])[0]))(jnp.asarray(x))
    return jnp.sum(g * jnp.asarray(normals), axis=1)


def darcy_group(domain, order, families, warm=None):
    """Assemble and check the operators once per (cloud, order); fit every family on them."""
    x, xb, xa = (jnp.asarray(v) for v in (domain.get_int_bdry_nodes(), domain.get_bdry_nodes(), domain.get_all_nodes()))
    dim = x.shape[1]
    t0 = time.perf_counter()
    solver = darcy.build_solver(domain, order, "rbf", "cudss", 128)
    solver.prepare_boundary_operator(jnp.ones(len(xb)), jnp.zeros(len(xb)))        # pure Neumann (flux) rows
    spectral = neumann_operator_precheck(solver.lap_sparse, solver.bc_sparse, domain)
    setup_s = time.perf_counter() - t0
    n, nb = len(x), len(xb)
    normals_b = jnp.asarray(domain.get_nrmls())
    try:
        if not spectral["passed"]:
            for family in families:
                yield family, dict(status="spectral_rejected", reason=spectral.get("reason"), shared_setup_s=setup_s)
            return
        frozen = solver.freeze()
        for family in families:
            theta_true = darcy.sample_theta(family, dim)
            case = darcy.build_cases(family, theta_true, x, xb)[0]
            target = case.target; norm = jnp.linalg.norm(target)
            # Flux data g = K* du*/dn at the boundary nodes (fixed); the boundary rows are du/dn = g / K(theta).
            flux = darcy.coefficient_field(family, theta_true, xb) * normal_derivative(manufactured_state_1, xb, normals_b)

            def residual(theta, family=family, case=case, target=target, norm=norm, flux=flux):
                k = darcy.coefficient_field(family, theta, xa)
                u = frozen.solve_u_from_fields(k, case.forcing, flux / k[n - nb:n])
                u = u + (jnp.mean(target) - jnp.mean(u))          # the constant: mean over the physical nodes
                return (u - target) / norm

            yield family, fit(residual, theta_true, setup_s,
                              field=lambda th, family=family: darcy.coefficient_field(family, th, x), field_name="permeability",
                              theta0=None if warm is None else warm(family))
    finally:
        solver.close_cudss()


# ----------------------------------------------------------------------------- reaction-diffusion
def rd_parameterization(family, dim):
    """Unknowns and reaction law R(theta, u, x) of the inverse problem.

    The semilinear law R = a u + b u^3 is linear in (a, b), and we fit (a, b) directly, so that theta = 0 is
    pure diffusion. (The forward study writes a, b as shifted softplus functions of raw parameters; at raw 0 they
    are 0.69, a strong cubic growth that blows up on the H-channel before T.) The true (a, b) are the effective
    coefficients of the forward study, so R and the forcing are unchanged. The other laws exp(F(x).theta) u^2
    are fitted in theta.
    """
    theta_raw = rd.sample_theta(family, dim)
    if family == "semilinear":
        return rdi.effective_theta(family, theta_raw), lambda th, u, x: th[0] * u + th[1] * u ** 3
    return theta_raw, lambda th, u, x: rd.reaction_family_value(family, th, u, x)


def rd_group(domain, order, families, dt_factor, cfl_factor=None, warm=None):
    """One assembly, check, and SBDF3 factorization per (cloud, order); the factor does not depend on the family."""
    x = jnp.asarray(domain.get_int_bdry_nodes()); dim = x.shape[1]
    h = float(domain.get_sep_rad())
    dt, steps = timestep(h, order, T=T, factor=dt_factor, cfl_factor=cfl_factor)
    nu = rd.solver_settings(dim)[2]
    t0 = time.perf_counter()
    solver = rd.build_solver_template(domain, order=order, dt=dt, nu=nu, stencil="rbf", linear_solver="cudss",
                                      sparse_assembly_chunk_size=128)
    solver.prepare_boundary_operator(0., lambda z: jnp.ones(len(z)), lambda z: jnp.zeros(len(z)))   # Neumann
    spectral = neumann_operator_precheck(solver.lap_sparse, solver.bc_sparse, domain)
    try:
        if not spectral["passed"]:
            for family in families:
                yield family, dict(status="spectral_rejected", reason=spectral.get("reason"),
                                   shared_setup_s=time.perf_counter() - t0)
            return
        wrapper = StatelessReactionDiffusionSolver.from_solver(solver)
        setup_s = time.perf_counter() - t0
        bc = lambda _a, _b, nr, t, z: normal_derivative(lambda q: rd.exact_state(t, q), z, nr)   # du/dn = g
        history = tuple(rd.exact_state(k * dt, solver.x) for k in range(3))
        target = rd.exact_state(T, x); norm = jnp.linalg.norm(target)
        for family in families:
            forcing = rd.make_forcing(family, nu, rd.sample_theta(family, dim))   # frozen at the truth
            theta_true, reaction_of = rd_parameterization(family, dim)

            @jax.jit
            def residual(theta, reaction_of=reaction_of, forcing=forcing):
                reaction = lambda t, u, z: reaction_of(theta, u, z)
                def step(k, carry):
                    a, b, c = carry
                    return b, c, wrapper.bdf3_state(a, b, c, (k + 1) * dt, reaction=reaction, bc=bc, forcing=forcing)
                u = jax.lax.fori_loop(2, steps, step, history)[-1]
                return (u - target) / norm

            out = fit(residual, theta_true, setup_s,
                      field=lambda th, reaction_of=reaction_of: reaction_of(th, target, x),
                      field_name="reaction at the exact final state",
                      theta0=None if warm is None else warm(family))
            out.update(dt=dt, steps=steps)
            yield family, out
    finally:
        solver.close_cudss()


# ----------------------------------------------------------------------------- transport
def transport_group(domain, geometry, order, families, dt_factor, target_speed, cfl_factor=None, warm=None):
    """Assemble L, G, H and estimate tau, q once per (cloud, order).

    gamma (eq. gamma) is linear in the speed U, so we estimate it once at U = 1 and scale it by the
    true maximum speed of each family; only the SBDF3 factorization and the generator check are per family.
    """
    x, xa = jnp.asarray(domain.get_int_bdry_nodes()), jnp.asarray(domain.get_all_nodes())
    dim = x.shape[1]; h = float(domain.get_sep_rad())
    dt, steps = timestep(h, order, T=T, factor=dt_factor, cfl_factor=cfl_factor)
    theta_ref = theta_reference(dim)
    t0 = time.perf_counter()
    solver, gamma_unit, gamma_status = ad.build_solver_template(
        domain, order=order, dt=dt, nu=TRANSPORT_NU, stencil="rbf", linear_solver="cudss",
        velocity_amplitude=1., gamma_mode="estimate", fixed_gamma=-1e-4, gamma_scale=1., gamma_h_power=0.,
        hyperviscosity_power=str(order // 2 + 1), boundary_order=None, hyperviscosity_row_mask=HV_ROW_MASK["mask"],
        hyperviscosity_method="direct_rbf", hyperviscosity_treatment="implicit", laplacian_power_solver="sparse",
        hyperviscosity_mask_distance_factor=2., sparse_assembly_chunk_size=128, gamma_arpack_ncv=None,
        gamma_arpack_maxiter=None, setup_device="default", gmres_tol=1e-8, gmres_atol=1e-10, gmres_restart=40,
        gmres_maxiter=None, ilu_application="host_callback", linear_solve_diagnostics=False,
        linear_solve_diagnostics_threshold=1e-6, force_boundary_paired_ghost_stencil=False)
    laplacian = dirichlet_laplacian_precheck(solver.lap_sparse, solver.bc_sparse, domain)
    shared_s = time.perf_counter() - t0
    reaction = lambda t, u, z: jnp.zeros_like(u)
    bc = lambda _a, _b, _n, t, z: ad.exact_state(t, z)
    history = tuple(ad.exact_state(k * dt, xa) for k in range(3))
    target = ad.exact_state(T, x); norm = jnp.linalg.norm(target)

    def velocity_of(values):
        def velocity(_t, z):
            assert z.shape == x.shape
            return values
        return velocity

    try:
        for family in families:
            if not laplacian["passed"]:
                yield family, dict(status="spectral_rejected", reason=laplacian.get("reason"), shared_setup_s=shared_s)
                continue
            t1 = time.perf_counter()
            basis = rv.transport_basis(geometry, family, x)        # fixed reference field, same for every h
            # With target_speed U, scale the reference coefficients so that max_j |v(x_j)| = U on this cloud,
            # the normalization of the forward velocity (Section forward_problems).
            theta_true = rv.transport_theta(geometry, family, theta_ref, target_speed)   # fixed per geometry
            v_true = jnp.einsum("i,ind->nd", theta_true, basis)
            speed = float(jnp.max(jnp.linalg.norm(v_true, axis=1)))
            gamma = jnp.asarray(gamma_unit) * target_speed     # nominal U_max, as in the forward problem
            solver.close_cudss()                   # drop the factor of the previous gamma
            solver.set_gamma_components(gamma)
            spectral = dirichlet_implicit_generator_precheck(solver.lap_sparse, solver.hyper_sparse, solver.bc_sparse,
                                                             domain, nu=float(solver.nu), gamma=float(jnp.sum(gamma)))
            if not spectral["passed"]:
                yield family, dict(status="spectral_rejected", reason=spectral.get("reason"), shared_setup_s=shared_s,
                                   true_max_speed=speed)
                continue
            wrapper = StatelessEulerianAdvectionDiffusionReactionSolver.from_solver(solver, prepare_orders=(3,))
            setup_s = shared_s + time.perf_counter() - t1
            forcing = ad.manufactured_forcing(solver.nu, velocity_of(v_true), 0.)   # frozen at the truth

            @jax.jit
            def residual(theta, basis=basis, forcing=forcing, wrapper=wrapper, gamma=gamma):
                velocity = velocity_of(jnp.einsum("i,ind->nd", theta, basis))
                def step(k, carry):
                    a, b, c = carry
                    return b, c, wrapper.bdf3_full_state(a, b, c, (k - 2) * dt, (k - 1) * dt, k * dt, (k + 1) * dt,
                        velocity=velocity, reaction=reaction, bc=bc, forcing=forcing, gamma_components=gamma)
                u = jax.lax.fori_loop(2, steps, step, history)[-1][:solver.n]
                return (u - target) / norm

            out = fit(residual, theta_true, setup_s,
                      field=lambda th, basis=basis: jnp.einsum("i,ind->nd", th, basis), field_name="velocity",
                      theta0=None if warm is None else warm(family))
            out.update(dt=dt, steps=steps, hyperviscosity_power=int(solver.hyperviscosity_power),
                       gamma=np.asarray(gamma).tolist(), gamma_status=gamma_status, true_max_speed=speed)
            yield family, out
    finally:
        solver.close_cudss()


# ----------------------------------------------------------------------------- advection-diffusion-reaction
ADR_SPEED = 1.18          # maximum speed of the fixed no-slip velocity, as in the forward problem


def adr_group(domain, geometry, order, families, dt_factor, cfl_factor=None, warm=None):
    """Recover the reaction law of u_t + v.grad u = nu Lap u + R(u, x; theta) + f with Dirichlet data.

    The velocity v = ADR_SPEED w is fixed and steady, with w the unit-scale reference field of
    reference_velocity.forward_velocity: w = curl(phi0_+^2 m e_y) vanishes wherever phi0 = 0, on the outer
    walls and on any internal obstacle (no slip). The reaction laws and their true parameters are those of
    the reaction-diffusion inverse problem (rd_parameterization); f is computed with the true theta and fixed.
    One assembly, gamma estimate, check, and SBDF3 factorization per (cloud, order); the implicit operator
    nu L + Gamma H does not depend on theta.
    """
    x, xa = jnp.asarray(domain.get_int_bdry_nodes()), jnp.asarray(domain.get_all_nodes())
    dim = x.shape[1]; h = float(domain.get_sep_rad())
    dt, steps = timestep(h, order, T=T, factor=dt_factor, cfl_factor=cfl_factor)
    t0 = time.perf_counter()
    solver, gamma_unit, gamma_status = ad.build_solver_template(
        domain, order=order, dt=dt, nu=TRANSPORT_NU, stencil="rbf", linear_solver="cudss",
        velocity_amplitude=1., gamma_mode="estimate", fixed_gamma=-1e-4, gamma_scale=1., gamma_h_power=0.,
        hyperviscosity_power=str(order // 2 + 1), boundary_order=None, hyperviscosity_row_mask=HV_ROW_MASK["mask"],
        hyperviscosity_method="direct_rbf", hyperviscosity_treatment="implicit", laplacian_power_solver="sparse",
        hyperviscosity_mask_distance_factor=2., sparse_assembly_chunk_size=128, gamma_arpack_ncv=None,
        gamma_arpack_maxiter=None, setup_device="default", gmres_tol=1e-8, gmres_atol=1e-10, gmres_restart=40,
        gmres_maxiter=None, ilu_application="host_callback", linear_solve_diagnostics=False,
        linear_solve_diagnostics_threshold=1e-6, force_boundary_paired_ghost_stencil=False)
    v_true = ADR_SPEED * jnp.asarray(rv.forward_velocity(geometry, x))
    velocity = lambda _t, z: v_true
    gamma = jnp.asarray(gamma_unit) * ADR_SPEED
    solver.close_cudss()
    solver.set_gamma_components(gamma)
    spectral = dirichlet_implicit_generator_precheck(solver.lap_sparse, solver.hyper_sparse, solver.bc_sparse,
                                                     domain, nu=float(solver.nu), gamma=float(jnp.sum(gamma)))
    try:
        if not spectral["passed"]:
            for family in families:
                yield family, dict(status="spectral_rejected", reason=spectral.get("reason"),
                                   shared_setup_s=time.perf_counter() - t0)
            return
        wrapper = StatelessEulerianAdvectionDiffusionReactionSolver.from_solver(solver, prepare_orders=(3,))
        setup_s = time.perf_counter() - t0
        bc = lambda _a, _b, _n, t, z: ad.exact_state(t, z)
        history = tuple(ad.exact_state(k * dt, xa) for k in range(3))
        target = ad.exact_state(T, x); norm = jnp.linalg.norm(target)
        nu = float(solver.nu)
        for family in families:
            theta_true, reaction_of = rd_parameterization(family, dim)
            theta_star = jnp.asarray(theta_true)

            def forcing(_nu, t, z, reaction_of=reaction_of, theta_star=theta_star):
                u = ad.exact_state(t, z)
                return (ad.exact_time_derivative(t, z) + jnp.sum(v_true * ad.exact_gradient(t, z), axis=1)
                        - nu * ad.exact_laplacian(t, z) - reaction_of(theta_star, u, z))

            @jax.jit
            def residual(theta, reaction_of=reaction_of, forcing=forcing):
                reaction = lambda t, u, z: reaction_of(theta, u, z)
                def step(k, carry):
                    a, b, c = carry
                    return b, c, wrapper.bdf3_full_state(a, b, c, (k - 2) * dt, (k - 1) * dt, k * dt, (k + 1) * dt,
                        velocity=velocity, reaction=reaction, bc=bc, forcing=forcing, gamma_components=gamma)
                u = jax.lax.fori_loop(2, steps, step, history)[-1][:solver.n]
                return (u - target) / norm

            out = fit(residual, theta_true, setup_s,
                      field=lambda th, reaction_of=reaction_of: reaction_of(th, target, x),
                      field_name="reaction at the exact final state",
                      theta0=None if warm is None else warm(family))
            out.update(dt=dt, steps=steps, hyperviscosity_power=int(solver.hyperviscosity_power),
                       gamma=np.asarray(gamma).tolist(), gamma_status=gamma_status,
                       max_speed=float(jnp.max(jnp.linalg.norm(v_true, axis=1))))
            if SAVE_STATES is not None:
                # Solution plots: computed final states with the true and the recovered parameters.
                theta_fit = jnp.asarray(out["fit_theta"])
                SAVE_STATES.mkdir(parents=True, exist_ok=True)
                np.savez(SAVE_STATES / f"adr_{geometry}_h{h:.2f}_p{order}_{family}.npz",
                         x=np.asarray(x), exact=np.asarray(target), velocity=np.asarray(v_true),
                         u_true_params=np.asarray(residual(theta_star) * norm + target),
                         u_fit_params=np.asarray(residual(theta_fit) * norm + target),
                         reaction_true=np.asarray(reaction_of(theta_star, target, x)),
                         reaction_fit=np.asarray(reaction_of(theta_fit, target, x)),
                         theta_true=np.asarray(theta_star), theta_fit=np.asarray(theta_fit), h=h, order=order)
            yield family, out
    finally:
        solver.close_cudss()


# ----------------------------------------------------------------------------- common fit and report
def fit(residual, theta_true, setup_s, *, field, field_name, theta0=None):
    residual = dense_jvp(guarded(residual))
    warm_start = theta0 is not None
    theta0 = jnp.zeros_like(theta_true) if theta0 is None else jnp.asarray(theta0, dtype=theta_true.dtype)
    t0 = time.perf_counter()
    r_truth = float(jnp.linalg.norm(residual(theta_true)))      # forward discretization error at the truth
    truth_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    if warm_start:
        theta, objective, lm_steps, status, converged = run_gn(residual, theta0)
        solver_name = "gauss_newton"
        if not converged:                    # fall back to Levenberg-Marquardt from the warm start
            theta, objective, lm_steps, status, converged = run_lm(residual, theta0, LM_BUDGET["max_steps"])
            solver_name = "levenberg_marquardt (after gauss_newton)"
    else:
        theta, objective, lm_steps, status, converged = run_lm(residual, theta0, LM_BUDGET["max_steps"])
        solver_name = "levenberg_marquardt"
    valid = bool(np.isfinite(objective)) and objective < .5 * INVALID_RESIDUAL   # guarded values are >= 1e6
    converged = converged and valid
    fit_s = time.perf_counter() - t0
    out = dict(status="ok", setup_s=setup_s, truth_residual_s=truth_s, fit_s=fit_s,
               truth_theta=np.asarray(theta_true).tolist(), fit_theta=np.asarray(theta).tolist(),
               parameter_rel_error=rel(theta, theta_true),
               field_rel_error=rel(field(theta), field(theta_true)), field=field_name,
               objective_at_fit=objective, objective_at_truth=r_truth,
               lm_steps=lm_steps, lm_status=status, lm_converged=converged, valid_fit=valid,
               theta0=np.asarray(theta0).tolist(), warm_start=warm_start, solver=solver_name)
    out.update(jacobian_summary(residual, theta))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--problem", required=True, choices=("darcy", "rd", "transport", "adr"))
    p.add_argument("--domain-dir", type=Path, required=True, help="Clouds {geometry}_h{h:.2f}_s{seed}.npz")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--geometries", nargs="+", default=["h_channel", "treasure_chest"])
    p.add_argument("--h", nargs="+", type=float, default=[.16, .12, .08, .06, .04])
    p.add_argument("--orders", nargs="+", type=int, default=[2, 4, 6])
    p.add_argument("--families", nargs="+", default=None)
    p.add_argument("--max-lm-steps", type=int, default=MAX_LM_STEPS)
    p.add_argument("--transport-speed", type=float, default=1.18,
                   help="Scale the true transport coefficients so the true max speed on each cloud equals this value "
                        "(the forward study uses 1.18); default: unscaled reference coefficients.")
    p.add_argument("--redo-unconverged", action="store_true",
                   help="Refit only cases whose latest record is an ok fit that hit the LM step limit; the new record supersedes it.")
    p.add_argument("--dt-factor", type=float, default=None,
                   help="dt = c min(h, h^(p/3)); default .25 (RD) and .0625 (transport), as in the forward study")
    p.add_argument("--dt-cfl-factor", type=float, default=None,
                   help="With this, dt = min(c_cfl h, c h^(p/3)): separate CFL cap and accuracy constant (c = --dt-factor)")
    p.add_argument("--warm-start-file", type=Path, default=None,
                   help="results.jsonl whose fit_theta at the next coarser h (same geometry, order, family) starts each fit")
    p.add_argument("--hyperviscosity-rows", choices=("interior", "far"), default="interior",
                   help="transport: hyperviscosity on all interior rows, or only on rows farther than 2h from the boundary")
    p.add_argument("--boundary-aware-stencils", action="store_true",
                   help="Select stencils by ray tracing against the cloud's implicit layer (kernelpack.domain.boundary_aware)")
    p.add_argument("--fit-resolution", type=int, default=41)
    p.add_argument("--cfpu-max-sites", type=int, default=512)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--save-states", type=Path, default=None,
                   help="ADR only: save the computed final states and reaction fields (solution plots) here.")
    a = p.parse_args()
    global SAVE_STATES
    SAVE_STATES = a.save_states
    assert jax.default_backend() == "gpu"
    families = a.families or {"darcy": list(darcy.FAMILIES), "rd": list(rd.FAMILIES), "transport": list(TRANSPORT_FAMILIES), "adr": list(rd.FAMILIES)}[a.problem]
    dt_factor = a.dt_factor or {"rd": .25, "transport": .0625, "adr": .0625}.get(a.problem)
    a.output.mkdir(parents=True, exist_ok=True)
    out_path = a.output / "results.jsonl"
    done = set(); latest = {}
    LM_BUDGET["max_steps"] = a.max_lm_steps
    HV_ROW_MASK["mask"] = {"interior": "boundary", "far": "near_boundary"}[a.hyperviscosity_rows]
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line); key = (r["geometry"], r["h"], r["order"], r["family"])
            latest[key] = r
        for key, r in latest.items():
            if r["status"] == "failed":
                continue                                   # failed cases are always retried
            if not (a.redo_unconverged and r["status"] == "ok" and not r["lm_converged"]):
                done.add(key)
    (a.output / ("options_redo.json" if a.redo_unconverged else "options.json")).write_text(json.dumps(dict(vars(a), domain_dir=str(a.domain_dir), output=str(a.output), warm_start_file=str(a.warm_start_file),
        families=families, dt_factor=dt_factor), indent=2) + "\n")
    for geometry in a.geometries:
        for h in a.h:
            path = a.domain_dir / f"{geometry}_h{h:.2f}_s{a.seed}.npz"
            surface = None
            domain = load_domain(path, surface)
            if a.boundary_aware_stencils:
                import stencil_level_set
                stencil_level_set.attach(domain, geometry, h, a.seed)
            for order in a.orders:
                todo = [f for f in families if (geometry, h, order, f) not in done]
                if not todo:
                    continue
                common = dict(problem=a.problem, geometry=geometry, h=h, order=order, seed=a.seed,
                              physical_nodes=int(domain.get_num_int_bdry_nodes()), domain_file=str(path))
                print("GROUP", json.dumps(dict(common, families=todo)), flush=True)
                warm = None
                if a.warm_start_file is not None and a.warm_start_file.exists():
                    prev = {}
                    for line in a.warm_start_file.read_text().splitlines():
                        r = json.loads(line)
                        if r.get("status") == "ok" and r["geometry"] == geometry and r["order"] == order and r["h"] > h:
                            prev[(r["h"], r["family"])] = r["fit_theta"]          # latest record wins
                    def warm(family, prev=prev):
                        hs = sorted(hh for hh, f in prev if f == family)
                        return prev[(hs[0], family)] if hs else None             # next coarser level
                if a.problem == "darcy":
                    group = darcy_group(domain, order, todo, warm)
                elif a.problem == "rd":
                    group = rd_group(domain, order, todo, dt_factor, a.dt_cfl_factor, warm)
                elif a.problem == "transport":
                    group = transport_group(domain, geometry, order, todo, dt_factor, a.transport_speed, a.dt_cfl_factor, warm)
                elif a.problem == "adr":
                    group = adr_group(domain, geometry, order, todo, dt_factor, a.dt_cfl_factor, warm)
                t0 = time.perf_counter(); written = set()
                try:
                    for family, result in group:
                        row = dict(common, family=family, **result, total_s=time.perf_counter() - t0)
                        t0 = time.perf_counter()
                        with out_path.open("a") as f:
                            f.write(json.dumps(row) + "\n")
                        written.add(family)
                        print("RESULT", json.dumps(row), flush=True)
                except Exception as e:  # record the remaining families of this group as failed and continue
                    import traceback; traceback.print_exc()
                    with out_path.open("a") as f:
                        for family in todo:
                            if family not in written:
                                f.write(json.dumps(dict(common, family=family, status="failed", error=repr(e))) + "\n")
            del domain
            jax.clear_caches(); gc.collect()   # once per cloud; compiled kernels are reused across families and orders

if __name__ == "__main__":
    main()
