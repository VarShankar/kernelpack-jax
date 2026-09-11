from __future__ import annotations

from dataclasses import dataclass

from jax import jit, lax, vmap
import jax.numpy as jnp
from jax.scipy.special import gammaln


class JacobiPolynomials:
    @staticmethod
    def recurrence(n: int, alpha: float, beta: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        n = int(jnp.max(jnp.atleast_1d(n)))
        idx = jnp.arange(n, dtype=float)
        a = (beta**2 - alpha**2) * jnp.ones_like(idx)
        b = jnp.ones_like(idx)

        flags0 = idx == 0
        if bool(jnp.any(flags0)):
            a = a.at[flags0].set((beta - alpha) / (alpha + beta + 2.0))
            b0 = jnp.exp(
                (alpha + beta + 1.0) * jnp.log(2.0)
                + gammaln(alpha + 1.0)
                + gammaln(beta + 1.0)
                - gammaln(alpha + beta + 2.0)
            )
            b = b.at[flags0].set(b0)

        flags1 = idx == 1
        if bool(jnp.any(flags1)):
            a = a.at[flags1].set(a[flags1] / ((2.0 + alpha + beta) * (4.0 + alpha + beta)))
            b1 = (
                4.0
                * (1.0 + alpha)
                * (1.0 + beta)
                / ((2.0 + alpha + beta) ** 2 * (3.0 + alpha + beta))
            )
            b = b.at[flags1].set(b1)

        flags = ~(flags0 | flags1)
        if bool(jnp.any(flags)):
            nloc = idx[flags]
            a = a.at[flags].set(
                a[flags] / ((2.0 * nloc + alpha + beta) * (2.0 * nloc + alpha + beta + 2.0))
            )
            b = b.at[flags].set(
                4.0
                * nloc
                * (nloc + alpha)
                * (nloc + beta)
                * (nloc + alpha + beta)
                / (
                    (2.0 * nloc + alpha + beta) ** 2
                    * (2.0 * nloc + alpha + beta + 1.0)
                    * (2.0 * nloc + alpha + beta - 1.0)
                )
            )
        return a, b

    @staticmethod
    def evaluate(a: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray, n: int, d: int = 0) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=float).reshape(-1)
        a = jnp.asarray(a, dtype=float).reshape(-1)
        b = jnp.asarray(b, dtype=float).reshape(-1)
        if d < 0 or n < 0:
            raise ValueError("n and d must be nonnegative")
        if n >= len(a) or n >= len(b):
            raise ValueError("recurrence length is too short")
        if d > n:
            return jnp.zeros((x.shape[0], n + 1), dtype=float)
        tables = _jacobi_values_all(a[: n + 1], b[: n + 1], x)
        return tables[:, :, d]

    @staticmethod
    def tensor_evaluate(
        x: jnp.ndarray,
        alpha: jnp.ndarray,
        recurrence_handle,
        d: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=float)
        alpha = jnp.asarray(alpha, dtype=int)
        if d is None:
            d = jnp.zeros((1, x.shape[1]), dtype=int)
        d = jnp.asarray(d, dtype=int)
        if alpha.shape[1] != x.shape[1] or d.shape[1] != x.shape[1]:
            raise ValueError("dimension mismatch")

        max_alpha = int(alpha.max(initial=0))
        a, b = recurrence_handle(max_alpha + 1)
        if int(d.max(initial=0)) > max_alpha:
            raise ValueError("derivative orders exceed the available polynomial degree")
        return _tensor_evaluate_with_recurrence(x, alpha, d, a, b)


@jit
def _jacobi_values_all(a: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float).reshape(-1)
    n = a.shape[0] - 1
    sqrt_b = jnp.sqrt(b)
    logb_prefix = jnp.cumsum(jnp.log(b))
    base = jnp.zeros((x.shape[0], n + 1), dtype=float)
    base = base.at[:, 0].set(1.0 / sqrt_b[0])
    if n > 0:
        base = base.at[:, 1].set(((x - a[0]) * base[:, 0]) / sqrt_b[1])

        def degree_body(q: int, current: jnp.ndarray) -> jnp.ndarray:
            next_col = ((x - a[q]) * current[:, q] - sqrt_b[q] * current[:, q - 1]) / sqrt_b[q + 1]
            return current.at[:, q + 1].set(next_col)

        base = lax.fori_loop(1, n, degree_body, base)

    tables = jnp.zeros((x.shape[0], n + 1, n + 1), dtype=float)
    tables = tables.at[:, :, 0].set(base)

    def derivative_body(qd: int, full: jnp.ndarray) -> jnp.ndarray:
        prev = full[:, :, qd - 1]
        current = jnp.zeros((x.shape[0], n + 1), dtype=float)
        const = jnp.exp(gammaln(qd + 1.0) - 0.5 * logb_prefix[qd])
        current = current.at[:, qd].set(const)

        def q_body(q: int, cur: jnp.ndarray) -> jnp.ndarray:
            def update_column(state: jnp.ndarray) -> jnp.ndarray:
                prev2 = state[:, q - 2]
                next_col = ((x - a[q - 1]) * state[:, q - 1] - sqrt_b[q - 1] * prev2 + qd * prev[:, q - 1]) / sqrt_b[q]
                return state.at[:, q].set(next_col)

            return lax.cond(q > qd, update_column, lambda state: state, cur)

        current = lax.fori_loop(1, n + 1, q_body, current)
        return full.at[:, :, qd].set(current)

    tables = lax.fori_loop(1, n + 1, derivative_body, tables)
    return tables


@jit
def _tensor_evaluate_with_recurrence(
    x: jnp.ndarray,
    alpha: jnp.ndarray,
    d: jnp.ndarray,
    a: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=float)
    alpha = jnp.asarray(alpha, dtype=int)
    d = jnp.asarray(d, dtype=int)
    tables = vmap(lambda xcol: _jacobi_values_all(a, b, xcol), in_axes=1, out_axes=0)(x)
    sqrt_b0 = jnp.sqrt(b[0])

    def gather_dim(qdim: int) -> jnp.ndarray:
        table = tables[qdim]
        alpha_dim = alpha[:, qdim]
        deriv_dim = d[:, qdim]
        valid = alpha_dim[None, :, None] >= deriv_dim[None, None, :]
        safe_deriv = jnp.minimum(deriv_dim[None, None, :], alpha_dim[None, :, None])
        by_degree = jnp.take(table, alpha_dim, axis=1)
        values = jnp.take_along_axis(by_degree, safe_deriv, axis=2)
        active = (alpha_dim[None, :, None] > 0) | (deriv_dim[None, None, :] > 0)
        values = jnp.where(valid, values, 0.0)
        return jnp.where(active, values * sqrt_b0, 1.0)

    factors = vmap(gather_dim)(jnp.arange(x.shape[1], dtype=int))
    return jnp.prod(factors, axis=0)


def jacobi_recurrence(n: int, alpha: float, beta: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    return JacobiPolynomials.recurrence(n, alpha, beta)


def legendre_recurrence(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jacobi_recurrence(n, 0.0, 0.0)


def chebyshev_recurrence(n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jacobi_recurrence(n, -0.5, -0.5)


def poly_eval(a: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray, n: int, d: int = 0) -> jnp.ndarray:
    return JacobiPolynomials.evaluate(a, b, x, n, d)


def mpoly_eval(
    x: jnp.ndarray,
    alpha: jnp.ndarray,
    recurrence_handle,
    d: jnp.ndarray | None = None,
) -> jnp.ndarray:
    out = JacobiPolynomials.tensor_evaluate(jnp.asarray(x, dtype=float), jnp.asarray(alpha, dtype=int), recurrence_handle, d)
    return out[:, :, 0] if out.shape[2] == 1 else out


def chebyshev_eval(x: jnp.ndarray, alpha: jnp.ndarray, d: jnp.ndarray | None = None) -> jnp.ndarray:
    return mpoly_eval(x, alpha, chebyshev_recurrence, d)


def ratio_eval(a: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray, n: int) -> jnp.ndarray:
    p = poly_eval(a, b, x, n)
    denom = jnp.where(jnp.abs(p[:, :-1]) > 0, p[:, :-1], jnp.inf)
    return p[:, 1:] / denom


def total_degree_indices(d: int, k: int) -> jnp.ndarray:
    rows: list[tuple[int, ...]] = []
    for total in range(k + 1):
        rows.extend(_compositions(total, d))
    return jnp.asarray(rows, dtype=int)


def _compositions(total: int, d: int) -> list[tuple[int, ...]]:
    if d == 1:
        return [(total,)]
    out: list[tuple[int, ...]] = []
    for first in range(total, -1, -1):
        for rest in _compositions(total - first, d - 1):
            out.append((first, *rest))
    return out


def hyperbolic_cross_indices(d: int, k: int) -> jnp.ndarray:
    if d < 1 or k < 0:
        raise ValueError("d >= 1 and k >= 0 required")
    max_degree = max(k, 0)
    grids = jnp.indices((max_degree + 1,) * d).reshape(d, -1).T
    keep = jnp.prod(grids + 1, axis=1) <= (k + 1)
    out = grids[keep]
    order = jnp.lexsort(tuple(out[:, j] for j in range(d - 1, -1, -1)))
    return out[order]


@dataclass
class PolynomialBasis:
    index_set: jnp.ndarray
    family: str = "legendre"
    alpha: float = 0.0
    beta: float = 0.0
    center: jnp.ndarray | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        self.index_set = jnp.asarray(self.index_set, dtype=int)
        if self.index_set.ndim != 2:
            raise ValueError("index_set must be 2D")
        self.family = self.family.lower()
        self.dimension = int(self.index_set.shape[1])
        if self.center is None:
            self.center = jnp.zeros(self.dimension, dtype=float)
        else:
            self.center = jnp.asarray(self.center, dtype=float).reshape(-1)
        if self.center.size != self.dimension:
            raise ValueError("center dimension mismatch")
        if self.family == "legendre":
            self.alpha = 0.0
            self.beta = 0.0
        elif self.family == "chebyshev":
            self.alpha = -0.5
            self.beta = -0.5
        self.max_degree = int(self.index_set.max(initial=0))
        self.recurrence_a, self.recurrence_b = self.get_recurrence(self.max_degree + 1)

    @classmethod
    def from_total_degree(cls, dim: int, degree: int, **kwargs: object) -> "PolynomialBasis":
        return cls(total_degree_indices(dim, degree), **kwargs)

    @classmethod
    def from_hyperbolic_cross(cls, dim: int, degree: int, **kwargs: object) -> "PolynomialBasis":
        return cls(hyperbolic_cross_indices(dim, degree), **kwargs)

    def set_normalization(self, center: jnp.ndarray, scale: float) -> None:
        center = jnp.asarray(center, dtype=float).reshape(-1)
        if center.size != self.dimension:
            raise ValueError("center dimension mismatch")
        self.center = center
        self.scale = float(scale)

    def fit_normalization_from_points(self, x: jnp.ndarray) -> None:
        x = jnp.asarray(x, dtype=float)
        center = x.mean(axis=0)
        scale = jnp.linalg.norm(x - center, axis=1).max(initial=1.0)
        self.center = center
        self.scale = float(scale if float(scale) > 0 else 1.0)

    def normalize_points(self, x: jnp.ndarray) -> jnp.ndarray:
        return (jnp.asarray(x, dtype=float) - self.center) / self.scale

    def denormalize_points(self, xn: jnp.ndarray) -> jnp.ndarray:
        return self.scale * jnp.asarray(xn, dtype=float) + self.center

    def get_recurrence(self, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.family == "legendre":
            return legendre_recurrence(n)
        if self.family == "jacobi":
            return jacobi_recurrence(n, self.alpha, self.beta)
        if self.family == "chebyshev":
            return chebyshev_recurrence(n)
        raise ValueError(f"unknown family {self.family}")

    def evaluate(self, x: jnp.ndarray, d: jnp.ndarray | None = None, assume_normalized: bool = False) -> jnp.ndarray:
        if d is None:
            d = jnp.zeros((1, self.dimension), dtype=int)
        d = jnp.atleast_2d(jnp.asarray(d, dtype=int))
        xarr = jnp.asarray(x, dtype=float)
        if xarr.shape[-1] != self.dimension:
            raise ValueError("point dimension mismatch")
        lead_shape = xarr.shape[:-1]
        xwork = xarr if assume_normalized else self.normalize_points(xarr)
        xflat = xwork.reshape(-1, self.dimension)
        p = _tensor_evaluate_with_recurrence(xflat, self.index_set, d, self.recurrence_a, self.recurrence_b)
        p = p.reshape(*lead_shape, self.index_set.shape[0], d.shape[0])
        orders = d.sum(axis=1)
        scales = jnp.where(orders > 0, self.scale**orders, 1.0)
        p = p / scales.reshape((1,) * len(lead_shape) + (1, d.shape[0]))
        return p[..., 0] if p.shape[-1] == 1 else p
