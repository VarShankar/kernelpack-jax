"""Half-integer Matérn implementation extracted from symmetric collocation."""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import factorial, isfinite
import jax
import jax.numpy as jnp


@lru_cache(None)
def _radial_polynomials(p):
    polynomial = {j: Fraction(factorial(2*p-j)*factorial(p)*2**j,
                               factorial(2*p)*factorial(p-j)*factorial(j))
                  for j in range(p+1)}
    result = []
    for _ in range(2*p+1):
        result.append(tuple(sorted(polynomial.items())))
        new = {}
        for exponent, coefficient in polynomial.items():
            new[exponent-1] = new.get(exponent-1, 0)-coefficient/2
            if exponent:
                new[exponent-2] = new.get(exponent-2, 0)+exponent*coefficient/2
        polynomial = {e: c for e, c in new.items() if c}
    return tuple(result)


def matern_squared(q, p=2, shape=1.0):
    """Evaluate from squared distance; Cartesian AD through order 2p, nu=p+1/2."""
    polynomials = _radial_polynomials(p)
    def radial(q, order):
        s = shape*jnp.sqrt(jnp.where(q > 0, q, 1.0))
        s = jnp.where(q > 0, s, 0.0)
        safe_s = jnp.where(q > 0, s, 1.0)
        e = jnp.exp(-s)
        terms = polynomials[order]
        polynomial = sum(float(c)*safe_s**power for power, c in terms)
        limit = float(dict(terms).get(0, 0)) if order <= p else 0.0
        return jnp.where(q > 0, shape**(2*order)*e*polynomial,
                         shape**(2*order)*limit)
    functions = [None]*(2*p+1)
    functions[2*p] = lambda q: radial(q, 2*p)
    def make_function(order):
        @jax.custom_jvp
        def function(q):
            return radial(q, order)
        def rule(primals, tangents):
            (q,), (dq,) = primals, tangents
            return functions[order](q), functions[order+1](q)*dq
        function.defjvp(rule)
        return function
    for order in reversed(range(2*p)):
        functions[order] = make_function(order)
    return functions[0](q)


def _matern_half_integer(x, y, p, shape, scales):
    z = x-y if scales is None else (x-y)/scales
    return matern_squared(jnp.dot(z,z), p=p, shape=shape)


@dataclass(frozen=True)
class MaternKernel:
    """Half-integer Matérn; coordinate AD through order 2p, nu=p+1/2."""
    nu: float = 2.5
    shape: float = 1.0
    scales: object = None

    def __post_init__(self):
        p = self.nu-0.5
        if not isfinite(p) or p < 0 or p != int(p):
            raise ValueError("nu must be a positive half integer: 0.5, 1.5, 2.5, ...")
        if not isfinite(self.shape) or self.shape <= 0:
            raise ValueError("shape must be finite and positive")
        if self.scales is not None and not bool(jnp.all(jnp.isfinite(jnp.asarray(self.scales)) & (jnp.asarray(self.scales) > 0))):
            raise ValueError("coordinate scales must be finite and positive")

    @property
    def max_functional_order(self):
        return int(self.nu-0.5)

    def __call__(self,x,y):
        return _matern_half_integer(x,y,self.max_functional_order,self.shape,
            None if self.scales is None else jnp.asarray(self.scales))


def matern_c4(x,y,*,shape=1.0,scales=None):
    return _matern_half_integer(x,y,2,shape,scales)


matern_c4.max_functional_order = 2
