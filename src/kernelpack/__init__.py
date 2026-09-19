from jax import config as _jax_config

_jax_config.update("jax_enable_x64", True)

from . import accelerators, divfree, domain, geometry, manifold, nodes, poly, rbffd, solvers

__all__ = [
    "accelerators",
    "divfree",
    "domain",
    "geometry",
    "manifold",
    "nodes",
    "poly",
    "rbffd",
    "solvers",
]
