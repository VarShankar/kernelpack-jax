"""Optional reusable sparse GPU factorizations; no CUDA dependency on import."""

from .cudss import CudssFactor, CudssUnavailableError, cudss_available
from .products import csr_edge_inner_products, PaddedCSRAction

__all__ = ["CudssFactor", "CudssUnavailableError", "cudss_available", "csr_edge_inner_products", "PaddedCSRAction"]
