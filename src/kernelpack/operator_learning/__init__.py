"""Operator learning package.

FastOL uses the shared kernel, domain/stencil, divergence-free interpolation,
and sparse-factor modules. Operator Lagrange prediction remains its default.
The Kernel Neural Operator implementation is available under
``kernelpack.operator_learning.kno`` and re-exported for convenience.
"""

from . import kno
from .fastol import FastOL, TrainingNormalizer, wendland_c4_matrix
from .tensor_operator import TensorWendlandOperator
from .trainable import (KernelTrainingConfig, TrainingFit, KernelSelection,
    train_compact_kernel, calibrate_compact_kernel, train_validate_fastol)
from .local_dfphs import LocalDFPHSDecoder
from .selection import epsilon_for_sparsity, select_fastol_support
from .datasets import (FastOLDataset, load_fastol_dataset, load_frozen_split,
    load_fastol_paper_dataset, read_fastol_field_gpu, fastol_physical_indices)
from .kno import (
    AnisotropicGaussianKernel,
    GaussianKernel,
    GaussianSpectralMixtureKernel,
    KernelBaseClass,
    NonstationaryGaussianKernel,
    NonstationaryGaussianSpectralMixtureKernel,
    kernels,
)
from .models import (
    KNO_DARCY_PWC,
    KNO_DARCY_TRIANGLE,
    KNO_DIFFUSION_REACTION,
    KNO_NS_3D,
    KNO_NS_PIPE,
    KNO_REG_GRID_1D,
)
from .quadratures import triangle_quad_rule
from .utils import (
    UnitGaussianNormalizer,
    create_lifted_module,
    get_batch,
    is_trainable,
    shuffle,
)

__all__ = [
    "FastOL",
    "TensorWendlandOperator",
    "KernelTrainingConfig",
    "TrainingFit",
    "KernelSelection",
    "train_compact_kernel",
    "calibrate_compact_kernel",
    "train_validate_fastol",
    "TrainingNormalizer",
    "wendland_c4_matrix",
    "LocalDFPHSDecoder",
    "epsilon_for_sparsity",
    "select_fastol_support",
    "FastOLDataset",
    "load_fastol_dataset",
    "load_frozen_split",
    "load_fastol_paper_dataset",
    "read_fastol_field_gpu",
    "fastol_physical_indices",
    "AnisotropicGaussianKernel",
    "GaussianKernel",
    "GaussianSpectralMixtureKernel",
    "KernelBaseClass",
    "kno",
    "KNO_DARCY_PWC",
    "KNO_DARCY_TRIANGLE",
    "KNO_DIFFUSION_REACTION",
    "KNO_NS_3D",
    "KNO_NS_PIPE",
    "KNO_REG_GRID_1D",
    "NonstationaryGaussianKernel",
    "NonstationaryGaussianSpectralMixtureKernel",
    "UnitGaussianNormalizer",
    "create_lifted_module",
    "get_batch",
    "is_trainable",
    "kernels",
    "shuffle",
    "triangle_quad_rule",
]
