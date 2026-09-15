"""
Bundled custom dependencies for standalone MoGe.
Contains:
  - utils3d_moge: 3D computer vision and geometry operations.
  - flex_gemm: Triton GPU backend for sparse 3D submanifold convolutions.
"""

from . import utils3d_moge

try:
    from . import flex_gemm
except ImportError:
    flex_gemm = None

__all__ = ["utils3d_moge", "flex_gemm"]
