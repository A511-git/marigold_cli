"""
Lightweight, self-contained 3D geometry utilities for MoGe panorama inference.
Provides:
  - utils3d.np / utils3d.numpy (NumPy 3D transformations & maps)
  - utils3d.pt / utils3d.torch (PyTorch 3D transformations & maps)
"""
from . import numpy
from . import numpy as np
from . import torch
from . import torch as pt

# Re-export key functions directly at top-level
focal_to_fov = numpy.transforms.focal_to_fov
fov_to_focal = numpy.transforms.fov_to_focal

__all__ = ['numpy', 'torch', 'np', 'pt', 'focal_to_fov', 'fov_to_focal']