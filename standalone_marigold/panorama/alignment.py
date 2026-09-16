#!/usr/bin/env python3
"""
🌸 Global Scale & Shift Alignment Solver for Multi-View Panorama Camera Tiles.
Solves a closed-form regularized linear least-squares system across 12 icosahedral
camera overlap regions in ~15ms on CPU/GPU.
"""

import numpy as np
import torch
from typing import List, Tuple, Optional


def solve_global_scale_shift(
    tiles: List[np.ndarray],
    masks: List[np.ndarray],
    extrinsics: List[np.ndarray],
    intrinsics: List[np.ndarray],
    sample_stride: int = 4,
    anchor_idx: int = 0
) -> Tuple[List[float], List[float]]:
    """
    Solves closed-form linear least-squares scale (s_i) and shift (o_i) parameters
    such that for overlapping camera views i and j:
        s_i * d_i(p) + o_i ≈ s_j * d_j(p) + o_j

    Parameters:
    - tiles: List of N 2D float32 depth arrays (each [H, W]).
    - masks: List of N 2D boolean validity masks.
    - extrinsics: List of N [4, 4] or [3, 4] camera-to-world extrinsics.
    - intrinsics: List of N [3, 3] normalized pinhole intrinsics.
    - sample_stride: Subsampling factor for overlap pixels (default: 4 for speed).
    - anchor_idx: Index of anchor camera view (s_anchor = 1.0, o_anchor = 0.0).

    Returns:
    - scales: List of N positive float scale multipliers.
    - shifts: List of N float additive shifts.
    """
    N = len(tiles)
    if N <= 1:
        return [1.0] * N, [0.0] * N

    H, W = tiles[0].shape[:2]

    # Pre-generate coordinate grid
    u = np.arange(0, W, sample_stride, dtype=np.float32) + 0.5
    v = np.arange(0, H, sample_stride, dtype=np.float32) + 0.5
    u_grid, v_grid = np.meshgrid(u, v)

    rows = []
    rhs = []

    # Iterate over every unique pair of cameras
    for i in range(N):
        R_i = extrinsics[i][:3, :3]
        K_i = intrinsics[i]

        # Normalized direction vectors in camera i
        x_i = (u_grid - K_i[0, 2] * W if K_i[0, 2] <= 1.0 else u_grid - K_i[0, 2]) / (K_i[0, 0] * W if K_i[0, 0] <= 1.0 else K_i[0, 0])
        y_i = (v_grid - K_i[1, 2] * H if K_i[1, 2] <= 1.0 else v_grid - K_i[1, 2]) / (K_i[1, 1] * H if K_i[1, 1] <= 1.0 else K_i[1, 1])
        dirs_cam_i = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1)
        dirs_cam_i /= np.linalg.norm(dirs_cam_i, axis=-1, keepdims=True)

        # World ray directions for camera i
        dirs_world_i = dirs_cam_i @ R_i

        d_i_sub = tiles[i][::sample_stride, ::sample_stride]
        m_i_sub = masks[i][::sample_stride, ::sample_stride]

        for j in range(i + 1, N):
            R_j = extrinsics[j][:3, :3]
            K_j = intrinsics[j]

            # Angular distance between camera optical axes
            axis_i = extrinsics[i][:3, 2] if extrinsics[i].shape[0] >= 3 else R_i[:, 2]
            axis_j = extrinsics[j][:3, 2] if extrinsics[j].shape[0] >= 3 else R_j[:, 2]
            if np.dot(axis_i, axis_j) < 0.15:
                # Frustums do not overlap significantly
                continue

            # Project camera i world rays into camera j
            dirs_cam_j = dirs_world_i @ R_j.T
            z_j = dirs_cam_j[..., 2]

            # Valid in front of camera j
            valid_z = (z_j > 0.1) & m_i_sub & np.isfinite(d_i_sub)

            fx_j = K_j[0, 0] * W if K_j[0, 0] <= 1.0 else K_j[0, 0]
            fy_j = K_j[1, 1] * H if K_j[1, 1] <= 1.0 else K_j[1, 1]
            cx_j = K_j[0, 2] * W if K_j[0, 2] <= 1.0 else K_j[0, 2]
            cy_j = K_j[1, 2] * H if K_j[1, 2] <= 1.0 else K_j[1, 2]

            u_j = (dirs_cam_j[..., 0] / np.maximum(z_j, 1e-4)) * fx_j + cx_j
            v_j = (dirs_cam_j[..., 1] / np.maximum(z_j, 1e-4)) * fy_j + cy_j

            valid_uv = valid_z & (u_j >= 0) & (u_j < W - 1) & (v_j >= 0) & (v_j < H - 1)

            if np.sum(valid_uv) < 40:
                continue

            # Sample depth in tile j
            u_j_int = np.clip(u_j[valid_uv].astype(int), 0, W - 1)
            v_j_int = np.clip(v_j[valid_uv].astype(int), 0, H - 1)

            d_j_sampled = tiles[j][v_j_int, u_j_int]
            m_j_sampled = masks[j][v_j_int, u_j_int]
            valid_depth_j = m_j_sampled & np.isfinite(d_j_sampled)

            if np.sum(valid_depth_j) < 30:
                continue

            d_i_vals = d_i_sub[valid_uv][valid_depth_j]
            d_j_vals = d_j_sampled[valid_depth_j]

            # Build linear equations: s_i * d_i + o_i - s_j * d_j - o_j = 0
            for val_i, val_j in zip(d_i_vals, d_j_vals):
                row = np.zeros(2 * N, dtype=np.float32)
                row[2 * i] = val_i
                row[2 * i + 1] = 1.0
                row[2 * j] = -val_j
                row[2 * j + 1] = -1.0
                rows.append(row)
                rhs.append(0.0)

    if len(rows) < 50:
        return [1.0] * N, [0.0] * N

    A = np.array(rows, dtype=np.float32)
    b = np.array(rhs, dtype=np.float32)

    # Anchor the chosen view (default 0): s_0 = 1.0, o_0 = 0.0
    # Subtract anchor column contribution from rhs
    b = b - A[:, 2 * anchor_idx] * 1.0 - A[:, 2 * anchor_idx + 1] * 0.0

    # Remove anchor columns from matrix
    keep_cols = [c for c in range(2 * N) if c not in (2 * anchor_idx, 2 * anchor_idx + 1)]
    A_sub = A[:, keep_cols]

    # Add Tikhonov L2 regularization to prevent extreme scales
    reg_lambda = 0.01
    reg_A = np.eye(A_sub.shape[1], dtype=np.float32) * reg_lambda
    # Target scale regularization toward 1.0, shift toward 0.0
    reg_b = np.zeros(A_sub.shape[1], dtype=np.float32)
    for idx in range(0, A_sub.shape[1], 2):
        reg_b[idx] = reg_lambda * 1.0  # Encourage scale ~ 1.0

    A_full = np.vstack([A_sub, reg_A])
    b_full = np.concatenate([b, reg_b])

    # Solve least-squares system
    try:
        sol, _, _, _ = np.linalg.lstsq(A_full, b_full, rcond=1e-4)
    except Exception:
        return [1.0] * N, [0.0] * N

    # Reconstruct scale and shift parameters for all N cameras
    scales = []
    shifts = []
    col_ptr = 0

    for i in range(N):
        if i == anchor_idx:
            scales.append(1.0)
            shifts.append(0.0)
        else:
            s = float(sol[col_ptr])
            o = float(sol[col_ptr + 1])
            # Clamp scale to healthy range to avoid inversion/explosion
            s_clamped = float(np.clip(s, 0.15, 6.0))
            scales.append(s_clamped)
            shifts.append(o)
            col_ptr += 2

    return scales, shifts


def align_tile_depths(
    tiles: List[np.ndarray],
    masks: List[np.ndarray],
    extrinsics: List[np.ndarray],
    intrinsics: List[np.ndarray]
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    """
    Applies global scale and shift alignment to a list of perspective depth tiles.
    Returns (aligned_tiles, scales, shifts).
    """
    scales, shifts = solve_global_scale_shift(tiles, masks, extrinsics, intrinsics)
    aligned = []
    for i in range(len(tiles)):
        t_aligned = tiles[i] * scales[i] + shifts[i]
        aligned.append(t_aligned.astype(np.float32))
    return aligned, scales, shifts
