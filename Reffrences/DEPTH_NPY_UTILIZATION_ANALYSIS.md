# Diagnostic & Technical Analysis: Why `depth.npy` Fails in Downstream Tasks

## Executive Problem Statement
When running the current Marigold-360 pipeline and feeding the resulting `depth.npy` / `depth.exr` into downstream 3D reconstruction tools (such as **SPAG-4D 3D Gaussian Splatting**, mesh generators, or point cloud renderers), the 3D geometry appears severely distorted, inverted, collapsed into a narrow spherical band, or full of discontinuous tear artifacts.

This document diagnoses the exact mathematical and architectural causes and provides explicit solutions.

---

## 1. Root Causes Analysis

### Root Cause 1: Metric Scale vs Affine-Invariant Log-Depth

* **Downstream Expectation (e.g. SPAG-4D / Open3D)**:
  Expects real-world Euclidean distances or radial depths $R \in [0.1\text{m}, 50.0\text{m}]$ measured in **meters**:
  $$\mathbf{P}(\theta, \phi) = R(\theta, \phi) \cdot \begin{bmatrix} \sin\phi \cos\theta \\ \sin\phi \sin\theta \\ \cos\phi \end{bmatrix}$$
* **Marigold V2 Reality**:
  Marigold V2 produces an **affine-normalized relative log-depth map**:
  $$d_{\text{marigold}} \in [-1.0, 1.0]$$
* **The Current Flawed Heuristic in `v2_engine.py`**:
  ```python
  d = np.exp(np.clip(d, -5.0, 5.0))
  ```
  When $d \in [-1.0, 1.0]$, $\exp(d) \in [\exp(-1), \exp(1)] \approx [0.368, 2.718]$.
  * **Result**: All objects in the room (whether 0.5m or 25m away) are artificially compressed into an arbitrary range between $0.36\text{m}$ and $2.71\text{m}$.
  * Ceiling, floor, and distant walls are pulled forward, destroying the true geometry of the room.

---

### Root Cause 2: Scale and Shift Inconsistency Across the 12 Splitted Views

* In **MoGe**:
  * MoGe predicts true metric scale $s_i$ per view using its DINOv2 `[CLS]` scale head.
  * Every tile $i \in \{1, \dots, 12\}$ is in physical meters.
  * In overlapping regions between tile $A$ and tile $B$, $R_A(u, v) \approx R_B(u', v')$.
  * The Poisson solver gradients $\nabla R$ are consistent and conservative.

* In **Marigold V2**:
  * Each tile $i$ is estimated independently with its own unknown min/max percentile normalization $[\alpha_i, \beta_i]$.
  * Tile 0 (pointing at a close desk 0.8m away) maps $[0.5\text{m}, 1.5\text{m}] \to [-1, 1]$.
  * Tile 1 (pointing at a distant wall 12m away) maps $[4.0\text{m}, 15.0\text{m}] \to [-1, 1]$.
  * In the overlap zone between Tile 0 and Tile 1, Tile 0 reports $+0.8$ (representing ~1.4m), while Tile 1 reports $-0.8$ (representing ~5m)!
  * When Poisson blending solves $\Delta R = \text{div}(\mathbf{g})$, the conflicting gradient field creates massive edge cliffs, severe distortions, and seam ruptures.

---

### Root Cause 3: Spherical Ray Distance vs Planar Z-Depth Mismatch

* For equirectangular panoramas, the depth stored at pixel $(x, y)$ corresponding to longitude $\theta$ and latitude $\phi$ is the **radial distance** from the camera center $O(0,0,0)$ to the surface point:
  $$R = \|\mathbf{P}\| = \sqrt{X^2 + Y^2 + Z^2}$$
* In perspective tiles, the depth is **planar depth** $Z$ along the local camera z-axis.
* The conversion factor is:
  $$\text{ray\_scale}(u, v) = \sqrt{1 + \left(\frac{u - c_x}{f_x}\right)^2 + \left(\frac{v - c_y}{f_y}\right)^2}$$
* If relative depth is multiplied by `ray_scale` before solving scale/shift, the scale distortion is non-linearly amplified toward the corners of each perspective tile ($f_x, f_y$ for $78.5^\circ$ FOV yields corner magnification $\approx 1.4\times$).

---

## 2. Solutions & Fix Strategies

### Strategy A: Pairwise Scale-and-Shift Alignment Optimization (Recommended for Marigold-360)
Before sending the 12 perspective distance maps into the Poisson solver, optimize a set of 12 scale factors $a_i > 0$ and shifts $b_i$:

$$\min_{\{a_i, b_i\}} \sum_{i < j} \sum_{p \in \Omega_{i, j}} w(p) \left( (a_i R_i(p) + b_i) - (a_j R_j(p) + b_j) \right)^2$$

Where $\Omega_{i, j}$ is the overlapping spherical region between camera $i$ and camera $j$.
* This aligns all 12 views into a single, mutually consistent relative coordinate frame.
* The Poisson solver then operates on consistent gradients without seam tears.

### Strategy B: Metric Calibration Anchor / Metric Prior
To map the aligned 360 panorama depth $R_{\text{aligned}}$ to real-world meters for SPAG-4D:
1. Provide a single metric anchor (e.g. camera height $H_{\text{cam}} = 1.5\text{m}$ above the floor, or known room dimension).
2. Or utilize a lightweight metric predictor (like MoGe-2's scale head or Depth-Pro metric focal) to scale $R_{\text{metric}} = s \cdot R_{\text{aligned}}$.

### Strategy C: Directly Exporting Compatible Point Cloud & Disparity Formats
* For SPAG-4D: Ensure `depth.npy` is saved as true radial distance $R$ with normalized bounding or user-specified metric scale.
* For Marigold relative depth evaluation: Save disparity or normalized depth with explicit metadata headers.
