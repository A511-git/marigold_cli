# Layer & Architecture Error Analysis: Adapting Marigold V2 into 360° Spherical Pipelines

## Executive Overview
This document analyzes the exact network layers, latent representations, tuning errors, and mathematical incompatibilities encountered when adapting **Marigold V2** into the 12-camera icosahedron spherical pipeline originally developed for **MoGe-v3**.

---

## 1. Network Layer & Tensor Flow Breakdown

### 1.1 MoGe-v3 Network Layers & Flow
```text
[Input RGB: 3xHxW]
       │
       ▼
[DINOv2 ViT Backbone] (ViT-L / ViT-G)
       │
       ├──► [CLS Token] ──► [Scale Head MLP] ──► exp(s) -> metric_scale (meters)
       │
       ▼
[Multi-Scale Neck] ──► Injects View-Plane UV coordinates [Normalized Aspect Ratio]
       │
       ├──► [Points Head: ConvStack] ──► Remap (sinh/exp) ──► Canonical Points [x', y', z']
       ├──► [Normal Head: ConvStack] ──► L2 Normalize ──► Surface Normals [Nx, Ny, Nz]
       └──► [Mask Head: ConvStack]   ──► Sigmoid ──► Confidence Mask [0, 1]
       │
       ▼
[Closed-Form Collinearity Solver: recover_focal_shift]
       │  • Solves: min || ray_dir x P ||
       │  • Recovers: f_x, f_y, shift_z
       │  • Applies: Z = z' + shift_z
       ▼
[Metric Multiplier]
       │  • P_metric = P * metric_scale
       │  • Depth_metric = Z * metric_scale
       ▼
[Sparse 3D Volumetric Refiner (3 Steps)] (MoGe-3 SSO)
       │  • Self-guided sparse point refinement for high-frequency micro-details
       ▼
[Final Output]: Points (H, W, 3) in METERS, Depth (H, W) in METERS, Intrinsics (3, 3)
```

---

### 1.2 Marigold V2 Network Layers & Flow
```text
[Input RGB: 3xHxW] (Normalized to [-1.0, 1.0])
       │
       ▼
[Qwen AutoencoderKL VAE Encoder]
       │  • Latent shape: [B, 16, 1, H/8, W/8] (5D video/image latent)
       │  • Normalized by VAE stats: (latents - latents_mean) / latents_std
       ▼
[Latent Packing Layer: _pack_latents]
       │  • Packs 2x2 spatial latent blocks -> [B, (H/16)*(W/16), 64]
       ▼
[Qwen-Image-Edit-2509 DiT Backbone + Marigold LoRA Adapters]
       │  • Rank-128 LoRA on Q, K, V, O attention projections & MLP layers
       │  • Conditioned on: Timestep t = 0.499 (Single-Step Flow Matching)
       │  • Conditioned on: Precomputed Prompt Embeddings [1, 77, 4096]
       │  • Quantization: 4-bit NF4 via BitsAndBytes
       ▼
[Output Velocity: v]
       │  • Velocity vector field in latent space
       ▼
[Latent Flow Integration]
       │  • z_out = z_in - v
       ▼
[Latent Unpacking & VAE Decode]
       │  • Unpacks 5D latents -> Unnormalizes with mean/std
       │  • Decodes via Qwen VAE Decoder -> Decoded RGB Image [B, 3, H, W]
       ▼
[FolderDepthPrediction Adapter Layer]
       │  • Channel averaging: d_norm = mean(Decoded, dim=1) in [-1.0, 1.0]
       ▼
[Output]: Normalized Affine-Invariant Log-Depth Map d_norm [-1.0, 1.0]
```

---

## 2. Granular Analysis of Tuning Errors & Pipeline Mismatches

### Error 1: The Exponential Range Compression Error

* **The Code in `v2_engine.py`**:
  ```python
  d = pred[b]                           # Output from VAE decode, range [-1.0, 1.0]
  d = np.exp(np.clip(d, -5.0, 5.0))     # Flawed attempt to convert to metric depth
  ```
* **Why this is mathematically invalid**:
  * In Marigold V2, $d_{\text{norm}} \in [-1.0, 1.0]$ represents:
    $$d_{\text{norm}} = 2 \cdot \frac{\log Z - p_2}{p_{98} - p_2} - 1$$
  * To invert this to true metric depth $Z$, we need the original scene percentiles $p_2 = \log(Z_{\min})$ and $p_{98} = \log(Z_{\max})$:
    $$\log Z = \frac{d_{\text{norm}} + 1}{2} (p_{98} - p_2) + p_2$$
    $$Z = \exp\left( \frac{d_{\text{norm}} + 1}{2} (p_{98} - p_2) + p_2 \right)$$
  * Because Marigold V2 does **not** predict $p_2$ and $p_{98}$, applying a raw $\exp(d_{\text{norm}})$ implicitly assumes $p_2 = -1$ and $p_{98} = 1$:
    $$Z \in [\exp(-1), \exp(1)] \approx [0.368\text{m}, 2.718\text{m}]$$
  * **Consequence**: An entire 30-meter hall or outdoor plaza is collapsed into a 2.3-meter shell!

---

### Error 2: The Multi-View Scale/Shift Gradient Conflict in Poisson Merging

* **The Poisson Equation**:
  $$\Delta R_{\text{pano}}(\theta, \phi) = \text{div}\left( \sum_{i=1}^{12} w_i(\theta, \phi) \nabla R_i(\theta, \phi) \right)$$
* **The Condition for Convergence to Real Geometry**:
  $$\forall i, j \text{ overlapping at } p: \quad \nabla R_i(p) \approx \nabla R_j(p)$$
* **What Happens With Marigold V2**:
  * For tile $i$: $R_i(p) = \alpha_i \cdot \hat{R}(p) + \beta_i$.
  * For tile $j$: $R_j(p) = \alpha_j \cdot \hat{R}(p) + \beta_j$.
  * Therefore, $\nabla R_i(p) = \alpha_i \nabla \hat{R}(p)$ and $\nabla R_j(p) = \alpha_j \nabla \hat{R}(p)$.
  * When $\alpha_i \neq \alpha_j$, the blended gradient field $\mathbf{g} = w_i \nabla R_i + w_j \nabla R_j$ has non-zero curl ($\text{curl}(\mathbf{g}) \neq 0$), making the field **non-conservative**.
  * The Poisson solver attempts to find the least-squares potential, which results in:
    1. Severe boundary steps and seam tears along camera frustum edges.
    2. Ringing artifacts near high-contrast corners.
    3. Global curvature warping of flat walls.

---

### Error 3: Planar-to-Spherical Ray Scale Amplification

* **The Ray Scale Formula**:
  $$\text{ray\_scale}(u, v) = \sqrt{1 + \left(\frac{u - c_x}{f_x}\right)^2 + \left(\frac{v - c_y}{f_y}\right)^2}$$
* **The Problem**:
  * In MoGe, $Z$ is in true meters, so $R = Z \cdot \text{ray\_scale}$ is the true Euclidean distance to the 3D surface point.
  * In Marigold V2, $d_{\text{norm}}$ is an affine log-depth. Multiplying an affine quantity by $\text{ray\_scale}$ applies a non-linear spatial distortion that is larger at the image corners ($\approx 1.4\times$) than at the center ($1.0\times$).
  * This breaks the affine-invariance property, because $\alpha \cdot d \cdot \text{ray\_scale}$ is no longer an affine function of $\log Z$!

---

## 3. Required Adaptation Architecture for Marigold V2 360 Panorama

To correctly adapt Marigold V2 into a 360 panorama pipeline without the above errors, the following adaptation architecture must be utilized:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                       12 PERSPECTIVE MARIGOLD V2 INFERENCE                  │
│   • Predicts 12 relative log-depth tiles: d_1, d_2, ..., d_12 in [-1, 1]    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             STEP 1: PAIRWISE SCALE & SHIFT ALIGNMENT OPTIMIZATION           │
│   • Overlap Mask: Ω_ij = Spherical intersection between tile i and tile j   │
│   • Loss: min_{a_i, b_i} Σ_{i < j} Σ_{p ∈ Ω_ij} ( (a_i d_i(p) + b_i)        │
│                                                  - (a_j d_j(p) + b_j) )^2   │
│   • Solves global 12-tile consistent relative coordinate system             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             STEP 2: MULTI-SCALE POISSON GRADIENT INTEGRATION                │
│   • Gradients are now scale-consistent across all tile boundaries           │
│   • Produces seamless equirectangular relative log-depth D_rel(θ, φ)        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│             STEP 3: METRIC ANCHOR / SCALE CALIBRATION LAYER                 │
│   • Maps D_rel to metric meters:                                            │
│     Z_metric = exp( s_metric * D_rel + shift_metric )                       │
│   • Anchor Options:                                                         │
│     a) User-specified scene scale / camera height (e.g. H = 1.5m)           │
│     b) Lightweight Metric ViT Head (e.g. MoGe-2 scale head / Depth-Pro)     │
│     c) Bounded range normalization for SPAG-4D                              │
└─────────────────────────────────────────────────────────────────────────────┘
```
