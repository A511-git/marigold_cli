# Pipeline Audit: MoGe-v3 CLI vs Marigold V2 CLI

This document provides a line-by-line architectural and algorithmic audit contrasting the 360 panorama inference pipeline of **MoGe-v3 CLI** (`Reffrences/MoGe_CLI`) with **Marigold V2 CLI** (`standalone_marigold`).

---

## 1. High-Level Pipeline Architecture Flow

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               360° EQUIRECTANGULAR INPUT IMAGE                                  │
└───────────────────────────────────────────────┬─────────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      STEP 1: 12-CAMERA ICOSAHEDRAL SPHERICAL SPLITTING                          │
│  • Extrinsics: 12 camera rotations (1 Top, 5 Upper, 5 Lower, 1 Bottom)                          │
│  • Intrinsics: Normalized pinhole FOV ~78.5°                                                    │
│  • Output: 12 perspective RGB tiles [12, 3, Split_Res, Split_Res]                               │
└───────────────────────┬─────────────────────────────────────────────────┬───────────────────────┘
                        │                                                 │
                        ▼                                                 ▼
┌───────────────────────────────────────────────┐ ┌───────────────────────────────────────────────┐
│          MoGe-v3 Pipeline (MoGe_CLI)          │ │       Marigold V2 Pipeline (marigold_cli)     │
├───────────────────────────────────────────────┤ ├───────────────────────────────────────────────┤
│ • Passes exact FOV: fov_x tensor              │ │ • Cannot accept FOV (Camera-agnostic DiT)     │
│ • DINOv2 ViT-L/ViT-G Backbone                 │ │ • Qwen-Image-Edit-2509 DiT Backbone           │
│ • Sparse 3D Volumetric Refiner (3 steps)      │ │ • Single-Step Flow-Matching at t=0.499        │
│ • Metric Scale Head from [CLS] Token          │ │ • 4-bit NF4 Quantization + LoRA Adapters      │
│ • Output: Metric 3D Point Map in METERS       │ │ • Output: Affine-Invariant Log-Depth [-1, 1]  │
│   P = [X, Y, Z] (Physical scale resolved)     │ │   d_norm (Arbitrary per-tile scale & shift)   │
│                                               │ │                                               │
│ • Distance Map Computation:                   │ │ • Distance Map Computation:                   │
│   R = ||P|| = sqrt(X^2 + Y^2 + Z^2)           │ │   d_exp = exp(clip(d_norm, -5, 5)) [FLAWED]   │
│   (Consistent metric meters across all tiles) │ │   dist = d_exp * ray_scale                    │
│                                               │ │   (Scale α_i & shift β_i conflict per tile!)  │
└───────────────────────┬───────────────────────┘ └───────────────────────┬───────────────────────┘
                        │                                                 │
                        ▼                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      STEP 3: MULTI-SCALE POISSON GRADIENT MERGING                               │
│  • Solves: ΔR_pano = div(g_weighted)                                                            │
│  • MoGe-v3: Seamless, conservative gradients -> Metric Equirectangular Depth in Meters          │
│  • Marigold V2: Gradient field conflicts -> Tear seams, boundary steps, geometry collapse       │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Line-by-Line Pipeline Comparison

### Stage 1: Perspective Tile Inference

#### MoGe-v3 (`Reffrences/MoGe_CLI/standalone_moge/infer_panorama.py:L227-L256`):
```python
# 1. Compute exact FOV for each perspective camera
fov_x, _ = np.rad2deg(utils3d.np.intrinsics_to_fov(np.array(splitted_intrinsics[i:i + batch_size])))
fov_x_tensor = torch.tensor(fov_x, dtype=torch.float32, device=device)

# 2. Feed FOV directly into MoGe-v3 with sparse refinement
infer_kwargs = {
    'fov_x': fov_x_tensor,
    'resolution_level': resolution_level,
    'apply_mask': False,
    'refine_steps': refine_steps,
    'use_fp16': use_fp16,
}
output = model.infer(image_tensor, **infer_kwargs)

# 3. True Euclidean distance from metric 3D point map
distance_map = output['points'].norm(dim=-1).cpu().numpy()
```
* **Why it succeeds**: 
  1. MoGe-v3 knows the exact camera geometry (`fov_x_tensor`).
  2. The point map `output['points']` is in **true physical meters**.
  3. `.norm(dim=-1)` calculates exact spherical Euclidean distance $R = \sqrt{X^2 + Y^2 + Z^2}$ in meters.
  4. Tile overlaps match in physical units.

---

#### Marigold V2 (`standalone_marigold/infer_panorama.py:L145-L167` & `v2_engine.py:L388-L391`):
```python
# 1. Marigold V2 predicts depth batch (Qwen DiT Flow Matching)
splitted_depth_maps = engine.predict_depth_batch(splitted_images_bgr, batch_size=batch_size)

# In v2_engine.py:
for b in range(B):
    d = pred[b]  # Raw VAE decode average in [-1.0, 1.0]
    d = np.exp(np.clip(d, -5.0, 5.0))  # <-- CRITICAL TUNING ERROR
    depths.append(d)

# In infer_panorama.py:
ray_scale = np.sqrt(1.0 + ((u_grid - cx) / fx)**2 + ((v_grid - cy) / fy)**2)
dist_map = (tile_depth * ray_scale).astype(np.float32)
```
* **Why fitting Marigold into MoGe's pipeline failed**:
  1. **No Camera FOV Awareness**: Marigold V2 was trained with text embeddings, not camera intrinsics. It cannot condition on `fov_x`.
  2. **Tuning Error on Output Mapping**: Marigold V2 outputs affine log-depth $d \in [-1, 1]$. Exponentiating it directly via $\exp(d)$ compresses all distances into $[\exp(-1), \exp(1)] \approx [0.368, 2.718]$.
  3. **Tile-to-Tile Scale Divergence**:
     * Perspective Tile 0 (pointing at a foreground laptop 0.6m away) maps $[0.4\text{m}, 1.2\text{m}] \to [-1, 1] \to [0.36, 2.71]$.
     * Perspective Tile 1 (pointing at a rear window 10m away) maps $[3.0\text{m}, 15.0\text{m}] \to [-1, 1] \to [0.36, 2.71]$.
     * In their overlapping boundary, Tile 0 gives $+0.5 \to 1.65$, while Tile 1 gives $-0.5 \to 0.60$.
     * The Poisson solver is fed two conflicting distance values for the exact same physical ray!

---

## 3. Structural Comparison of Intermediate Representations

| Property | MoGe-v3 (`MoGe_CLI`) | Marigold V2 (`marigold_cli`) |
| :--- | :--- | :--- |
| **Input Conditioning** | RGB Image + `fov_x` (Degrees) | RGB Image + Precomputed Prompt Embeddings |
| **Intermediate Layers** | DINOv2 Pyramidal Features + 3D Sparse Refiner | Qwen 5D Latents + DiT Multi-Head Cross Attention |
| **Output Layer** | Geometry Heads (Point/Normal/Mask/Scale) | Single-Step Latent Velocity Subtraction + VAE Decode |
| **Tile Output Format** | Metric 3D Point Coordinates $(X, Y, Z)$ [Meters] | Normalized Affine Log-Depth $[-1, 1]$ [Dimensionless] |
| **Distance Map $R$** | Exact Euclidean Norm $\|\mathbf{P}\|_2$ [Meters] | Synthetic `exp(d) * ray_scale` [Dimensionless] |
| **Inter-Tile Consistency** | **Physically Consistent** across all 12 views | **Independently Scaled** per view |
| **Poisson Gradient Solvability** | Exact, integrable conservative field | Non-conservative, conflicting edge gradients |
| **SPAG-4D 3DGS Usability** | **Directly usable** (Accurate metric 3D room) | **Requires Pairwise Alignment & Metric Scale** |
