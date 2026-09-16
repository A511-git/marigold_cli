# Architecture & Inference Deep Dive: MoGe vs Marigold V2

## 1. What the Model Returns: Raw Forward Pass vs Post-Processing

### 1.1 MoGe (Model Forward vs Inference Return)

#### A. Raw `model.forward(image, num_tokens)`:
* **Inputs**:
  * `image`: `torch.Tensor` of shape `[B, 3, H, W]`, normalized to `[0.0, 1.0]`.
  * `num_tokens`: Target ViT patch tokens count (e.g. 1200 - 2500).
* **Internals**:
  1. Base ViT patches computed from `(num_tokens * aspect_ratio)^0.5`.
  2. DINOv2 forward extracts intermediate multi-scale feature maps + `cls_token`.
  3. Feature pyramid neck injects normalized view-plane UV coordinate grids.
  4. ConvStack heads predict raw output channels:
     - `points_head`: 3 channels (remapped through `torch.sinh` or `torch.exp`).
     - `normal_head`: 3 channels (normalized with `F.normalize`).
     - `mask_head`: 1 channel (passed through `torch.sigmoid`).
     - `scale_head`: MLP on `cls_token` (passed through `torch.exp`).
* **Raw Return Dictionary**:
  ```python
  {
      'points': torch.Tensor,       # Shape: [B, H, W, 3], canonical affine points
      'normal': torch.Tensor,       # Shape: [B, H, W, 3], camera-space normals
      'mask': torch.Tensor,         # Shape: [B, H, W], float in [0.0, 1.0]
      'metric_scale': torch.Tensor  # Shape: [B], metric scalar in meters (MoGe-2/3)
  }
  ```

#### B. Post-Forward `model.infer(...)`:
1. **Ray-Collinearity Intrinsics & Shift Recovery**:
   * Calls `recover_focal_shift(points, mask_binary)`.
   * Solves least-squares optimization finding $(f_x, f_y, s_z)$ minimizing ray divergence.
   * `points[..., 2] += shift`: Transforms affine z into camera-space optical z.
2. **Metric Scaling**:
   * `points = points * metric_scale`
   * `depth = points[..., 2] = depth * metric_scale` (in meters)
3. **Pinhole Projection Enforcement**:
   * `points = utils3d.pt.depth_map_to_point_map(depth, intrinsics)`
4. **Final Return**:
   ```python
   {
       'points': np.ndarray,      # [H, W, 3] Float32 (Real-world metric meters)
       'depth': np.ndarray,       # [H, W] Float32 (Planar Z-depth in meters)
       'intrinsics': np.ndarray,  # [3, 3] Float32 (Normalized camera intrinsics)
       'mask': np.ndarray,        # [H, W] Bool (Valid geometry mask)
       'normal': np.ndarray       # [H, W, 3] Float32 (OpenCV camera normals)
   }
   ```

---

### 1.2 Marigold V2 (DiT Flow-Matching vs Post-Processing)

#### A. Raw `transformer.forward(...)`:
* **Inputs**:
  * `hidden_states`: `[B, (lat_h/2)*(lat_w/2), C_lat*4]` (Packed 5D VAE latents from Qwen VAE).
  * `timestep`: `torch.full((B,), 499.0) / 1000.0` (Fixed single-step flow time $t=0.499$).
  * `encoder_hidden_states`: Precomputed prompt embeddings `[B, 77, 4096]`.
  * `encoder_attention_mask`: Prompt mask `[B, 77]`.
* **Raw Return**:
  * `velocity`: `[B, (lat_h/2)*(lat_w/2), C_lat*4]` (Vector field velocity $\mathbf{v}$ in latent space).

#### B. Latent Integration & VAE Decoding:
1. **Unpack Velocity**: Unpacks into 5D latent tensor `[B, C_lat, 1, lat_h, lat_w]`.
2. **Euler Flow Step**:
   $$\mathbf{z}_{\text{decoded\_latents}} = \mathbf{z}_{\text{input\_latents}} - \mathbf{v}$$
3. **VAE Decode**:
   $$\hat{\mathbf{Y}} = \text{VAE.decode}\left(\frac{\mathbf{z}_{\text{decoded\_latents}}}{\text{std\_inv}} + \text{mean}\right) \in \mathbb{R}^{B \times 3 \times H \times W}$$
4. **Channel Projection / Averaging (`FolderDepthPrediction`)**:
   $$d_{\text{rel}}(u, v) = \frac{1}{3}\sum_{c=1}^3 \hat{Y}_c(u, v) \quad \in [-1.0, 1.0]$$
5. **Output**:
   * Raw affine-invariant log-depth $d_{\text{rel}}$.
   * **Note**: No camera intrinsics $(f_x, f_y)$ and no metric scale factor $s$ exist!

---

## 2. The Spherical 360 Pipeline & Poisson Solver Interaction

### 2.1 12-Camera Icosahedron Splitting
To convert an equirectangular panorama ($2:1$ aspect ratio, $360^\circ \times 180^\circ$) into perspective tiles:
1. Places 12 virtual perspective cameras at the center of an icosahedron (regular 20-faced polyhedron):
   - 1 top camera ($\text{pitch} = +90^\circ$)
   - 5 upper cameras ($\text{pitch} = +26.57^\circ$, azimuth steps $72^\circ$)
   - 5 lower cameras ($\text{pitch} = -26.57^\circ$, azimuth steps $72^\circ$, offset $36^\circ$)
   - 1 bottom camera ($\text{pitch} = -90^\circ$)
2. Each camera has FOV $\approx 78.5^\circ$ with overlap between adjacent frustums.

### 2.2 Planar Z-Depth vs Spherical Ray Distance
* In perspective view $i$, a pixel $(u, v)$ has optical axis planar depth $Z_i(u, v)$.
* The spherical distance from the panorama center to the 3D surface point is:
  $$R_i(u, v) = Z_i(u, v) \cdot \sqrt{1 + \left(\frac{u - c_x}{f_x}\right)^2 + \left(\frac{v - c_y}{f_y}\right)^2}$$

```
                +-------------------+ (Planar Image Plane)
                |       . (u,v)     |
                |      /            |
                |     / R           |
                |    /              |
                |   /               |
                |  /  Z             |
                | /                 |
                +-------------------+
                |/
                O (Camera Center)
```

### 2.3 Multi-Scale Poisson Gradient Solver
* The 12 distance maps $R_1, \dots, R_{12}$ overlap on the sphere.
* At every spherical coordinate $(\theta, \phi)$, the solver extracts spatial gradients:
  $$\mathbf{g}(\theta, \phi) = \sum_{i=1}^{12} w_i(\theta, \phi) \cdot \nabla R_i(\theta, \phi)$$
* Solves the Poisson equation on the equirectangular grid:
  $$\Delta R_{\text{pano}} = \text{div}(\mathbf{g})$$
* **Fundamental Assumption of Poisson Blending**:
  All overlapping tiles $R_i$ must share the **same global metric scale and physical units (meters)**. If tile $i$ is scaled by $\alpha_i$ and tile $j$ is scaled by $\alpha_j$ ($\alpha_i \neq \alpha_j$), the gradient field $\mathbf{g}$ becomes inconsistent and non-integrable, resulting in severe boundary artifacts and surface warping!
