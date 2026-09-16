# 📸 Visual Artifacts Diagnosis & Case Study Analysis

This document provides a case study and visual diagnosis of the 3D reconstruction artifacts generated when evaluating the Marigold 360 panorama pipeline on bedroom/indoor test scenes.

---

## 1. Visual Findings & Artifact Breakdown

### 🔷 Artifact 1: The 12-Sided Faceted Shell / Icosahedral Polygon (Ceiling & Exterior View)
* **Visual Observation**:
  * When viewing the 3D reconstructed mesh/point cloud from the outside or looking up at the ceiling, the room forms a distinct **12-sided faceted polygon shell**.
  * The ceiling fan and wall moldings are bent across 12 distinct dome-like facets with visible boundary creases.
* **Underlying Code Cause**:
  * The 12 perspective cameras decompose the equirectangular sphere along an icosahedron geometry (1 Top, 5 Upper, 5 Lower, 1 Bottom).
  * Because Marigold V2 predicts affine-invariant depth, each perspective tile $i$ produces depth with an uncalibrated, independent scale factor $\alpha_i$ and shift $\beta_i$.
  * During multi-scale Poisson blending, the gradient solver is fed conflicting boundaries ($\alpha_i \neq \alpha_j$), forming a 12-sided geometric polyhedron.

---

### 🔷 Artifact 2: The "Spherical Bowl / Fishbowl" Curvature (Walls, Floor & Furniture)
* **Visual Observation**:
  * Top-down and interior views reveal that straight walls, horizontal wooden floors, TV consoles, desks, and beds are warped into a **spherical bowl / curved dome**.
  * Flat floorboards bow upward in the distance, and vertical walls bend along circular arcs toward the camera center.
* **Underlying Code Cause**:
  * In `v2_engine.py`:
    ```python
    d = np.exp(np.clip(d, -5.0, 5.0))
    ```
  * Marigold V2 produces normalized log-depth $d \in [-1.0, 1.0]$.
  * Applying $\exp(d)$ without knowing the true scene min/max percentiles forces all radial distances into:
    $$R \in [\exp(-1), \exp(1)] \approx [0.368\text{m}, 2.718\text{m}]$$
  * A back wall that was 6 meters away is pulled forward to 2.5 meters.
  * A floor that should be a flat horizontal plane ($Z = -1.5\text{m}$) is wrapped around the camera center into a constant-radius spherical bowl.

---

### 🔷 Artifact 3: "Pillowing" & Diamond Bulges Along Camera Frustum Corners
* **Visual Observation**:
  * Wall panels and bed mattresses show periodic radial bulging ("pillowing") aligned with the 12 camera view centers.
* **Underlying Code Cause**:
  * In `infer_panorama.py`:
    ```python
    ray_scale = np.sqrt(1.0 + ((u_grid - cx) / fx)**2 + ((v_grid - cy) / fy)**2)
    dist_map = (tile_depth * ray_scale).astype(np.float32)
    ```
  * Multiplying an uncalibrated affine relative depth by perspective `ray_scale` amplifies corner pixels by up to $\sim 1.4\times$ relative to center pixels, warping flat surfaces into diamond bulges.

---

## 2. Before vs After: Target Geometry Rectification

```text
       CURRENT BROKEN OUTPUT (Spherical Bowl)             DESIRED RECTIFIED OUTPUT (True Metric)
       ──────────────────────────────────────             ──────────────────────────────────────
                      .-'""'-.                                     ┌──────────────────┐
                    .'        '.                                   │                  │
                   /   Room     \                                  │   Real Room      │
                  ;  Collapsed   ;                                 │   (Flat Walls,   │
                   \  in Bowl   /                                  │   Flat Floors)   │
                    '.        .'                                   │                  │
                      '-....-'                                     └──────────────────┘
            • 12-sided faceted polyhedron                  • Smooth, continuous planar geometry
            • Distances compressed to [0.36m, 2.7m]        • True metric scale (e.g. [0.5m, 10m])
            • Floor & walls bent into a bowl               • True horizontal & vertical planes
```

---

## 3. The 3-Step Fix Summary

1. **Step 1: 12-Tile Overlap Alignment**:
   * Solve $\min_{\{a_i, b_i\}} \sum (a_i d_i - a_j d_j)^2$ across overlapping camera frustum pixels to eliminate the 12-sided faceted seams.
2. **Step 2: Clean 360 Poisson Integration**:
   * Blend the scale-aligned tiles with the Poisson solver on relative log-depth.
3. **Step 3: Real-World Metric Scaling**:
   * Scale the 360 depth into real meters using camera height calibration or user depth bounds, un-bowing the curved floor and walls into flat 3D planes.
