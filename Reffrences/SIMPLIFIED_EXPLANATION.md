# 💡 Plain-English Guide: Affine Depth, MoGe vs Marigold, and How to Fix the 3D Output

---

## 1. What is "Affine Depth" in Simple Words?

Imagine taking a photo of a room with a table, a chair, and a far wall.

### 📏 1. Metric Depth (The Laser Measuring Tape) — Used by MoGe
* **How it works**: MoGe acts like a laser tape measure.
* **The output**: 
  * Table = **1.2 meters** away
  * Chair = **2.5 meters** away
  * Wall = **8.0 meters** away
* **Real-world meaning**: Every number has a real physical unit (meters). If you put this into 3D software (like SPAG-4D or Blender), the room has real-world dimensions.

---

### 🎨 2. Affine Depth (The Rubber Sheet Drawing) — Used by Marigold
* **How it works**: Marigold acts like an artist drawing on a rubber sheet.
* **The output**:
  * It knows the table is closer than the chair, and the chair is closer than the wall.
  * But it **does not know real-world units**. It just gives numbers normalized between `-1.0` (closest object in photo) and `+1.0` (farthest object in photo).
* **The "Affine" problem**:
  * An "affine" change means **stretching** (multiplying by a scale) or **shifting** (adding a number).
  * If you stretch or shift a rubber sheet, the relative order stays the same (the table is still in front of the chair), but the actual distance numbers are completely arbitrary!

---

## 2. Why Did Our Pipeline Break When We Replaced MoGe with Marigold?

We took a 360° panorama, sliced it into **12 perspective camera tiles**, ran depth on each tile, and stitched them back together.

Here is why it worked for MoGe, but failed for Marigold:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE 12-CAMERA PROBLEM                                  │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│             MoGe (Metric)                │             Marigold (Affine)               │
├──────────────────────────────────────────┼─────────────────────────────────────────────┤
│ • Tile 1 (looks at table):               │ • Tile 1 (looks at table):                  │
│   Measures: 0.8m to 1.5m                 │   Maps [0.8m, 1.5m] -> [-1.0, +1.0]         │
│                                          │                                             │
│ • Tile 2 (looks at far wall):            │ • Tile 2 (looks at far wall):               │
│   Measures: 4.0m to 10.0m                │   Maps [4.0m, 10.0m] -> [-1.0, +1.0]        │
│                                          │                                             │
│ • Where Tile 1 and Tile 2 OVERLAP:       │ • Where Tile 1 and Tile 2 OVERLAP:          │
│   Both tiles say the overlap is ~1.5m.   │   Tile 1 says "+1.0" (farthest in its view) │
│   ✅ Perfect seamless match!             │   Tile 2 says "-1.0" (closest in its view)  │
│                                          │   ❌ Massive conflict! (+1.0 vs -1.0)       │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
```

When the stitcher (Poisson solver) tries to blend Tile 1 and Tile 2:
1. For **MoGe**: The numbers match because 1.5 meters is 1.5 meters everywhere.
2. For **Marigold**: The stitcher gets conflicting numbers (`+1.0` from one tile vs `-1.0` from the next tile for the exact same point in the room). This causes **seam tears, sharp edge cuts, and wavy walls**.

---

## 3. Why Did Downstream 3D Tools (like SPAG-4D) Fail on `depth.npy`?

Downstream 3D Gaussian Splatting tools like SPAG-4D expect **real metric distance in meters** so they can build a real 3D room.

In our current code (`v2_engine.py`), we had this line:
```python
d = np.exp(np.clip(d, -5.0, 5.0))
```

### What happened mathematically:
* Marigold produces numbers between `-1.0` and `+1.0`.
* Taking $\exp(-1.0) \approx \mathbf{0.36}$ and $\exp(+1.0) \approx \mathbf{2.71}$.
* **The Result**: Every object in the whole 360 scene (whether a tiny pen on a desk 0.3m away or a skyscraper 100m away) was forced into a tiny, narrow shell between **0.36 meters and 2.71 meters**!
* In 3D viewing, the room looks like a squished hollow egg where all walls and furniture are pulled right in front of your face.

---

## 4. How Do We Fix It? (Step-by-Step Action Plan)

To make Marigold V2 produce clean, usable 360 depth and 3D point clouds, we need **3 simple fixes**:

```text
  [12 Marigold Tile Predictions]
                │
                ▼
  [Step 1: Tile Alignment Layer] ──► Adjusts all 12 tiles so overlapping areas agree
                │
                ▼
  [Step 2: Smooth 360 Stitching] ──► Poisson blending creates a smooth 360 panorama map
                │
                ▼
  [Step 3: Real-World Scale Layer] ──► Stretches the relative map into real meters (e.g. 0.5m to 15m)
                │
                ▼
  [Output: Clean depth.npy ready for SPAG-4D & 3DGS]
```

### Fix 1: Align the 12 Tiles Before Stitching
Before blending, compare the overlapping pixels between neighboring cameras. Scale and shift the tiles mathematically so that where Tile 1 meets Tile 2, their numbers match.

### Fix 2: Remove the Arbitrary $\exp(\text{clip}(d))$
Do not blindly exponentiate `[-1, 1]`. Instead, keep the clean 360 relative depth map intact.

### Fix 3: Add a Metric Scale Anchor
Convert the relative 360 map to real meters using one of these simple approaches:
1. **Camera Height Anchor (Easiest)**: Assume the camera is on a tripod ~1.5 meters above the floor. Use the floor pixels to automatically scale the entire room to real meters.
2. **User Bounding Range**: Let the user specify the scene type (e.g. `indoor_room: 0.5m - 8m` or `outdoor: 1m - 50m`).
3. **Lightweight Scale Predictor**: Use a lightweight model (or MoGe's scale head) to predict one single metric multiplier for the whole scene.

---

## 5. Summary Cheat-Sheet

| Term | What it Means | Real-World Analogy |
| :--- | :--- | :--- |
| **Metric Depth** | Real distance measured in meters. | A laser tape measure. |
| **Affine Depth** | Relative depth without real-world scale (up to stretch & shift). | A drawing on a stretchable rubber sheet. |
| **Overlap Conflict** | Two cameras seeing the same wall with different relative scale numbers. | Two rulers that use different, unmarked units. |
| **The Fix** | Align the 12 tiles in the overlap zones, then scale the whole 360 map to real meters. | Calibrating the rulers so they match before measuring the room. |
