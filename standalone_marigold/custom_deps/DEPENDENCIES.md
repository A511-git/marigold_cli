# Bundled Dependencies & Upstream Provenance

To guarantee 100% build reproducibility, long-term stability, and offline execution without reliance on external Git repositories during container builds, the specialized 3D modules below have been extracted and bundled directly inside `standalone_moge/custom_deps/`.

---

## 1. `utils3d_moge`

- **Repository**: [https://github.com/EasternJournalist/utils3d-moge](https://github.com/EasternJournalist/utils3d-moge)
- **Author**: Ruicheng Wang (Author of MoGe)
- **Pinned Commit**: `62f09d58509485564e24d5d9f6aac9ee9ebc0c37`
- **Location**: `standalone_moge/custom_deps/utils3d_moge/`
- **Description**: Specialized 3D computer vision and geometry operations tailored for MoGe.
- **Key Modules Used**:
  - `utils3d.np`: Viewpoint icosahedron meshes, camera intrinsics/extrinsics conversions, unproject/project raycasting, and normal map calculation.
  - `utils3d.pt`: Depth map to 3D point map transformations on GPU, tensor coordinate mappings, and sliding window spatial operators.

---

## 2. `flex_gemm`

- **Repository**: [https://github.com/JeffreyXiang/FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM)
- **Pinned Commit**: `b2fadb29d41846c7981ade6801ffc689fae119cf`
- **Commit Details**: `Merge pull request #28 from EasternJournalist/dev/all_triton`
- **Location**: `standalone_moge/custom_deps/flex_gemm/`
- **Description**: Pure Triton GPU backend for High-Performance Sparse 3D Submanifold Convolutions and indexing ops.
- **Key Modules Used**:
  - `flex_gemm.nn.SubmanifoldConv3d`: 3D sparse convolutions in MoGe-3 residual blocks without expanding point cloud boundary dilations.
  - `flex_gemm.nn.SparsePool3d`: 3D sparse voxel grid downsampling with mean reduction.
  - `flex_gemm.nn.SparseUpsample3d`: 3D sparse voxel grid nearest-neighbor upsampling.
  - `flex_gemm.ops.NeighborCache`: 3D spatial neighbor graph index caching between encoder down-stages and decoder up-stages.

---

## 3. Maintenance & Updating Guide for Developers

If you ever need to update or audit either upstream repository in the future:

```bash
# 1. Update utils3d_moge
git clone https://github.com/EasternJournalist/utils3d-moge.git temp_utils3d
cd temp_utils3d
git checkout <COMMIT_HASH>
cd ..
cp -r temp_utils3d/utils3d_moge standalone_moge/custom_deps/utils3d_moge
rm -rf temp_utils3d

# 2. Update flex_gemm
git clone https://github.com/JeffreyXiang/FlexGEMM.git temp_flexgemm
cd temp_flexgemm
git checkout <COMMIT_HASH>
cd ..
cp -r temp_flexgemm/flex_gemm standalone_moge/custom_deps/flex_gemm
rm -rf temp_flexgemm
```
