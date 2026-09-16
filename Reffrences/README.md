# 📚 Research & Reference Library

This directory contains comprehensive research documentation, official research papers, architectural breakdowns, pipeline audits, layer error analyses, and comparative taxonomies for **MoGe** (MoGe-1, MoGe-2, MoGe-3) and **Marigold** (Marigold V1, Marigold V2).

---

## 📑 Table of Contents

1. [Papers Repository (`Reffrences/papers/`)](#-papers-repository)
2. [Technical Documentation & Pipeline Audits](#-technical-documentation--pipeline-audits)
3. [Architecture & Pipeline Comparison Summary](#-architecture--pipeline-comparison-summary)
4. [Downstream 3D Reconstruction & SPAG-4D Compatibility](#-downstream-3d-reconstruction--spag-4d-compatibility)

---

## 📄 Papers Repository

| Paper Title | Year / Venue | Authors | File | Key Innovation |
| :--- | :--- | :--- | :--- | :--- |
| **Marigold V2: Revisiting Diffusion Transformers for Monocular Depth Estimation** | 2026 (SIGGRAPH Asia / ACM TOG) | Igor Pavlovic, Thiemo Wandel, Anton Obukhov et al. | [`papers/marigold_v2_siggraph_asia_2026.pdf`](papers/marigold_v2_siggraph_asia_2026.pdf) | Single-step Flow-Matching DiT (Qwen-Image-Edit-2509) with 4-bit QLoRA & Sinkhorn block-matching loss. |
| **MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision** | 2025 (CVPR Oral) | Ruicheng Wang, Sicheng Xu, Jiaolong Yang et al. | [`papers/moge_v1_cvpr2025.pdf`](papers/moge_v1_cvpr2025.pdf) | Canonical affine 3D point map representation, closed-form focal/shift ray collinearity solver. |
| **MoGe-2: Accurate Monocular Geometry with Metric Scale and Sharp Details** | 2025 | Ruicheng Wang et al. | [`papers/moge_v2_2025.pdf`](papers/moge_v2_2025.pdf) | Decoupled metric scale regression on CLS token, delivering true metric point clouds and depth in meters. |
| **MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement** | 2026 | Ruicheng Wang et al. | [`papers/moge_v3_2026.pdf`](papers/moge_v3_2026.pdf) | Self-guided sparse volumetric refinement (SSO) for micro-details, hair, foliage, and thin structures. |
| **Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation (Marigold V1)** | 2024 (CVPR) | Bingxin Ke, Anton Obukhov, Konrad Schindler et al. | [`papers/marigold_v1_cvpr2024.pdf`](papers/marigold_v1_cvpr2024.pdf) | Zero-shot monocular depth from fine-tuned Stable Diffusion U-Net. |

---

## 🔬 Technical Documentation & Pipeline Audits

* 💡 [**Plain-English Simplified Guide (`SIMPLIFIED_EXPLANATION.md`)**](SIMPLIFIED_EXPLANATION.md)  
  *Zero-jargon, intuitive explanation of affine vs metric depth, why 12-camera splitting fails with rubber-sheet numbers, and how to fix `depth.npy` for SPAG-4D.*

* 📘 [**Pipeline Audit: MoGe-v3 CLI vs Marigold V2 CLI (`MoGe_CLI_VS_MARIGOLD_CLI_PIPELINE_AUDIT.md`)**](MoGe_CLI_VS_MARIGOLD_CLI_PIPELINE_AUDIT.md)  
  *Granular, line-by-line comparison of how MoGe-v3 (`Reffrences/MoGe_CLI`) executes 12-camera splitting, distance map calculation, and Poisson gradient merging vs Marigold V2 (`standalone_marigold`).*

* 📙 [**Layer & Architecture Error Analysis (`LAYER_AND_ARCHITECTURE_ERROR_ANALYSIS.md`)**](LAYER_AND_ARCHITECTURE_ERROR_ANALYSIS.md)  
  *In-depth diagnostic of network layers, latent channels, LoRA injection, exponential compression tuning errors, and non-conservative Poisson gradient tears.*

* 📗 [**MoGe vs Marigold Taxonomy (`MOGE_VS_MARIGOLD_TAXONOMY.md`)**](MOGE_VS_MARIGOLD_TAXONOMY.md)  
  *Side-by-side comparison matrix of backbones, loss functions, parameterizations, coordinate systems, and inference speeds.*

* 📕 [**Architecture & Inference Deep Dive (`ARCHITECTURE_AND_INFERENCE_DEEP_DIVE.md`)**](ARCHITECTURE_AND_INFERENCE_DEEP_DIVE.md)  
  *Exact trace of raw model forward outputs (`points`, `mask`, `normals`, `metric_scale`, `velocity`, `latent_states`) vs post-inference steps (`recover_focal_shift`, VAE decode, channel projection).*

* 📓 [**Depth.npy Utilization & Diagnostic Analysis (`DEPTH_NPY_UTILIZATION_ANALYSIS.md`)**](DEPTH_NPY_UTILIZATION_ANALYSIS.md)  
  *Root cause analysis of why Marigold's `depth.npy` fails in downstream 3D Gaussian Splatting (SPAG-4D) / 3D point cloud backprojection, with mathematical solutions.*

* 📸 [**Visual Artifacts & Case Study Analysis (`VISUAL_ARTIFACTS_AND_CASE_STUDY.md`)**](VISUAL_ARTIFACTS_AND_CASE_STUDY.md)  
  *Direct visual diagnosis of the 12-sided faceted shell, bowl curvature, and pillowing artifacts from test bedroom reconstruction views.*

* 🛠️ [**Professional Solutions & Community Implementations (`PROFESSIONAL_FIXES_AND_COMMUNITY_SOLUTIONS.md`)**](PROFESSIONAL_FIXES_AND_COMMUNITY_SOLUTIONS.md)  
  *Academic formulations (360MonoDepth, Metric3D, ZoeDepth), closed-form linear scale/shift alignment solver ($A x = b$), metric anchoring, and fast binary PLY exporter (< 50ms).*

---

## 🧠 Architecture & Pipeline Comparison Summary

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MODEL FORWARD PASS                                     │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│         MoGe-v3 SSO (MoGe_CLI)           │                 Marigold V2                 │
├──────────────────────────────────────────┼─────────────────────────────────────────────┤
│ • Canonical Affine Points: [B, H, W, 3]  │ • Latent Velocity: [B, Tokens, Channels]    │
│ • Surface Normals: [B, H, W, 3]          │ • Text Conditioning: [B, 77, 4096]          │
│ • Valid Mask: [B, H, W]                  │ • Time Step: t = 0.499                      │
│ • Metric Scale Head from [CLS] Token     │ • Single-Step Flow: Latents - Velocity      │
│ • Sparse 3D Refiner (3 Refine Steps)     │ • 4-bit NF4 Quantization + LoRA Adapters    │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               POST-INFERENCE OUTPUT                                    │
├──────────────────────────────────────────┬─────────────────────────────────────────────┤
│ • Metric 3D Point Map in Meters          │ • Affine-Invariant Log-Depth [-1.0, 1.0]    │
│ • Metric Planar Z-Depth in Meters        │ • No Camera Intrinsics (f_x, f_y unknown)   │
│ • Radial Distance: R = ||P|| in METERS   │ • Relative / Arbitrary Scale & Shift        │
│ • Overlaps are physically identical      │ • Overlaps have conflicting scale/shift     │
│ • 3D Points Ready for SPAG-4D / Open3D   │ • Requires Scale Calibration for 3D Metrics │
└──────────────────────────────────────────┴─────────────────────────────────────────────┘
```
