# 🌸 Marigold 360° Panorama Inference CLI (Dockerized)

A GPU-accelerated, self-contained **Marigold V2 360° Panorama Depth Estimation CLI** packaged into a standalone Docker container and cross-platform Python CLI. 

Combines **MoGe's 12-camera icosahedron spherical geometry splitter & multi-scale Poisson gradient solver** with **Marigold V2's Diffusion Transformer (DiT) monocular depth estimation** to produce seamless 360° panoramic depth maps (`depth.npy`, `depth.exr`, colorized `depth_vis.png`, and `pointcloud.ply`).

---

## 🔬 Architectural Research & Pipeline Audit: MoGe-v3 vs Marigold V2

### 1. Underlying Model Architectures

| Dimension | MoGe-v3 SSO (`Reffrences/MoGe_CLI`) | Marigold V2 (Current System) |
| :--- | :--- | :--- |
| **Model Family** | Feedforward Discriminative ViT + Sparse 3D Refiner | Single-Step Flow-Matching Diffusion Transformer (DiT) |
| **Backbone** | DINOv2 (ViT-L: 370M / ViT-G: 1.25B) | `Qwen/Qwen-Image-Edit-2509` + LoRA Adapters |
| **Quantization** | Native FP16 / BF16 | 4-bit NF4 (BitsAndBytes) / FP16 |
| **Primary Output** | 3D Metric Point Map $\mathbf{P}(u, v) = [X, Y, Z]^T$ in **meters** | Affine-Invariant Relative Log-Depth $d_{\text{norm}} \in [-1.0, 1.0]$ |
| **Camera Intrinsics** | Accepts exact perspective `fov_x` tensor & solves shift | Unknown / Camera-Agnostic |
| **Physical Scale** | **Resolved Metric Scale** (Real-world meters) | **Affine Invariant** (Scale $\alpha$ & shift $\beta$ ambiguous) |
| **Distance Map $R$** | Exact Euclidean norm $R = \|\mathbf{P}\|_2$ in meters | Synthetic $d_{\text{exp}} \cdot \text{ray\_scale}$ |
| **Poisson Merging** | Seamless, conservative gradients (overlap matches) | Conflicts in scale $\alpha_i$ cause edge tearing unless aligned |
| **Inference Time** | ~120ms per tile (with 3-step sparse refinement) | ~350ms per tile (Single-step DiT) |

---

### 2. What the Models Return (Raw Forward Pass vs Post-Processing)

#### A. MoGe-v3 Series (`Reffrences/MoGe_CLI`)
* **Raw `forward()` Output**:
  - `points`: Canonical affine 3D point map $(x', y', z')$ up to unknown focal and optical z-shift.
  - `normal`: Camera-space unit surface normals $[B, H, W, 3]$.
  - `mask`: Valid geometry mask $[B, H, W] \in [0, 1]$.
  - `metric_scale`: Metric scalar factor $\exp(s) \in \mathbb{R}^+$ from `[CLS]` token.
* **Post-Inference (`infer()` return)**:
  - Takes `fov_x_tensor` and applies closed-form ray-collinearity $\min \| \mathbf{r} \times \mathbf{P} \|$ to extract focal and depth shift $s_z$.
  - Multiplies points and planar z-depth by `metric_scale` to output **true metric meters**.
  - Applies 3-step self-guided sparse volumetric refinement for ultra-sharp thin structures and foliage.

#### B. Marigold V2
* **Raw `transformer.forward()` Output**:
  - Predicts latent vector field velocity $\mathbf{v}$ at single flow-matching step $t = 0.499$.
  - Latent Euler integration $\mathbf{z}_{\text{out}} = \mathbf{z}_{\text{in}} - \mathbf{v}$.
  - VAE Decoder decodes latents into a 3-channel feature map $\hat{\mathbf{Y}} \in \mathbb{R}^{B \times 3 \times H \times W}$.
* **Post-Inference (`predict_depth_batch` return)**:
  - Channel projection: $d_{\text{rel}} = \frac{1}{3} \sum_{c=1}^3 \hat{Y}_c \in [-1.0, 1.0]$.
  - Parameterization is **affine-invariant log-depth**:
    $$\log(Z(u, v)) = \alpha \cdot d_{\text{rel}}(u, v) + \beta$$
  - **Key Note**: $\alpha$ and $\beta$ are uncalibrated per-image scale and shift constants.

---

### 3. Tuning & Layer Error Findings (Why Fitting Marigold into MoGe's Pipeline Required Diagnostics)

1. **Exponential Compression Error (`np.exp(clip(d, -5, 5))` in `v2_engine.py`)**:
   * Marigold V2 outputs $d_{\text{norm}} \in [-1, 1]$. Exponentiating it directly without knowing the scene min/max percentiles $(p_2, p_{98})$ implicitly assumes $p_2 = -1, p_{98} = 1$, collapsing every 3D room into a narrow $[0.36\text{m}, 2.71\text{m}]$ shell.
2. **Multi-View Scale Conflicts in Poisson Solver**:
   * MoGe-v3 produces metric distances in real meters for all 12 perspective cameras, ensuring that tile overlap boundaries agree.
   * Marigold V2 produces 12 independent relative scales $(\alpha_i, \beta_i)$. Merging them directly produces non-conservative gradients with non-zero curl, causing seam tears and distorted walls.
   * **Solution**: A pairwise scale/shift alignment layer must optimize $\min \sum (a_i d_i - a_j d_j)^2$ over overlap masks before feeding into the Poisson solver.

> 📚 **Detailed Research & Technical Audit Documents**:
> - [MoGe-v3 CLI vs Marigold V2 Pipeline Audit](Reffrences/MoGe_CLI_VS_MARIGOLD_CLI_PIPELINE_AUDIT.md)
> - [Layer & Architecture Error Analysis](Reffrences/LAYER_AND_ARCHITECTURE_ERROR_ANALYSIS.md)
> - [MoGe vs Marigold Comprehensive Taxonomy](Reffrences/MOGE_VS_MARIGOLD_TAXONOMY.md)
> - [Architecture & Tensor Flow Deep Dive](Reffrences/ARCHITECTURE_AND_INFERENCE_DEEP_DIVE.md)
> - [Diagnostic Analysis of `depth.npy` Utilization](Reffrences/DEPTH_NPY_UTILIZATION_ANALYSIS.md)
> - [Research Papers Directory (`Reffrences/papers/`)](Reffrences/README.md)

---

## 📁 Directory Structure

```text
marigold_cli/
├── standalone_marigold/           # 100% standalone Marigold V2 360 module
│   ├── custom_deps/               # Bundled dependencies (utils3d_moge, flex_gemm)
│   ├── marigoldv2/                # Native Marigold V2 DiT architecture
│   ├── panorama/                  # 12-camera spherical geometry & Poisson solver
│   ├── utils/                     # Visualizers, EXR I/O, spherical UV mapping
│   ├── v2_engine.py               # Single-step DiT flow-matching engine
│   └── infer_panorama.py          # Standalone 360 panorama inference pipeline
├── Reffrences/                    # Comprehensive research documentation & papers
│   ├── papers/                    # Downloaded official research PDFs
│   ├── MoGe_CLI/                  # MoGe-v3 reference CLI implementation
│   ├── MoGe_CLI_VS_MARIGOLD_CLI_PIPELINE_AUDIT.md
│   ├── LAYER_AND_ARCHITECTURE_ERROR_ANALYSIS.md
│   ├── MOGE_VS_MARIGOLD_TAXONOMY.md
│   ├── ARCHITECTURE_AND_INFERENCE_DEEP_DIVE.md
│   ├── DEPTH_NPY_UTILIZATION_ANALYSIS.md
│   └── README.md                  # Reference library bibliography & index
├── app.py                         # Cross-platform CLI entrypoint
├── Dockerfile                     # Multi-stage GPU build
├── docker-compose.yml             # Container orchestration
├── pano-infrence.ipynb            # Interactive Kaggle / Colab notebook
├── moge-splat.ipynb               # Downstream 3D Gaussian Splatting notebook
├── requirements.txt               # Python package dependencies
└── README.md
```

---

## 🐳 Quick Start with Docker

### Step 1: Build Docker Image
```bash
docker build -t marigold-panorama-cli .
```

### Step 2: Run Inference via CLI

#### 1. Single Image Inference (Output `depth.npy`)
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  marigold-panorama-cli \
  -i /data/input_panorama.jpg \
  -o /data/outputs
```

#### 2. Visual Maps & Raw OpenEXR Outputs
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  marigold-panorama-cli \
  -i /data/input_panorama.jpg \
  -o /data/outputs \
  --maps
```

#### 3. Batch Folder Processing (High Quality 1024x1024 Tiles)
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  marigold-panorama-cli \
  -i /data/input_folder \
  -o /data/outputs \
  --split_resolution 1024 \
  --maps
```

---

## 📋 Full CLI Argument Reference

| Option | Shorthand | Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--input` | `-i` | *None (CLI Required)* | `Path` | **Mandatory** | Input panorama image or folder path (JPG, PNG, WEBP). |
| `--output` | `-o` | `MARIGOLD_OUTPUT` | `Path` | `./output` | Output destination directory. |
| `--checkpoint` | `-c` | `MARIGOLD_CHECKPOINT` | `String` | `huawei-bayerlab/marigold-v2-0` | Marigold V2 checkpoint path or HuggingFace repo. |
| `--base_model` | | `MARIGOLD_BASE_MODEL` | `String` | `Qwen/Qwen-Image-Edit-2509` | Base Qwen DiT model repo or local directory. |
| `--modality` | `-m` | `MARIGOLD_MODALITY` | `Choice` | `depth` | Modality: `depth`, `normals`, or `albedo`. |
| `--quantization` | `-q` | `MARIGOLD_QUANTIZATION` | `Choice` | `4bit` | DiT quantization level: `4bit`, `8bit`, `none`. |
| `--device` | | `MARIGOLD_DEVICE` | `String` | `cuda` | Execution compute device (`cuda`, `cuda:0`, `cpu`). |
| `--fp16` | | `MARIGOLD_FP16` | `Flag` | `True` | Enables FP16/BF16 precision for faster inference. |
| `--resize` | | `MARIGOLD_RESIZE` | `Int` | `None` | Max dimension ceiling (default: keep original). |
| `--split_resolution` | | `MARIGOLD_SPLIT_RESOLUTION` | `Int` | `512` | Resolution for each perspective tile (512 or 1024). |
| `--batch_size` | | `MARIGOLD_BATCH_SIZE` | `Int` | `1` | Perspective view batch size (keep 1 for low VRAM). |
| `--camera_height` | | `MARIGOLD_CAMERA_HEIGHT` | `Float` | `1.5` | Camera height above floor in meters for metric scaling (default: 1.5m). Set 0 to disable. |
| `--min_depth` | | `MARIGOLD_MIN_DEPTH` | `Float` | `0.3` | Minimum physical depth clamp in meters. |
| `--max_depth` | | `MARIGOLD_MAX_DEPTH` | `Float` | `15.0` | Maximum physical depth clamp in meters. |
| `--align / --no-align` | | `MARIGOLD_ALIGN` | `Bool` | `True` | Performs 12-tile least-squares scale & shift alignment before Poisson blending. |
| `--debug` | | `MARIGOLD_DEBUG` | `Flag` | `False` | Saves 12 perspective tiles, distance maps, and `cameras.json`. |
| `--maps` | | `MARIGOLD_MAPS` | `Flag` | `False` | Saves `depth_vis.png`, `depth.exr`, and `mask.png`. |
| `--depth_npy / --no-depth_npy` | | `MARIGOLD_DEPTH_NPY` | `Bool` | `True` | Saves primary `depth.npy` float32 array in real meters. |
| `--points_ply` | | `MARIGOLD_POINTS_PLY` | `Flag` | `False` | Exports 3D point cloud in fast binary `pointcloud.ply` format. |
