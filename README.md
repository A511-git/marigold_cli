# Marigold 360° Panorama Inference CLI (Dockerized)

A GPU-accelerated, self-contained **Marigold 360° Panorama Depth Estimation CLI** packaged into a standalone Docker container. 

Combines **MoGe's 12-camera icosahedron spherical splitter & multi-scale Poisson gradient solver** with **Marigold's Diffusion-based monocular depth estimation** to produce seamless, artifact-free 360° panoramic depth maps (`depth.npy`, `depth.exr`, and colorized `depth_vis.png`).

---

## 📁 Directory Structure

```text
marigold_pano_cli/
├── standalone_marigold/           # 100% independent Marigold-360 core module
│   ├── custom_deps/               # Bundled dependencies (utils3d_moge, flex_gemm, DEPENDENCIES.md)
│   ├── marigoldv2/                # Native Marigold V2 architecture & network graph
│   ├── panorama/                  # Spherical geometry, multi-view splitter & merger
│   ├── scripts/                   # Native Marigold inference & downloading scripts
│   ├── utils/                     # Exporters, visualizers (vis.py, io.py, etc.)
│   └── infer_panorama.py          # Canonical Marigold-360 Panorama CLI script
├── app.py                         # Cross-platform CLI entrypoint
├── Dockerfile                     # Multi-stage build (builder -> slim runtime)
├── docker-compose.yml             # Compose service definitions
├── requirements.txt               # Standalone dependencies
├── .dockerignore                  # Build context exclusions
├── .gitignore                     # Git tracking exclusions
└── README.md
```

---

## 🐳 Quick Start with Docker

### Step 1: Build Docker Image
```bash
cd marigold_pano_cli
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

#### 4. Debug Mode (Save 12 Splitted Perspective Tiles & Camera Metadata)
```bash
docker run --rm --gpus all \
  -v $(pwd)/checkpoints:/checkpoints \
  -v $(pwd)/data:/data \
  marigold-panorama-cli \
  -i /data/input_panorama.jpg \
  -o /data/outputs \
  --debug
```

---

## 📋 Full CLI Argument & Environment Variable Reference

All options can be configured via **CLI flags**, **Docker environment variables (`-e`)**, or inside `docker-compose.yml`. CLI flags always take highest precedence.

| Option | Shorthand | Environment Variable | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `--input` | `-i` | *None (CLI Required)* | `Path` | **Mandatory** | Input panorama image or folder path (JPG, PNG, WEBP). |
| `--output` | `-o` | `MARIGOLD_OUTPUT` | `Path` | `./output` | Output destination directory. |
| `--checkpoint` | `-c` | `MARIGOLD_CHECKPOINT` | `String` | `huawei-bayerlab/marigold-v2-0` | Marigold checkpoint path or HuggingFace repo. |
| `--device` | | `MARIGOLD_DEVICE` | `String` | `cuda` | Execution device (`cuda`, `cuda:0`, `cpu`). |
| `--fp16` | | `MARIGOLD_FP16` | `Flag` | `False` | Enables FP16 half precision for faster inference. |
| `--diffusers` | | `MARIGOLD_DIFFUSERS` | `Flag` | `False` | Uses HuggingFace Diffusers Marigold pipeline backend. |
| `--resize` | | `MARIGOLD_RESIZE` | `Int` | `None` | Max dimension ceiling (default: keep original). |
| `--split_resolution` | | `MARIGOLD_SPLIT_RESOLUTION` | `Int` | `512` | Resolution for each perspective tile (512 or 1024). |
| `--batch_size` | | `MARIGOLD_BATCH_SIZE` | `Int` | `1` | Perspective view batch size (keep 1 for low VRAM). |
| `--debug` | | `MARIGOLD_DEBUG` | `Flag` | `False` | Saves 12 perspective tiles, distance maps, and `cameras.json`. |
| `--maps` | | `MARIGOLD_MAPS` | `Flag` | `False` | Saves `depth_vis.png`, `depth.exr`, and `mask.png`. |
| `--depth_npy / --no-depth_npy` | | `MARIGOLD_DEPTH_NPY` | `Bool` | `True` | Saves primary `depth.npy` float32 array. |
| `--points_ply` | | `MARIGOLD_POINTS_PLY` | `Flag` | `False` | Exports 3D point cloud in `.ply` format. |
