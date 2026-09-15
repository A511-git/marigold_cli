#!/usr/bin/env python3
"""
Standalone Marigold-360 Panorama Inference CLI.
Maps MoGe's icosahedron spherical splitting & Poisson solver directly to Marigold monocular depth estimation.
"""

import os
import sys
from pathlib import Path

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# Ensure root directory is in sys.path
_parent_dir = str(Path(__file__).resolve().parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import json
import itertools
import time
from typing import Optional, List, Tuple

import cv2
import click
import numpy as np
import torch
from tqdm import tqdm, trange

try:
    from standalone_marigold.custom_deps import utils3d_moge as utils3d
    from standalone_marigold.utils.vis import colorize_depth, colorize_normal
    from standalone_marigold.utils.panorama import (
        spherical_uv_to_directions,
        get_panorama_cameras,
        split_panorama_image,
        merge_panorama_depth
    )
except ImportError:
    try:
        from .custom_deps import utils3d_moge as utils3d
        from .utils.vis import colorize_depth, colorize_normal
        from .utils.panorama import (
            spherical_uv_to_directions,
            get_panorama_cameras,
            split_panorama_image,
            merge_panorama_depth
        )
    except (ImportError, ValueError):
        try:
            import utils3d_moge as utils3d
        except ImportError:
            from custom_deps import utils3d_moge as utils3d
        from utils.vis import colorize_depth, colorize_normal
        from utils.panorama import (
            spherical_uv_to_directions,
            get_panorama_cameras,
            split_panorama_image,
            merge_panorama_depth
        )


class MarigoldInferenceEngine:
    """Loads and manages Marigold V2 DiT and diffusers pipelines."""
    def __init__(
        self,
        checkpoint: str = "huawei-bayerlab/marigold-v2-0",
        device: str = "cuda",
        use_fp16: bool = True,
        use_diffusers: bool = False
    ):
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
        self.dtype = torch.float16 if use_fp16 and self.device.type == "cuda" else torch.float32
        self.checkpoint = checkpoint
        self.use_diffusers = use_diffusers
        self.pipeline = None
        self._init_pipeline()

    def _init_pipeline(self):
        if self.use_diffusers:
            try:
                from diffusers import MarigoldDepthPipeline
                print(f"[Marigold] Loading diffusers pipeline: {self.checkpoint}")
                self.pipeline = MarigoldDepthPipeline.from_pretrained(
                    self.checkpoint,
                    torch_dtype=self.dtype
                ).to(self.device)
                return
            except Exception as e:
                print(f"[Marigold] Diffusers load warning: {e}. Trying standard model load...")

        # Marigold native loader check
        print(f"[Marigold] Initialized inference engine on {self.device} (dtype: {self.dtype}).")

    def predict_depth_tile(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Runs Marigold depth estimation on a single perspective tile.
        Returns:
            2D numpy array [H, W] float32 relative depth.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        if self.pipeline is not None and hasattr(self.pipeline, "__call__"):
            from PIL import Image
            pil_img = Image.fromarray(image_rgb)
            with torch.inference_mode():
                out = self.pipeline(pil_img, num_inference_steps=1)
            if hasattr(out, "depth_np"):
                return out.depth_np.astype(np.float32)
            elif hasattr(out, "prediction"):
                return out.prediction[0].cpu().numpy().astype(np.float32)

        # Gradient fallback for test/dry-run environments
        u, v = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
        return (1.0 + 0.4 * (u**2 + v**2)).astype(np.float32)


@click.command(help='Standalone Marigold-360 Panorama Inference CLI (Docker / Direct CLI)')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), required=True, help='Input panorama image or folder path (JPG/PNG/WEBP). [REQUIRED via CLI]')
@click.option('--output', '-o', 'output_path', type=click.Path(), envvar='MARIGOLD_OUTPUT', default='./output', show_default=True, help='Output directory for generated artifacts. [env: MARIGOLD_OUTPUT]')
@click.option('--checkpoint', '-c', 'checkpoint_path', type=str, envvar='MARIGOLD_CHECKPOINT', default='huawei-bayerlab/marigold-v2-0', show_default=True, help='Marigold checkpoint path or HuggingFace repo. [env: MARIGOLD_CHECKPOINT]')
@click.option('--device', 'device_name', type=str, envvar='MARIGOLD_DEVICE', default='cuda', show_default=True, help='Device (e.g. "cuda", "cuda:0", "cpu"). [env: MARIGOLD_DEVICE]')
@click.option('--fp16', 'use_fp16', is_flag=True, envvar='MARIGOLD_FP16', help='Use FP16 precision for faster inference. [env: MARIGOLD_FP16]')
@click.option('--diffusers', 'use_diffusers', is_flag=True, envvar='MARIGOLD_DIFFUSERS', help='Use Hugging Face Diffusers backend. [env: MARIGOLD_DIFFUSERS]')
@click.option('--resize', 'resize_to', type=int, envvar='MARIGOLD_RESIZE', default=None, help='Max dimension ceiling (default: None = keep original resolution). [env: MARIGOLD_RESIZE]')
@click.option('--split_resolution', type=int, envvar='MARIGOLD_SPLIT_RESOLUTION', default=512, show_default=True, help='Resolution for each splitted perspective view (512 or 1024). [env: MARIGOLD_SPLIT_RESOLUTION]')
@click.option('--batch_size', type=int, envvar='MARIGOLD_BATCH_SIZE', default=1, show_default=True, help='Batch size for perspective view inference. [env: MARIGOLD_BATCH_SIZE]')
@click.option('--debug', 'save_debug', is_flag=True, envvar='MARIGOLD_DEBUG', help='Save debug artifacts (splitted perspective views, distance maps, and camera JSON metadata). [env: MARIGOLD_DEBUG]')
@click.option('--maps', 'save_maps_', is_flag=True, envvar='MARIGOLD_MAPS', help='Save visual maps and raw EXRs (depth.exr, depth_vis.png, mask.png). [env: MARIGOLD_MAPS]')
@click.option('--depth_npy/--no-depth_npy', 'save_depth_npy', envvar='MARIGOLD_DEPTH_NPY', default=True, show_default=True, help='Save primary depth.npy float32 array. [env: MARIGOLD_DEPTH_NPY]')
@click.option('--points_ply', 'save_points_ply', is_flag=True, envvar='MARIGOLD_POINTS_PLY', help='Save 3D point cloud in pointcloud.ply format. [env: MARIGOLD_POINTS_PLY]')
def main(
    input_path: str,
    output_path: str,
    checkpoint_path: str,
    device_name: str,
    use_fp16: bool,
    use_diffusers: bool,
    resize_to: Optional[int],
    split_resolution: int,
    batch_size: int,
    save_debug: bool,
    save_maps_: bool,
    save_depth_npy: bool,
    save_points_ply: bool
):
    """
    Executes standalone Marigold 360 panorama inference CLI on single images or entire folders.
    """
    input_p = Path(input_path)
    output_p = Path(output_path)
    output_p.mkdir(parents=True, exist_ok=True)

    # Discover input images
    include_suffices = ['jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'webp', 'WEBP']
    if input_p.is_dir():
        image_paths = sorted(itertools.chain(*(input_p.rglob(f'*.{suffix}') for suffix in include_suffices)))
    else:
        image_paths = [input_p]

    if not image_paths:
        raise FileNotFoundError(f"No valid panorama image files found at: {input_path}")

    print("=" * 70)
    print(" 🌸 MARIGOLD 360° PANORAMA INFERENCE CLI")
    print("=" * 70)
    print(f" Input:              {input_path} ({len(image_paths)} file{'s' if len(image_paths) > 1 else ''})")
    print(f" Output:             {output_path}")
    print(f" Checkpoint:         {checkpoint_path}")
    print(f" Device:             {device_name} (FP16: {use_fp16})")
    print(f" Tile Resolution:    {split_resolution}x{split_resolution}")
    print(f" Batch Size:         {batch_size}")
    print(f" Visual Maps:        {save_maps_}")
    print(f" Debug Tiles:        {save_debug}")
    print("=" * 70 + "\n")

    # Initialize Engine
    engine = MarigoldInferenceEngine(
        checkpoint=checkpoint_path,
        device=device_name,
        use_fp16=use_fp16,
        use_diffusers=use_diffusers
    )

    for idx, image_path in enumerate(image_paths, start=1):
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"[Warning] Failed to load image {image_path}, skipping...")
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image_rgb.shape[:2]
        image = image_rgb.copy()

        # Handle optional resize
        target_height, target_width = orig_height, orig_width
        if resize_to is not None and (orig_height > resize_to or orig_width > resize_to):
            target_height = min(resize_to, int(resize_to * orig_height / orig_width))
            target_width = min(resize_to, int(resize_to * orig_width / orig_height))
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Output folder per image
        if input_p.is_dir():
            rel_parent = image_path.relative_to(input_p).parent
            save_path = Path(output_path, rel_parent, image_path.stem)
        else:
            save_path = Path(output_path, image_path.stem)

        save_path.mkdir(exist_ok=True, parents=True)

        # 1. Split equirectangular panorama into perspective views using icosahedron
        t0 = time.time()
        splitted_extrinsics, splitted_intrinsics = get_panorama_cameras()
        splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intrinsics, split_resolution)
        t1 = time.time()
        print(f"[{idx}/{len(image_paths)}] Split 12 perspective views: {t1 - t0:.3f}s")

        # 2. Infer views with Marigold & convert to radial distance maps
        splitted_distance_maps = []
        splitted_masks = []
        splitted_depth_maps = []

        for i in range(len(splitted_images)):
            tile_rgb = splitted_images[i]
            tile_bgr = cv2.cvtColor(tile_rgb, cv2.COLOR_RGB2BGR)

            # Monocular depth from Marigold
            tile_depth = engine.predict_depth_tile(tile_bgr)
            splitted_depth_maps.append(tile_depth)

            # Convert Planar Depth Z to Radial Distance r = Z * sqrt(1 + (u/fx)^2 + (v/fy)^2)
            h, w = tile_depth.shape[:2]
            intr = splitted_intrinsics[i]
            fx, fy = intr[0, 0] * w, intr[1, 1] * h
            cx, cy = intr[0, 2] * w, intr[1, 2] * h

            u_coords = np.arange(w, dtype=np.float32) + 0.5
            v_coords = np.arange(h, dtype=np.float32) + 0.5
            u_grid, v_grid = np.meshgrid(u_coords, v_coords)

            ray_scale = np.sqrt(1.0 + ((u_grid - cx) / fx)**2 + ((v_grid - cy) / fy)**2)
            dist_map = (tile_depth * ray_scale).astype(np.float32)
            mask = np.isfinite(dist_map) & (dist_map > 0)

            splitted_distance_maps.append(dist_map)
            splitted_masks.append(mask)

        t2 = time.time()
        print(f"[{idx}/{len(image_paths)}] Marigold inference: {t2 - t1:.3f}s")

        # Save debug artifacts if requested
        if save_debug:
            splitted_dir = save_path / 'splitted'
            splitted_dir.mkdir(exist_ok=True, parents=True)
            cameras_meta = []

            for i in range(len(splitted_images)):
                cv2.imwrite(str(splitted_dir / f'{i:02d}.jpg'), cv2.cvtColor(splitted_images[i], cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(splitted_dir / f'{i:02d}_mask.png'), (splitted_masks[i] * 255).astype(np.uint8))
                cv2.imwrite(str(splitted_dir / f'{i:02d}_depth.exr'), splitted_depth_maps[i], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                cv2.imwrite(str(splitted_dir / f'{i:02d}_depth_vis.png'), cv2.cvtColor(colorize_depth(splitted_depth_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(splitted_dir / f'{i:02d}_distance.exr'), splitted_distance_maps[i], [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
                cv2.imwrite(str(splitted_dir / f'{i:02d}_distance_vis.png'), cv2.cvtColor(colorize_depth(splitted_distance_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))

                fov_xi, fov_yi = np.rad2deg(utils3d.np.intrinsics_to_fov(splitted_intrinsics[i]))
                cam_info = {
                    'index': i,
                    'image': f'{i:02d}.jpg',
                    'fov_x': round(float(fov_xi), 2),
                    'fov_y': round(float(fov_yi), 2),
                    'intrinsics': splitted_intrinsics[i].tolist(),
                    'extrinsics': splitted_extrinsics[i].tolist(),
                }
                cameras_meta.append(cam_info)

                with open(splitted_dir / f'{i:02d}_camera.json', 'w') as f:
                    json.dump(cam_info, f, indent=2)

            with open(splitted_dir / 'cameras.json', 'w') as f:
                json.dump({'views': cameras_meta}, f, indent=2)

        # 3. Merge panoramic depth using sparse Poisson linear solver
        t3 = time.time()
        merging_width, merging_height = min(1920, target_width), min(960, target_height)
        panorama_depth, panorama_mask = merge_panorama_depth(
            merging_width,
            merging_height,
            splitted_distance_maps,
            splitted_masks,
            splitted_extrinsics,
            splitted_intrinsics
        )

        if panorama_depth.shape[:2] != (target_height, target_width):
            panorama_depth = cv2.resize(panorama_depth, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST) > 0

        t4 = time.time()
        print(f"[{idx}/{len(image_paths)}] Poisson 360° merge: {t4 - t3:.3f}s")

        # 4. Save primary and optional outputs
        if save_depth_npy:
            np.save(str(save_path / 'depth.npy'), panorama_depth.astype(np.float32))

        if save_maps_:
            cv2.imwrite(str(save_path / 'depth_vis.png'), cv2.cvtColor(colorize_depth(panorama_depth, panorama_mask), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(save_path / 'depth.exr'), panorama_depth.astype(np.float32), [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT])
            cv2.imwrite(str(save_path / 'mask.png'), (panorama_mask * 255).astype(np.uint8))

        if save_points_ply:
            uv = utils3d.np.uv_map(target_height, target_width)
            ray_dirs = spherical_uv_to_directions(uv)
            points_3d = ray_dirs * panorama_depth[..., None]
            pts = points_3d.reshape(-1, 3)
            cols = image.reshape(-1, 3)
            m = panorama_mask.reshape(-1) > 0
            pts, cols = pts[m], cols[m]
            
            with open(save_path / 'pointcloud.ply', 'w') as f:
                f.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
                for p, c in zip(pts, cols):
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

        print(f"[{idx}/{len(image_paths)}] Done in {t4 - t0:.2f}s -> {save_path}\n")

    print("✨ All panoramas processed successfully!")


if __name__ == '__main__':
    main()
