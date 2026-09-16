#!/usr/bin/env python3
"""
🌸 Standalone Marigold V2 360° Panorama Inference CLI.
Maps MoGe's icosahedron spherical camera splitting & PyTorch GPU Poisson solver directly to Marigold V2 DiT monocular depth estimation.
Includes 12-tile global scale & shift alignment, camera height metric calibration (default 1.5m), and fast binary PLY export.
"""

import os
if os.environ.get('MPLBACKEND', '').startswith('module://'):
    os.environ['MPLBACKEND'] = 'Agg'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
from pathlib import Path

# Ensure root directory is in sys.path
_parent_dir = str(Path(__file__).resolve().parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import json
import itertools
import time
from typing import Optional, List, Tuple, Dict, Any, Union

import cv2
import click
import numpy as np
import torch
from tqdm import tqdm, trange

def save_exr_safely(filepath, img_array: np.ndarray):
    """Safely saves float32 EXR image using OpenCV or imageio fallback without throwing uncaught exceptions."""
    arr = np.ascontiguousarray(img_array.astype(np.float32))
    try:
        if cv2.imwrite(str(filepath), arr, [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT]):
            return
    except Exception:
        pass
    try:
        import imageio.v3 as iio
        iio.imwrite(str(filepath), arr)
    except Exception:
        pass

try:
    from standalone_marigold.custom_deps import utils3d_moge as utils3d
    from standalone_marigold.utils.vis import colorize_depth, colorize_normal
    from standalone_marigold.utils.panorama import (
        spherical_uv_to_directions,
        get_panorama_cameras,
        split_panorama_image,
        merge_panorama_depth,
        calibrate_camera_height_metric_scale,
        apply_metric_range_scaling,
        export_binary_ply
    )
    from standalone_marigold.panorama.alignment import align_tile_depths, solve_global_scale_shift
    from standalone_marigold.v2_engine import MarigoldV2InferenceEngine
except ImportError:
    try:
        from .custom_deps import utils3d_moge as utils3d
        from .utils.vis import colorize_depth, colorize_normal
        from .utils.panorama import (
            spherical_uv_to_directions,
            get_panorama_cameras,
            split_panorama_image,
            merge_panorama_depth,
            calibrate_camera_height_metric_scale,
            apply_metric_range_scaling,
            export_binary_ply
        )
        from .panorama.alignment import align_tile_depths, solve_global_scale_shift
        from .v2_engine import MarigoldV2InferenceEngine
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
            merge_panorama_depth,
            calibrate_camera_height_metric_scale,
            apply_metric_range_scaling,
            export_binary_ply
        )
        try:
            from panorama.alignment import align_tile_depths, solve_global_scale_shift
        except ImportError:
            from .panorama.alignment import align_tile_depths, solve_global_scale_shift
        from v2_engine import MarigoldV2InferenceEngine


def create_inference_engine(
    checkpoint: str = "huawei-bayerlab/marigold-v2-0",
    base_model: str = "Qwen/Qwen-Image-Edit-2509",
    modality: str = "depth",
    device: str = "cuda",
    quantization: str = "4bit",
    use_fp16: bool = True
) -> MarigoldV2InferenceEngine:
    """
    Creates and initializes the Marigold V2 DiT inference engine.
    Checks local existence first, or downloads directly from Hugging Face Hub.
    """
    return MarigoldV2InferenceEngine(
        checkpoint=checkpoint,
        base_model=base_model,
        modality=modality,
        device=device,
        quantization=quantization,
        use_fp16=use_fp16
    )


def process_single_panorama(
    engine: MarigoldV2InferenceEngine,
    image_path: Union[str, Path],
    save_path: Union[str, Path],
    split_resolution: int = 512,
    batch_size: int = 1,
    resize_to: Optional[int] = None,
    align_tiles: bool = True,
    camera_height: float = 1.5,
    min_depth: float = 0.3,
    max_depth: float = 15.0,
    save_maps_: bool = True,
    save_depth_npy: bool = True,
    save_points_ply: bool = True,
    save_debug: bool = False
) -> Dict[str, Any]:
    """
    Processes a single 360 panorama image using an in-memory Marigold V2 DiT engine.
    Includes pairwise 12-tile alignment, GPU Poisson merging, and camera height metric calibration.
    """
    image_path = Path(image_path)
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True, parents=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to load image from: {image_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_height, orig_width = image_rgb.shape[:2]
    image = image_rgb.copy()

    # Handle optional resize
    target_height, target_width = orig_height, orig_width
    if resize_to is not None and (orig_height > resize_to or orig_width > resize_to):
        target_height = min(resize_to, int(resize_to * orig_height / orig_width))
        target_width = min(resize_to, int(resize_to * orig_width / orig_height))
        image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # 1. Split equirectangular panorama into 12 perspective views using icosahedron geometry
    t0 = time.time()
    splitted_extrinsics, splitted_intrinsics = get_panorama_cameras()
    splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intrinsics, split_resolution)
    t1 = time.time()

    # 2. Batched GPU inference on perspective views via Marigold V2 DiT
    splitted_images_bgr = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in splitted_images]
    splitted_depth_maps = engine.predict_depth_batch(splitted_images_bgr, batch_size=batch_size)

    # 3. Global 12-camera scale and shift alignment across overlap regions
    raw_masks = [np.isfinite(d) for d in splitted_depth_maps]
    if align_tiles and len(splitted_depth_maps) > 1:
        aligned_tiles, scales, shifts = align_tile_depths(
            splitted_depth_maps, raw_masks, splitted_extrinsics, splitted_intrinsics
        )
    else:
        aligned_tiles = splitted_depth_maps

    splitted_distance_maps = []
    splitted_masks = []

    for i in range(len(aligned_tiles)):
        tile_d = aligned_tiles[i]
        h, w = tile_d.shape[:2]
        intr = splitted_intrinsics[i]
        fx = intr[0, 0] * w if intr[0, 0] <= 1.0 else intr[0, 0]
        fy = intr[1, 1] * h if intr[1, 1] <= 1.0 else intr[1, 1]
        cx = intr[0, 2] * w if intr[0, 2] <= 1.0 else intr[0, 2]
        cy = intr[1, 2] * h if intr[1, 2] <= 1.0 else intr[1, 2]

        u_coords = np.arange(w, dtype=np.float32) + 0.5
        v_coords = np.arange(h, dtype=np.float32) + 0.5
        u_grid, v_grid = np.meshgrid(u_coords, v_coords)

        ray_scale = np.sqrt(1.0 + ((u_grid - cx) / fx)**2 + ((v_grid - cy) / fy)**2)
        # Map aligned relative log-depth to positive radial distance
        dist_map = (np.exp(np.clip(tile_d, -6.0, 6.0)) * ray_scale).astype(np.float32)
        mask = np.isfinite(dist_map) & (dist_map > 1e-4)

        splitted_distance_maps.append(dist_map)
        splitted_masks.append(mask)

    t2 = time.time()

    # Save debug artifacts if requested
    if save_debug:
        splitted_dir = save_path / 'splitted'
        splitted_dir.mkdir(exist_ok=True, parents=True)
        cameras_meta = []

        for i in range(len(splitted_images)):
            cv2.imwrite(str(splitted_dir / f'{i:02d}.jpg'), cv2.cvtColor(splitted_images[i], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(splitted_dir / f'{i:02d}_mask.png'), (splitted_masks[i] * 255).astype(np.uint8))
            save_exr_safely(splitted_dir / f'{i:02d}_depth.exr', splitted_depth_maps[i])
            cv2.imwrite(str(splitted_dir / f'{i:02d}_depth_vis.png'), cv2.cvtColor(colorize_depth(splitted_depth_maps[i], splitted_masks[i]), cv2.COLOR_RGB2BGR))
            save_exr_safely(splitted_dir / f'{i:02d}_distance.exr', splitted_distance_maps[i])
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

    # 4. Merge panoramic depth using GPU Poisson solver (~4.6s on CUDA)
    t3 = time.time()
    merging_width, merging_height = min(1920, target_width), min(960, target_height)
    panorama_depth, panorama_mask = merge_panorama_depth(
        merging_width,
        merging_height,
        splitted_distance_maps,
        splitted_masks,
        splitted_extrinsics,
        splitted_intrinsics,
        device=engine.device
    )

    if panorama_depth.shape[:2] != (target_height, target_width):
        panorama_depth = cv2.resize(panorama_depth, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST) > 0

    # 5. Apply Camera Height Metric Calibration (or range scaling)
    if camera_height > 0.0:
        panorama_depth, metric_scale = calibrate_camera_height_metric_scale(
            panorama_depth,
            target_camera_height_m=camera_height,
            min_clamp_m=min_depth,
            max_clamp_m=max_depth
        )
    else:
        panorama_depth = apply_metric_range_scaling(
            panorama_depth,
            min_depth_m=min_depth,
            max_depth_m=max_depth
        )

    t4 = time.time()

    # 6. Save primary and optional outputs
    if save_depth_npy:
        np.save(str(save_path / 'depth.npy'), panorama_depth.astype(np.float32))

    if save_maps_:
        cv2.imwrite(str(save_path / 'depth_vis.png'), cv2.cvtColor(colorize_depth(panorama_depth, panorama_mask), cv2.COLOR_RGB2BGR))
        save_exr_safely(save_path / 'depth.exr', panorama_depth)
        cv2.imwrite(str(save_path / 'mask.png'), (panorama_mask * 255).astype(np.uint8))

    if save_points_ply:
        uv = utils3d.np.uv_map(target_height, target_width)
        ray_dirs = spherical_uv_to_directions(uv)
        points_3d = ray_dirs * panorama_depth[..., None]
        pts = points_3d.reshape(-1, 3)
        cols = image.reshape(-1, 3)
        m = panorama_mask.reshape(-1) > 0
        pts, cols = pts[m], cols[m]
        # Fast binary little-endian PLY export (< 50ms)
        export_binary_ply(save_path / 'pointcloud.ply', pts, cols)

    return {
        'depth': panorama_depth,
        'mask': panorama_mask,
        'timings': {
            'split': t1 - t0,
            'inference': t2 - t1,
            'merge': t4 - t3,
            'total': t4 - t0
        }
    }


@click.command(help='🌸 Marigold V2 360° Panorama Inference CLI')
@click.option('--input', '-i', 'input_path', type=click.Path(exists=True), required=True, help='Input panorama image or directory (JPG/PNG/WEBP). [REQUIRED]')
@click.option('--output', '-o', 'output_path', type=click.Path(), envvar='MARIGOLD_OUTPUT', default='./output', show_default=True, help='Output directory for generated depth & artifacts. [env: MARIGOLD_OUTPUT]')
@click.option('--checkpoint', '-c', 'checkpoint_path', type=str, envvar='MARIGOLD_CHECKPOINT', default='huawei-bayerlab/marigold-v2-0', show_default=True, help='Marigold V2 checkpoint path or HuggingFace repo. [env: MARIGOLD_CHECKPOINT]')
@click.option('--base_model', 'base_model', type=str, envvar='MARIGOLD_BASE_MODEL', default='Qwen/Qwen-Image-Edit-2509', show_default=True, help='Base Qwen DiT model repo or local directory. [env: MARIGOLD_BASE_MODEL]')
@click.option('--modality', '-m', 'modality', type=click.Choice(['depth', 'normals', 'albedo']), default='depth', show_default=True, help='Estimation modality. [env: MARIGOLD_MODALITY]')
@click.option('--quantization', '-q', 'quantization', type=click.Choice(['4bit', '8bit', 'none']), default='4bit', show_default=True, help='DiT quantization level. [env: MARIGOLD_QUANTIZATION]')
@click.option('--device', 'device_name', type=str, envvar='MARIGOLD_DEVICE', default='cuda', show_default=True, help='Compute device ("cuda", "cuda:0", "cpu"). [env: MARIGOLD_DEVICE]')
@click.option('--fp16', 'use_fp16', is_flag=True, envvar='MARIGOLD_FP16', default=True, help='Use FP16/BF16 precision. [env: MARIGOLD_FP16]')
@click.option('--resize', 'resize_to', type=int, envvar='MARIGOLD_RESIZE', default=None, help='Max dimension ceiling (default: None = full resolution). [env: MARIGOLD_RESIZE]')
@click.option('--split_resolution', type=int, envvar='MARIGOLD_SPLIT_RESOLUTION', default=512, show_default=True, help='Resolution for each perspective tile (512 or 1024). [env: MARIGOLD_SPLIT_RESOLUTION]')
@click.option('--batch_size', type=int, envvar='MARIGOLD_BATCH_SIZE', default=1, show_default=True, help='Batch size for perspective view inference. [env: MARIGOLD_BATCH_SIZE]')
@click.option('--camera_height', type=float, envvar='MARIGOLD_CAMERA_HEIGHT', default=1.5, show_default=True, help='Camera mounting height above floor in meters (default 1.5m). Set 0 to disable. [env: MARIGOLD_CAMERA_HEIGHT]')
@click.option('--min_depth', type=float, envvar='MARIGOLD_MIN_DEPTH', default=0.3, show_default=True, help='Minimum physical depth clamp in meters. [env: MARIGOLD_MIN_DEPTH]')
@click.option('--max_depth', type=float, envvar='MARIGOLD_MAX_DEPTH', default=15.0, show_default=True, help='Maximum physical depth clamp in meters. [env: MARIGOLD_MAX_DEPTH]')
@click.option('--align/--no-align', 'align_tiles', envvar='MARIGOLD_ALIGN', default=True, show_default=True, help='Perform global scale & shift alignment across overlapping tiles. [env: MARIGOLD_ALIGN]')
@click.option('--debug', 'save_debug', is_flag=True, envvar='MARIGOLD_DEBUG', help='Save debug artifacts (tiles, distance maps, camera JSONs). [env: MARIGOLD_DEBUG]')
@click.option('--maps', 'save_maps_', is_flag=True, envvar='MARIGOLD_MAPS', help='Save visual maps (depth.exr, depth_vis.png, mask.png). [env: MARIGOLD_MAPS]')
@click.option('--depth_npy/--no-depth_npy', 'save_depth_npy', envvar='MARIGOLD_DEPTH_NPY', default=True, show_default=True, help='Save primary depth.npy float32 array in meters. [env: MARIGOLD_DEPTH_NPY]')
@click.option('--points_ply', 'save_points_ply', is_flag=True, envvar='MARIGOLD_POINTS_PLY', help='Save 3D point cloud in binary pointcloud.ply format. [env: MARIGOLD_POINTS_PLY]')
def main(
    input_path: str,
    output_path: str,
    checkpoint_path: str,
    base_model: str,
    modality: str,
    quantization: str,
    device_name: str,
    use_fp16: bool,
    resize_to: Optional[int],
    split_resolution: int,
    batch_size: int,
    camera_height: float,
    min_depth: float,
    max_depth: float,
    align_tiles: bool,
    save_debug: bool,
    save_maps_: bool,
    save_depth_npy: bool,
    save_points_ply: bool
):
    """
    Executes Marigold V2 360° panorama inference on single images or entire folders.
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
    print(" 🌸 MARIGOLD V2 360° PANORAMA INFERENCE CLI")
    print("=" * 70)
    print(f" Input:              {input_path} ({len(image_paths)} file{'s' if len(image_paths) > 1 else ''})")
    print(f" Output:             {output_path}")
    print(f" Checkpoint:         {checkpoint_path}")
    print(f" Base Model:         {base_model}")
    print(f" Modality:           {modality}")
    print(f" Quantization:       {quantization}")
    print(f" Device:             {device_name} (FP16: {use_fp16})")
    print(f" Tile Resolution:    {split_resolution}x{split_resolution}")
    print(f" Batch Size:         {batch_size}")
    print(f" Tile Alignment:     {align_tiles}")
    print(f" Camera Height:      {camera_height}m (Range: {min_depth}m - {max_depth}m)")
    print(f" Visual Maps:        {save_maps_}")
    print(f" Debug Tiles:        {save_debug}")
    print("=" * 70 + "\n")

    # Initialize Marigold V2 DiT Engine
    engine = create_inference_engine(
        checkpoint=checkpoint_path,
        base_model=base_model,
        modality=modality,
        device=device_name,
        quantization=quantization,
        use_fp16=use_fp16
    )

    for idx, image_path in enumerate(image_paths, start=1):
        if input_p.is_dir():
            rel_parent = image_path.relative_to(input_p).parent
            save_path = Path(output_path, rel_parent, image_path.stem)
        else:
            save_path = Path(output_path, image_path.stem)

        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}...")
        try:
            res = process_single_panorama(
                engine=engine,
                image_path=image_path,
                save_path=save_path,
                split_resolution=split_resolution,
                batch_size=batch_size,
                resize_to=resize_to,
                align_tiles=align_tiles,
                camera_height=camera_height,
                min_depth=min_depth,
                max_depth=max_depth,
                save_maps_=save_maps_,
                save_depth_npy=save_depth_npy,
                save_points_ply=save_points_ply,
                save_debug=save_debug
            )
            print(f"[{idx}/{len(image_paths)}] ✅ Done in {res['timings']['total']:.2f}s (Split: {res['timings']['split']:.2f}s | Infer: {res['timings']['inference']:.2f}s | Merge: {res['timings']['merge']:.3f}s) -> {save_path}\n")
        except Exception as e:
            print(f"[{idx}/{len(image_paths)}] ❌ Failed {image_path.name}: {e}\n")

    print("✨ All panoramas processed successfully with Marigold V2!")


if __name__ == '__main__':
    main()
