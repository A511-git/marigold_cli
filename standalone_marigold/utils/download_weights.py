import os
import argparse
from pathlib import Path
from typing import List, Tuple
from huggingface_hub import hf_hub_download
import shutil

# MoGe-3 Official Models
MODELS = {
    "vitl": ("Ruicheng/moge-3-vitl", "moge-3-vitl.pt", "370M"),
    "vitg": ("Ruicheng/moge-3-vitg", "moge-3-vitg.pt", "1.25B"),
}

ALIASES = {
    "moge-3-vitl": "vitl",
    "moge-3-vitg": "vitg",
    "v3": "vitl",
}


def normalize_model_name(name: str) -> str:
    key = name.lower().strip()
    return ALIASES.get(key, key)


def download_single(model_key: str, output_dir: Path) -> str:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Supported models: {list(MODELS.keys())}")

    repo_id, target_filename, params = MODELS[model_key]
    target_path = output_dir / target_filename

    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"[OK] MoGe-3 ({model_key.upper()} - {params}) checkpoint already exists at: {target_path}")
        return str(target_path)

    print(f"[MoGe CLI] Checkpoint not found at {target_path}. Auto-downloading MoGe-3 ({model_key.upper()} - {params}) from {repo_id} ('model.pt') via Hugging Face...")
    cached_path = hf_hub_download(repo_id=repo_id, filename="model.pt")

    print(f"[MoGe CLI] Saving checkpoint to: {target_path}")
    shutil.copyfile(cached_path, target_path)
    size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"[SUCCESS] Download completed: {target_path} ({size_mb:.2f} MB)")
    return str(target_path)


def download(model: str = "vitl", output_dir: str = "checkpoints") -> List[str]:
    """Download pretrained MoGe-3 checkpoint(s) from Hugging Face and save locally."""
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    normalized = normalize_model_name(model)
    if normalized == "all":
        results = []
        for key in MODELS:
            results.append(download_single(key, target_dir))
        return results
    else:
        return [download_single(normalized, target_dir)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MoGe-3 pretrained model weights (ViT-L: 370M, ViT-G: 1.25B)")
    parser.add_argument(
        "--model", "-m",
        choices=["vitl", "vitg", "moge-3-vitl", "moge-3-vitg", "all"],
        default="vitl",
        help="MoGe-3 model variant to download: 'vitl' (370M) or 'vitg' (1.25B) or 'all' (default: vitl)"
    )
    parser.add_argument(
        "--version",
        default=None,
        help="Deprecated alias for --model (e.g. 'v3')"
    )
    # Pick /checkpoints if in container root, else ./checkpoints
    default_dir = "/checkpoints" if Path("/checkpoints").is_dir() else "checkpoints"
    parser.add_argument(
        "--output_dir", "-o",
        default=default_dir,
        help=f"Output directory for checkpoints (default: {default_dir})"
    )
    args = parser.parse_args()

    selected_model = args.version if args.version is not None else args.model
    download(selected_model, args.output_dir)
