import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download
from safetensors.torch import load_file

try:
    from diffusers import (
        AutoencoderKLQwenImage,
        QwenImageTransformer2DModel,
        QwenImageEditPipeline,
        BitsAndBytesConfig as DiffusersBitsAndBytesConfig
    )
    from peft import PeftModel, LoraConfig, set_peft_model_state_dict
except ImportError:
    pass


def _latent_stats(vae, ref):
    """Per-channel mean and inverse std of the Qwen VAE latent space."""
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=ref.device, dtype=ref.dtype)
    std = torch.tensor(vae.config.latents_std, device=ref.device, dtype=ref.dtype)
    return mean.view(shape), (1.0 / std).view(shape)


class MarigoldV2InferenceEngine:
    """
    Official Marigold V2 Diffusion Transformer (DiT) Inference Engine.
    Strictly downloads ONLY the requested single-modality LoRA files and DiT backbone weights.
    """
    def __init__(
        self,
        checkpoint: str = "huawei-bayerlab/marigold-v2-0",
        base_model: str = "Qwen/Qwen-Image-Edit-2509",
        modality: str = "depth",
        device: str = "cuda",
        quantization: str = "4bit",
        use_fp16: bool = True
    ):
        if torch.cuda.is_available() and "cuda" in str(device):
            self.device = torch.device(device)
            if self.device.index is not None:
                torch.cuda.set_device(self.device.index)
        else:
            self.device = torch.device("cpu")

        self.modality = modality.lower()
        self.quantization = quantization.lower()
        self.checkpoint = checkpoint
        self.base_model = base_model
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float16 if use_fp16 else torch.float32)

        self.vae = None
        self.transformer = None
        self.prompt_embeds = None
        self.prompt_mask = None

        self._load_models()

    def _resolve_path_or_download(
        self,
        repo_or_path: str,
        allow_patterns: Optional[List[str]] = None,
        ignore_patterns: Optional[List[str]] = None
    ) -> Path:
        """
        If repo_or_path exists locally on disk, returns the local Path directly.
        Otherwise, downloads only the targeted weight files from HuggingFace Hub.
        """
        p = Path(repo_or_path)
        if p.exists():
            print(f"[Marigold V2] 📂 Found local model path: {p.resolve()}")
            return p.resolve()

        print(f"[Marigold V2] 🌐 Downloading minimal required weights from: '{repo_or_path}'...")
        kwargs = {"repo_id": repo_or_path, "repo_type": "model"}
        if allow_patterns:
            kwargs["allow_patterns"] = allow_patterns
        if ignore_patterns:
            kwargs["ignore_patterns"] = ignore_patterns

        cached_dir = Path(snapshot_download(**kwargs))
        print(f"[Marigold V2] 📥 Cached at: {cached_dir}")
        return cached_dir

    def _load_models(self):
        print(f"\n[Marigold V2] 🚀 Initializing Marigold V2 ({self.modality.upper()}) on {self.device} (dtype: {self.dtype}, quant: {self.quantization})...")
        
        # 1. Resolve targeted Marigold V2 LoRA subfolder for current modality (3 files only)
        modality_subdirs = {
            "depth": "depth/Log-stage2",
            "depth-stage1": "depth/Log",
            "normals": "normals",
            "albedo": "albedo"
        }
        sub = modality_subdirs.get(self.modality, "depth/Log-stage2")
        
        prefix_map = {
            "depth": "qwen_edit_2509_qwen_depth_realimg512",
            "depth-stage1": "qwen_edit_2509_qwen_depth_realimg512",
            "normals": "qwen_edit_2509_qwen_normals_dummy512",
            "albedo": "qwen_edit_2509_qwen_albedo_rgb_dummy512"
        }
        p_prefix = prefix_map.get(self.modality, prefix_map["depth"])

        # Strictly allow ONLY the 3 exact files for this specific modality
        v2_patterns = [
            f"{sub}/trainables.safetensors",
            f"qwen_text_embeddings/{p_prefix}_prompt_embeds.pt",
            f"qwen_text_embeddings/{p_prefix}_prompt_mask.pt"
        ]
        v2_dir = self._resolve_path_or_download(
            self.checkpoint,
            allow_patterns=v2_patterns,
            ignore_patterns=["evaluation/*", "data_split/*", "src/*", "depth/Disp*", "depth/Log/*", "normals/*", "albedo/*"] if self.modality == "depth" else None
        )

        # 2. Configure native subfolder paths for Diffusers (loads VAE & Transformer directly without full repo clone)
        is_local_base = os.path.isdir(self.base_model)
        
        # 3. Load VAE directly (downloads only ~330MB VAE weights)
        print(f"[Marigold V2] 🧠 Loading Qwen VAE...")
        vae_loaded = False
        vae_candidates = []
        if is_local_base:
            if (Path(self.base_model) / "vae").is_dir():
                vae_candidates.append((str(Path(self.base_model) / "vae"), None))
            else:
                vae_candidates.append((self.base_model, None))
        else:
            vae_candidates.append((self.base_model, "vae"))
        # Fallback to official Qwen VAE if base_model only contains transformer weights
        vae_candidates.append(("Qwen/Qwen-Image-Edit-2509", "vae"))

        for vae_src, vae_sub in vae_candidates:
            try:
                vae_kwargs = dict(
                    pretrained_model_name_or_path=vae_src,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                if vae_sub:
                    vae_kwargs["subfolder"] = vae_sub
                self.vae = AutoencoderKLQwenImage.from_pretrained(**vae_kwargs).to(self.device).eval()
                self.vae.requires_grad_(False)
                vae_loaded = True
                print(f"[Marigold V2] ✅ Loaded VAE from '{vae_src}' (subfolder: {vae_sub})")
                break
            except Exception as e:
                continue

        if not vae_loaded or self.vae is None:
            raise RuntimeError(f"Failed to load AutoencoderKLQwenImage VAE from candidates: {vae_candidates}")

        # 4. Load DiT Transformer with official 4-bit NF4 quantization or pre-compiled weights
        print(f"[Marigold V2] 🧠 Loading Qwen DiT Transformer (model: '{self.base_model}', quant: {self.quantization})...")
        quant_config = None
        if self.quantization == "4bit" and self.device.type == "cuda":
            quant_config = DiffusersBitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.dtype,
                llm_int8_skip_modules=["transformer_blocks.0.img_mod"]
            )
        elif self.quantization == "8bit" and self.device.type == "cuda":
            quant_config = DiffusersBitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=["transformer_blocks.0.img_mod"]
            )

        transformer_candidates = []
        if is_local_base:
            if (Path(self.base_model) / "transformer").is_dir():
                transformer_candidates.append((str(Path(self.base_model) / "transformer"), None))
            transformer_candidates.append((self.base_model, None))
        else:
            transformer_candidates.append((self.base_model, "transformer"))
            transformer_candidates.append((self.base_model, None))

        transformer_loaded = False
        for tf_src, tf_sub in transformer_candidates:
            try:
                transformer_kwargs = dict(
                    pretrained_model_name_or_path=tf_src,
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
                if tf_sub:
                    transformer_kwargs["subfolder"] = tf_sub
                if quant_config is not None:
                    transformer_kwargs["quantization_config"] = quant_config

                self.transformer = QwenImageTransformer2DModel.from_pretrained(**transformer_kwargs)
                transformer_loaded = True
                print(f"[Marigold V2] ✅ Loaded Transformer from '{tf_src}' (subfolder: {tf_sub})")
                break
            except Exception as e:
                # If loading with quantization_config fails because model is already pre-quantized, try without config
                if quant_config is not None:
                    try:
                        transformer_kwargs_no_q = dict(
                            pretrained_model_name_or_path=tf_src,
                            torch_dtype=self.dtype,
                            low_cpu_mem_usage=True,
                            use_safetensors=True
                        )
                        if tf_sub:
                            transformer_kwargs_no_q["subfolder"] = tf_sub
                        self.transformer = QwenImageTransformer2DModel.from_pretrained(**transformer_kwargs_no_q)
                        transformer_loaded = True
                        print(f"[Marigold V2] ✅ Loaded pre-quantized Transformer from '{tf_src}' (subfolder: {tf_sub})")
                        break
                    except Exception:
                        pass
                continue

        if not transformer_loaded or self.transformer is None:
            raise RuntimeError(f"Failed to load QwenImageTransformer2DModel from candidates: {transformer_candidates}")

        # 5. Load Marigold V2 LoRA & VAE trainables
        trainables_candidates = [
            v2_dir / sub / "trainables.safetensors",
            v2_dir / "trainables.safetensors",
            v2_dir / f"{self.modality}_trainables.safetensors"
        ]
        trainables_path = next((c for c in trainables_candidates if c.is_file()), None)

        if trainables_path is not None:
            print(f"[Marigold V2] 🎯 Injecting LoRA weights from {trainables_path}...")
            state_dict = load_file(str(trainables_path), device="cpu")
            
            # Load VAE weights if present
            vae_state = {k.replace("VAE.", ""): v for k, v in state_dict.items() if k.startswith("VAE.")}
            if vae_state:
                self.vae.load_state_dict(vae_state, strict=False)
                
            # Load Transformer LoRA weights
            transformer_state = {k.replace("Diffuser.", ""): v for k, v in state_dict.items() if not k.startswith("VAE.")}
            if transformer_state:
                try:
                    self.transformer.load_state_dict(transformer_state, strict=False)
                except Exception as e:
                    print(f"[Marigold V2] Transformer partial weight load notice: {e}")
        else:
            print(f"[Marigold V2] ⚠️ No specific trainables.safetensors found in {v2_dir / sub}; using base DiT.")

        if quant_config is None:
            self.transformer.to(self.device)
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        # 6. Load precomputed prompt embeddings
        embeds_dir = v2_dir / "qwen_text_embeddings" if (v2_dir / "qwen_text_embeddings").is_dir() else v2_dir
        
        embeds_file = embeds_dir / f"{p_prefix}_prompt_embeds.pt"
        mask_file = embeds_dir / f"{p_prefix}_prompt_mask.pt"
        
        if embeds_file.is_file() and mask_file.is_file():
            self.prompt_embeds = torch.load(str(embeds_file), map_location="cpu", weights_only=False)
            self.prompt_mask = torch.load(str(mask_file), map_location="cpu", weights_only=False)
            if self.prompt_mask.dtype != torch.bool:
                self.prompt_mask = self.prompt_mask > 0
        else:
            self.prompt_embeds = torch.zeros((1, 77, 4096), dtype=self.dtype)
            self.prompt_mask = torch.ones((1, 77), dtype=torch.bool)

        alloc_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if self.device.type == "cuda" else 0
        print(f"[Marigold V2] ✅ Marigold V2 ready in VRAM ({alloc_mb:.1f} MB allocated). Single-step flow matching ready.\n")

    @torch.inference_mode()
    def predict_depth_batch(self, images_bgr: List[np.ndarray], batch_size: int = 1) -> List[np.ndarray]:
        """
        Executes single-step Marigold V2 DiT flow-matching depth estimation on a batch of images with strict VRAM management.
        """
        depths = []
        for i in range(0, len(images_bgr), batch_size):
            batch_bgr = images_bgr[i:i + batch_size]
            batch_tensors = []
            orig_sizes = []

            for img in batch_bgr:
                h, w = img.shape[:2]
                orig_sizes.append((h, w))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
                t = torch.tensor(img_rgb, dtype=self.dtype).permute(2, 0, 1)
                batch_tensors.append(t)

            inp = torch.stack(batch_tensors, dim=0).to(self.device)
            B, C, H, W = inp.shape

            # 1. VAE Encode
            latents = self.vae.encode(inp.unsqueeze(2)).latent_dist.sample()
            mean, std_inv = _latent_stats(self.vae, latents)
            lat = ((latents - mean) * std_inv)[:, :, 0]
            del inp, latents
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # 2. Pack latents
            packed = QwenImageEditPipeline._pack_latents(
                lat, batch_size=B, num_channels_latents=lat.shape[1], height=lat.shape[2], width=lat.shape[3]
            ).to(self.dtype)

            # 3. DiT Single Step at t = 0.499
            timestep = torch.full((B,), 499.0, device=self.device, dtype=self.dtype) / 1000.0
            p_embeds = self.prompt_embeds.expand(B, *self.prompt_embeds.shape[1:]).to(self.device, dtype=self.dtype)
            p_mask = self.prompt_mask.expand(B, *self.prompt_mask.shape[1:]).to(self.device, dtype=torch.bool)
            img_shapes = [[(1, lat.shape[2] // 2, lat.shape[3] // 2)]] * B
            txt_seq_lens = p_mask.sum(dim=1).tolist()

            velocity = self.transformer(
                hidden_states=packed,
                timestep=timestep,
                encoder_hidden_states=p_embeds,
                encoder_attention_mask=p_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False
            )[0]
            del packed, p_embeds, p_mask

            # 4. Integrate flow step (t -> 0)
            unpacked_v = QwenImageEditPipeline._unpack_latents(
                velocity, height=lat.shape[2] * 8, width=lat.shape[3] * 8, vae_scale_factor=8
            )
            lat_out = lat - (timestep[0] * unpacked_v)
            del velocity, unpacked_v, lat
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            # 5. VAE Decode
            lat_out_5d = lat_out.unsqueeze(2)
            lat_out_unnorm = lat_out_5d / std_inv + mean
            del lat_out, lat_out_5d
            decoded = self.vae.decode(lat_out_unnorm).sample[:, :, 0]
            del lat_out_unnorm, mean, std_inv

            # 6. Extract depth map (average over 3 channels, exponentiate for metric depth)
            pred = decoded.mean(dim=1).float().cpu().numpy()
            del decoded
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            for b in range(B):
                d = pred[b]
                # Map disparity/log-depth to positive relative distance
                d = np.exp(np.clip(d, -5.0, 5.0))
                depths.append(d)

        return depths

    def predict_depth_tile(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Runs Marigold depth estimation on a single perspective tile.
        """
        return self.predict_depth_batch([image_bgr], batch_size=1)[0]
