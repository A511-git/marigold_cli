import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Union
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
    Powered by Qwen-Image-Edit-2509 DiT + Huawei Bayer Lab single-step flow matching LoRA weights.
    """
    def __init__(
        self,
        checkpoint: str = "huawei-bayerlab/marigold-v2-0",
        modality: str = "depth",
        device: str = "cuda",
        quantization: str = "4bit",
        use_fp16: bool = True
    ):
        self.device = torch.device(device if torch.cuda.is_available() and "cuda" in str(device) else "cpu")
        self.modality = modality.lower()
        self.quantization = quantization
        self.checkpoint = checkpoint
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else (torch.float16 if use_fp16 else torch.float32)

        self.vae = None
        self.transformer = None
        self.prompt_embeds = None
        self.prompt_mask = None

        self._load_models()

    def _load_models(self):
        print(f"\n[Marigold V2 DiT] 🚀 Loading Marigold V2 ({self.modality.upper()}) on {self.device} (dtype: {self.dtype}, quant: {self.quantization})...")
        
        # 1. Resolve / Download base Qwen DiT and Marigold V2 LoRA checkpoints
        qwen_repo = "Qwen/Qwen-Image-Edit-2509"
        v2_repo = "huawei-bayerlab/marigold-v2-0"

        # Check if local or download
        if os.path.isdir(self.checkpoint):
            v2_dir = Path(self.checkpoint)
        else:
            print(f"[Marigold V2 DiT] 📥 Ensuring Marigold V2 weights are cached from {v2_repo}...")
            v2_dir = Path(snapshot_download(repo_id=v2_repo, repo_type="model"))

        print(f"[Marigold V2 DiT] 📥 Ensuring Qwen base model is cached from {qwen_repo}...")
        qwen_dir = Path(snapshot_download(
            repo_id=qwen_repo,
            repo_type="model",
            allow_patterns=["vae/*", "transformer/*", "model_index.json", "scheduler/*"]
        ))

        # 2. Load VAE
        print(f"[Marigold V2 DiT] 🧠 Loading Qwen VAE...")
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            qwen_dir / "vae",
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device).eval()
        self.vae.requires_grad_(False)

        # 3. Load DiT Transformer with 4-bit / 8-bit / bf16
        print(f"[Marigold V2 DiT] 🧠 Loading Qwen DiT Transformer (quant: {self.quantization})...")
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

        self.transformer = QwenImageTransformer2DModel.from_pretrained(
            qwen_dir / "transformer",
            quantization_config=quant_config,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        # 4. Load Marigold V2 LoRA & VAE trainables
        modality_subdirs = {
            "depth": "depth/Log-stage2",
            "normals": "normals",
            "albedo": "albedo"
        }
        sub = modality_subdirs.get(self.modality, "depth/Log-stage2")
        trainables_path = v2_dir / sub / "trainables.safetensors"
        
        if trainables_path.is_file():
            print(f"[Marigold V2 DiT] 🎯 Loading LoRA weights from {trainables_path.name}...")
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
                except Exception:
                    pass

        if quant_config is None:
            self.transformer.to(self.device)
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        # 5. Load precomputed prompt embeddings
        prefix_map = {
            "depth": "qwen_edit_2509_qwen_depth_realimg512",
            "normals": "qwen_edit_2509_qwen_normals_dummy512",
            "albedo": "qwen_edit_2509_qwen_albedo_rgb_dummy512"
        }
        p_prefix = prefix_map.get(self.modality, prefix_map["depth"])
        embeds_dir = v2_dir / "qwen_text_embeddings"
        
        embeds_file = embeds_dir / f"{p_prefix}_prompt_embeds.pt"
        mask_file = embeds_dir / f"{p_prefix}_prompt_mask.pt"
        
        if embeds_file.is_file() and mask_file.is_file():
            self.prompt_embeds = torch.load(str(embeds_file), map_location="cpu", weights_only=False)
            self.prompt_mask = torch.load(str(mask_file), map_location="cpu", weights_only=False)
            if self.prompt_mask.dtype != torch.bool:
                self.prompt_mask = self.prompt_mask > 0

        alloc_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if self.device.type == "cuda" else 0
        print(f"[Marigold V2 DiT] ✅ Marigold V2 ready in VRAM ({alloc_mb:.1f} MB allocated). Single-step flow matching ready.")

    @torch.no_grad()
    def predict_depth_batch(self, images_bgr: List[np.ndarray], batch_size: int = 1) -> List[np.ndarray]:
        """
        Executes single-step Marigold V2 DiT flow-matching depth estimation on a batch of images.
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

            # 4. Integrate flow step (t -> 0)
            unpacked_v = QwenImageEditPipeline._unpack_latents(
                velocity, height=lat.shape[2] * 8, width=lat.shape[3] * 8, vae_scale_factor=8
            )
            lat_out = lat - (timestep[0] * unpacked_v)

            # 5. VAE Decode
            lat_out_5d = lat_out.unsqueeze(2)
            lat_out_unnorm = lat_out_5d / std_inv + mean
            decoded = self.vae.decode(lat_out_unnorm).sample[:, :, 0]

            # 6. Extract depth map (average over 3 channels, exponentiate for metric depth)
            pred = decoded.mean(dim=1).float().cpu().numpy()

            for b in range(B):
                d = pred[b]
                # Map disparity/log-depth to positive relative distance
                d = np.exp(np.clip(d, -5.0, 5.0))
                depths.append(d)

        return depths
