"""White-box (per-sample gradient-norm) membership inference attack on NetDiffusion.


Threat model: the attacker has white-box access to the fine-tuned NetDiffusion
pipeline's weights and gradients (e.g., a released LoRA checkpoint). For each
candidate traffic sample the attacker:
  1. Loads the base Stable Diffusion model with the target LoRA adapter
     (resumed from an existing `checkpoint-*` directory produced during
     NetDiffusion fine-tuning; see ../../synthetic-data-generation/NetDiffusion).
  2. Runs a single forward + backward pass of the diffusion training
     objective (noise-prediction MSE at a random timestep) on that sample --
     no optimizer step is taken, so the model's weights are never actually
     updated.
  3. Records the training loss and the L2 norm of the loss gradient with
     respect to the trainable (LoRA) parameters.

Samples the model was fine-tuned on ("train") tend to sit closer to a local
loss minimum and so produce a smaller gradient norm than held-out ("test") or
out-of-distribution samples, which is the membership signal this script
measures.
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionLoraLoaderMixin, UNet2DConditionModel
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft


def import_text_encoder_class(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder", revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    raise ValueError(f"{model_class} is not supported by this streamlined script.")


class CandidateImageDataset(Dataset):
    """One candidate traffic-image sample per item, for gradient-norm scoring."""

    def __init__(self, data_dir, prompt, tokenizer, resolution=512):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"{data_dir} does not exist.")
        self.image_paths = sorted(self.data_dir.iterdir())
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.transforms = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        text_inputs = self.tokenizer(
            self.prompt, truncation=True, padding="max_length", max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": self.transforms(image),
            "input_ids": text_inputs.input_ids,
            "name": str(self.image_paths[index]),
        }


def collate_fn(examples):
    return {
        "pixel_values": torch.stack([e["pixel_values"] for e in examples]).to(memory_format=torch.contiguous_format).float(),
        "input_ids": torch.cat([e["input_ids"] for e in examples], dim=0),
        "names": [e["name"] for e in examples],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="White-box gradient-norm MIA for NetDiffusion.")
    parser.add_argument("--pretrained_model_name_or_path", required=True, help="Base Stable Diffusion model.")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument(
        "--lora_checkpoint_dir",
        required=True,
        help="Directory with checkpoint-* subfolders from NetDiffusion LoRA fine-tuning "
        "(the target model being attacked); the latest checkpoint is loaded, matching "
        "accelerate's `--resume_from_checkpoint=latest` convention.",
    )
    parser.add_argument("--instance_data_dir", required=True, help="Folder of candidate images to score.")
    parser.add_argument("--instance_prompt", required=True, help="Prompt used when the model was fine-tuned.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank used during fine-tuning.")
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_log", required=True, help="Path to write the per-sample loss/grad-norm log.")
    return parser.parse_args()


def load_latest_lora_checkpoint(lora_checkpoint_dir, unet, text_encoder, train_text_encoder):
    """Mirrors accelerate's `--resume_from_checkpoint=latest` + the original
    script's `load_model_hook`, without needing a full Accelerator save/load
    round-trip: find the highest-numbered checkpoint-* dir and load its LoRA
    weights directly onto unet (and text_encoder, if trained)."""
    dirs = [d for d in os.listdir(lora_checkpoint_dir) if d.startswith("checkpoint")]
    if not dirs:
        raise FileNotFoundError(f"No checkpoint-* directories found under {lora_checkpoint_dir}")
    latest = sorted(dirs, key=lambda x: int(x.split("-")[1]))[-1]
    checkpoint_path = os.path.join(lora_checkpoint_dir, latest)

    lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(checkpoint_path)
    unet_state_dict = {k.replace("unet.", ""): v for k, v in lora_state_dict.items() if k.startswith("unet.")}
    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)
    set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")

    if train_text_encoder:
        from diffusers.training_utils import _set_state_dict_into_text_encoder

        _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder)

    return checkpoint_path


def main():
    args = parse_args()

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.lora_checkpoint_dir),
    )
    if args.seed is not None:
        set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, use_fast=False
    )
    text_encoder_cls = import_text_encoder_class(args.pretrained_model_name_or_path, args.revision)

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Re-create the same LoRA adapter shape the target model was fine-tuned with,
    # then load its trained weights (this is the "white-box" part of the attack:
    # the attacker needs the actual fine-tuned parameters, not just query access).
    unet_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"],
    )
    unet.add_adapter(unet_lora_config)
    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder.add_adapter(text_lora_config)

    checkpoint_path = load_latest_lora_checkpoint(args.lora_checkpoint_dir, unet, text_encoder, args.train_text_encoder)
    print(f"Loaded target LoRA checkpoint: {checkpoint_path}")

    if accelerator.mixed_precision == "fp16":
        models = [unet] + ([text_encoder] if args.train_text_encoder else [])
        cast_training_params(models, dtype=torch.float32)

    dataset = CandidateImageDataset(args.instance_data_dir, args.instance_prompt, tokenizer, args.resolution)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    unet, dataloader = accelerator.prepare(unet, dataloader)
    # The model is only ever used in eval mode: we compute gradients purely to
    # measure their norm as a membership signal, and never call optimizer.step().
    unet.eval()

    os.makedirs(os.path.dirname(args.output_log) or ".", exist_ok=True)
    with open(args.output_log, "a") as log_file:
        for step, batch in enumerate(tqdm(dataloader, desc="Scoring candidate samples")):
            pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
            model_input = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device).long()
            noisy_model_input = noise_scheduler.add_noise(model_input, noise, timesteps)

            encoder_hidden_states = text_encoder(batch["input_ids"].to(text_encoder.device), return_dict=False)[0]

            model_pred = unet(noisy_model_input, timesteps, encoder_hidden_states, return_dict=False)[0]
            if model_pred.shape[1] == 6:
                model_pred, _ = torch.chunk(model_pred, 2, dim=1)

            target = noise if noise_scheduler.config.prediction_type == "epsilon" else noise_scheduler.get_velocity(
                model_input, noise, timesteps
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            trainable_params = [p for p in unet.parameters() if p.requires_grad]
            if args.train_text_encoder:
                trainable_params += [p for p in text_encoder.parameters() if p.requires_grad]
            grads = torch.autograd.grad(loss, trainable_params, retain_graph=False, allow_unused=True)

            sq_sum = sum(
                g.detach().reshape(-1).to(torch.float64).pow(2).sum() for g in grads if g is not None
            )
            grad_norm = float(math.sqrt(float(sq_sum)))

            name = batch["names"][0]
            line = f"sample={name}, loss={loss.item()}, grad_norm={grad_norm}"
            print(line)
            log_file.write(line + "\n")


if __name__ == "__main__":
    main()
