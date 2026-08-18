"""Black-box (reconstruction-loss) membership inference attack on NetDiffusion.


Threat model: the attacker only has query access to the fine-tuned NetDiffusion
pipeline (Stable Diffusion + ControlNet-Canny + LoRA) -- no access to model
weights or gradients. For each candidate traffic sample, the attacker:
  1. Converts the sample's ground-truth traffic image to a Canny edge map.
  2. Conditions the pipeline on that edge map to regenerate ("reconstruct")
     the sample (this reconstruction step is done upstream by NetDiffusion's
     own generation pipeline -- see ../../synthetic-data-generation/NetDiffusion
     -- and is expected to have already produced `input.png` /
     `best_reconstruction.png` pairs on disk before this script runs).
  3. Scores the reconstruction by pixel-wise MSE against the ground truth.

Samples the model was fine-tuned on ("train") tend to reconstruct with lower
MSE than held-out samples ("test") or out-of-distribution samples
("test_otheractivity"), which is the membership signal this script measures.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image


def reconstruction_loss_mse(real_image_path: str, generated_image_path: str) -> float:
    """Pixel-wise MSE between a ground-truth image and its reconstruction."""
    real = Image.open(real_image_path).convert("RGB").resize((512, 512), resample=Image.NEAREST)
    generated = Image.open(generated_image_path).convert("RGB").resize((512, 512), resample=Image.NEAREST)

    real = np.array(real)
    generated = np.array(generated)

    if real.shape != generated.shape:
        raise ValueError("Images must be the same size.")

    return float(np.mean((real - generated) ** 2))


def score_subset(generated_outputs_dir: str, dataset: str, subset: str, output_dir: str) -> str:
    """Compute reconstruction-loss MIA scores for every sample in one subset.

    Expects `{generated_outputs_dir}/{dataset}/{subset}/*/` folders, each
    containing an `input.png` (ground truth) and a `best_reconstruction.png`
    (the model's regenerated reconstruction of that input).
    """
    sample_glob = os.path.join(generated_outputs_dir, dataset, subset, "*/")
    os.makedirs(output_dir, exist_ok=True)

    names, losses = [], []
    for sample_dir in glob.glob(sample_glob):
        real_path = os.path.join(sample_dir, "input.png")
        gen_path = os.path.join(sample_dir, "best_reconstruction.png")
        if os.path.exists(real_path) and os.path.exists(gen_path):
            losses.append(reconstruction_loss_mse(real_path, gen_path))
            names.append(sample_dir)

    out_path = os.path.join(output_dir, f"reconstruction_loss_{subset}.csv")
    pd.DataFrame({"name": names, "loss": losses}).to_csv(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Black-box reconstruction-loss MIA for NetDiffusion.")
    parser.add_argument(
        "--generated_outputs_dir",
        required=True,
        help="Root folder containing {dataset}/{subset}/*/{input.png,best_reconstruction.png}.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. CIRA-CIC-DoHBrw-2020_multiflow_1.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train", "test", "test_otheractivity"],
        help="Subsets to score. 'train' = members, 'test' = held-out non-members, "
        "'test_otheractivity' = out-of-distribution negative control.",
    )
    parser.add_argument("--output_dir", required=True, help="Where to write reconstruction_loss_{subset}.csv files.")
    args = parser.parse_args()

    for subset in args.subsets:
        out_path = score_subset(args.generated_outputs_dir, args.dataset, subset, args.output_dir)
        print(f"[{subset}] saved reconstruction-loss scores to {out_path}")


if __name__ == "__main__":
    main()
