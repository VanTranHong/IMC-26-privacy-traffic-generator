"""White-box (loss + gradient-norm) membership inference attack on NetSSM.

Adapted from `conduct_whitebox_attack()` and `calculate_gradient_loss_NetSSM()`
in the local (unpublished) NetSSM training harness's `generate_old.py`
(distinct from the vendored copy at ../../synthetic-data-generation/NetSSM;
cleaned up and parameterized here; no functional changes to the attack
itself). See ./README.md in this folder for the full attack methodology, and
../../synthetic-data-generation/NetSSM for the vendored NetSSM model/tokenizer
code this script depends on (`models/mixer_seq_simple.py`).

Threat model: the attacker has white-box access to the fine-tuned NetSSM
(Mamba) model's weights and gradients. Unlike NetDiffusion (which is
LoRA-fine-tuned, so only the LoRA adapter's gradient norm is measured), NetSSM
is fully fine-tuned, so gradients are taken with respect to *all* model
parameters. For each candidate traffic sample the attacker:
  1. Takes a short prefix of the sample's tokenized packet sequence (e.g. the
     first packet, or the first 3 packets, delimited by `<|pkt|>`).
  2. Runs a single forward + backward pass of the standard causal-LM
     objective (next-token cross-entropy, teacher-forced on the prefix
     itself) -- no optimizer step is taken, so the model's weights are never
     actually updated.
  3. Records the training loss and the L2 norm of the loss gradient with
     respect to *all* model parameters.

Samples the model was fine-tuned on ("train") tend to sit closer to a local
loss minimum and so produce a **lower** loss and **smaller** gradient norm
than held-out ("test") or out-of-distribution ("otheractivity") samples,
which is the membership signal this script measures.
"""

import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer

from models.mixer_seq_simple import MambaLMHeadModel


def extract_packet_prefix(data: str, num_packets: int) -> str:
    """First `num_packets` packets of a `<|label|> ... <|pkt|> ... <|pkt|>...`
    sample string, generalizing the original script's fixed 1-packet/3-packet
    prefixes to an arbitrary packet count."""
    segments = data.split("<|pkt|>")
    if len(segments) > num_packets:
        return "<|pkt|>".join(segments[:num_packets]) + "<|pkt|>"
    return data


def calculate_gradient_loss(model, tokenizer, text: str, device: str, max_length: int = 512):
    """Single forward + backward pass (no optimizer step): returns the
    teacher-forced causal-LM loss and the L2 norm of its gradient w.r.t. all
    model parameters."""
    torch.cuda.empty_cache()
    model.zero_grad()

    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).input_ids.to(device)
    logits = model(input_ids=input_ids, labels=input_ids).logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.CrossEntropyLoss()(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
    )
    loss.backward()

    grad_norm_sq = sum(
        p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None
    )
    return loss.item(), grad_norm_sq ** 0.5


def main():
    parser = argparse.ArgumentParser(description="White-box loss/gradient-norm MIA for NetSSM.")
    parser.add_argument("--model", required=True, help="Path to the target NetSSM (Mamba) checkpoint.")
    parser.add_argument("--tokenizer", required=True, help="Path to the matching custom tokenizer.")
    parser.add_argument(
        "--data_jsonl",
        required=True,
        help="JSONL file of candidate samples (one {'Data': '<|label|> ... <|pkt|> ...'} per line), "
        "e.g. a tokenizer-ready file from ../../synthetic-data-generation/NetSSM/preprocessing.",
    )
    parser.add_argument(
        "--packet_prefixes",
        type=int,
        nargs="+",
        default=[1, 3],
        help="Packet-prefix lengths to score, e.g. '1 3' scores both the first packet and first 3 packets.",
    )
    parser.add_argument("--max_length", type=int, default=512, help="Max tokens per prefix (truncated beyond this).")
    parser.add_argument("--output_csv", required=True, help="Where to write per-sample loss/grad-norm scores.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
    # Note: eval() only disables dropout/etc.; gradients are still computed
    # below since no optimizer step is ever taken, weights are unaffected.
    model.eval()

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    rows = []
    with open(args.data_jsonl, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)["Data"]
            label = data.split(" ")[0][2:-2]
            row = {"line": i, "label": label}
            try:
                for n in args.packet_prefixes:
                    prefix = extract_packet_prefix(data, n)
                    loss, grad_norm = calculate_gradient_loss(model, tokenizer, prefix, device, args.max_length)
                    row[f"loss_{n}packet"] = loss
                    row[f"grad_norm_{n}packet"] = grad_norm
                    print(f"[line {i}] {n}-packet prefix - loss={loss:.4f}, grad_norm={grad_norm:.4f}")
                rows.append(row)
            except Exception as e:
                print(f"[line {i}] error: {e}")

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved white-box loss/gradient-norm scores to {args.output_csv}")


if __name__ == "__main__":
    main()
