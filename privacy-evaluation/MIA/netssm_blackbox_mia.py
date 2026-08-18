"""Black-box (perplexity) membership inference attack on NetSSM.

Threat model: the attacker only needs the ability to score a sequence's
likelihood under the fine-tuned NetSSM (Mamba) model -- a forward pass under
`torch.no_grad()`, no gradients or weight access required. For each candidate
traffic sample the attacker:
  1. Takes a short prefix of the sample's tokenized packet sequence (e.g. the
     first packet, or the first 3 packets, delimited by `<|pkt|>`).
  2. Computes that prefix's perplexity under the target model via teacher-
     forced cross-entropy loss (exp(loss)).

Samples the model was fine-tuned on ("train") tend to have **lower**
perplexity (the model assigns them higher likelihood) than held-out ("test")
or out-of-distribution ("otheractivity") samples, which is the membership
signal this script measures.
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


def calculate_perplexity_on_sample(model, tokenizer, sample: str, device: str):
    """Teacher-forced perplexity of `sample` under `model` (no gradients)."""
    input_ids = tokenizer(sample, return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, labels=input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = torch.nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    return input_ids.size(1), torch.exp(loss).item()


def main():
    parser = argparse.ArgumentParser(description="Black-box perplexity MIA for NetSSM.")
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
    parser.add_argument("--output_csv", required=True, help="Where to write per-sample perplexity scores.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=dtype)
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
                    length, perplexity = calculate_perplexity_on_sample(model, tokenizer, prefix, device)
                    row[f"prompt_length_{n}packet"] = length
                    row[f"perplexity_{n}packet"] = perplexity
                    print(f"[line {i}] {n}-packet prefix - length={length}, perplexity={perplexity:.4f}")
                rows.append(row)
            except Exception as e:
                print(f"[line {i}] error: {e}")

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved black-box perplexity scores to {args.output_csv}")


if __name__ == "__main__":
    main()
