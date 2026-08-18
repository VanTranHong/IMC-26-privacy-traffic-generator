"""Black-box (perplexity) membership inference attack on TrafficLLM.

Adapted from the (commented-out) no-gradient `compute_metrics_for_input()`
variant and the "blackbox_data" evaluation loop in the local (unpublished)
TrafficLLM training harness's `evaluation.py` (distinct from the vendored
copy at ../../synthetic-data-generation/TrafficLLM; cleaned up and
parameterized here; the original file has two attacks tangled into one
function -- this script and ./trafficllm_whitebox_mia.py split them back out
along the black-box/white-box access boundary they were originally meant to
represent). See ./README.md in this folder for the full attack methodology,
and ../../synthetic-data-generation/TrafficLLM for the vendored TrafficLLM
code this depends on.

Threat model: the attacker only needs the ability to score a sequence's
likelihood under the fine-tuned TrafficLLM (ChatGLM2 + P-Tuning/LoRA) model --
a forward pass under `torch.no_grad()`, no gradients or weight access
required. For each candidate sample (an instruction/traffic-data prompt paired
with its target output), the attacker:
  1. Tokenizes `prompt + target`, masking out the prompt tokens (label = -100)
     so the loss is only computed over the target response.
  2. Computes that target's perplexity under the target model, both for the
     full response and for the hex-payload-only portion of it (the substring
     after the last `}`, i.e. excluding the generated header dict) -- since
     TrafficLLM's generation-task outputs are `{header dict} <hex payload>`.

Samples the model was fine-tuned on ("train") tend to have **lower**
perplexity (the model assigns them higher likelihood) than held-out ("test")
or out-of-distribution samples, which is the membership signal this script
measures.
"""

import argparse
import json
import os
from typing import Optional

import pandas as pd
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_model(model_name: str, ptuning_path: Optional[str]):
    """Mirrors TrafficLLM's own model-loading pattern (see
    ../../synthetic-data-generation/TrafficLLM/tutorials/generation.py and
    evaluation.py): load the base model, and if a PEFT/P-Tuning checkpoint is
    given, load its prefix-encoder weights on top."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if ptuning_path is not None:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, pre_seq_len=128)
        model = AutoModel.from_pretrained(
            model_name, config=config, trust_remote_code=True, load_in_8bit=True, device_map="auto"
        )
        prefix_state_dict = torch.load(os.path.join(ptuning_path, "pytorch_model.bin"))
        new_prefix_state_dict = {
            k[len("transformer.prefix_encoder."):]: v
            for k, v in prefix_state_dict.items()
            if k.startswith("transformer.prefix_encoder.")
        }
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        model = model.cuda()
        model.transformer.prefix_encoder.float()
    else:
        model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, load_in_8bit=True, device_map="auto"
        ).cuda()

    return model.eval(), tokenizer


def compute_loss_perplexity(model, tokenizer, prompt: str, target: str, max_length: int = 2048):
    """Teacher-forced loss/perplexity of `target` given `prompt`, under
    `torch.no_grad()` -- no gradients are ever computed, matching the
    black-box threat model (query access to the model's output likelihood
    only)."""
    full_text = prompt + target
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    prompt_length = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        loss = outputs.loss
        perplexity = torch.exp(loss).item() if loss is not None else float("inf")

    return (loss.item() if loss is not None else None), perplexity


def main():
    parser = argparse.ArgumentParser(description="Black-box perplexity MIA for TrafficLLM.")
    parser.add_argument("--model_name", required=True, help="Base model path/name, e.g. THUDM/chatglm2-6b.")
    parser.add_argument("--ptuning_path", default=None, help="Optional PEFT/P-Tuning checkpoint dir (the target model being attacked).")
    parser.add_argument(
        "--test_file",
        required=True,
        help="JSONL file of candidate samples (one {'instruction': ..., 'output': ...} per line), "
        "matching TrafficLLM's own train/test data format.",
    )
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, args.ptuning_path)

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    rows = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            prompt = record["instruction"]
            target = record["output"]
            target_hex_only = target.split("}")[-1]

            torch.cuda.empty_cache()
            loss, perplexity = compute_loss_perplexity(model, tokenizer, prompt, target, args.max_length)
            loss_hex, perplexity_hex = compute_loss_perplexity(model, tokenizer, prompt, target_hex_only, args.max_length)

            print(f"[{i}] perplexity={perplexity:.4f} | hex-only perplexity={perplexity_hex:.4f}")
            rows.append(
                {"line": i, "loss": loss, "perplexity": perplexity, "loss_hex": loss_hex, "perplexity_hex": perplexity_hex}
            )

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved black-box perplexity scores to {args.output_csv}")


if __name__ == "__main__":
    main()
