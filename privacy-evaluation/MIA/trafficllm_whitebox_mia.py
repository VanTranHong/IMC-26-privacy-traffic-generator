"""White-box (loss + gradient-norm) membership inference attack on TrafficLLM.

Adapted from `compute_metrics_for_input()` and the "whitebox_data" evaluation
loop in the local (unpublished) TrafficLLM training harness's `evaluation.py`
(distinct from the vendored copy at ../../synthetic-data-generation/TrafficLLM;
cleaned up and parameterized here; see ./trafficllm_blackbox_mia.py's docstring for how the
two attacks were split back out of that one file). See ./README.md in this
folder for the full attack methodology, and
../../synthetic-data-generation/TrafficLLM for the vendored TrafficLLM code
this depends on.

Threat model: the attacker has white-box access to the fine-tuned TrafficLLM
model's weights and gradients (e.g., a released P-Tuning/LoRA checkpoint). For
each candidate sample (an instruction/traffic-data prompt paired with its
target output), the attacker:
  1. Tokenizes `prompt + target`, masking out the prompt tokens (label = -100)
     so the loss is only computed over the target response.
  2. Runs a single forward + backward pass of the causal-LM objective on the
     target response -- no optimizer step is taken, so the model's weights
     are never actually updated.
  3. Records the loss, perplexity, and the L2 norm of the loss gradient with
     respect to all parameters that receive a gradient (in practice, with
     P-Tuning, this is just the trainable prefix-encoder parameters -- the
     base model is frozen).

Samples the model was fine-tuned on ("train") tend to sit closer to a local
loss minimum and so produce a **lower loss/perplexity and smaller gradient
norm** than held-out ("test") or out-of-distribution samples, which is the
membership signal this script measures.
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


def compute_loss_perplexity_gradnorm(model, tokenizer, prompt: str, target: str, max_length: int = 2048):
    """Single forward + backward pass (no optimizer step): returns the
    teacher-forced loss/perplexity of `target` given `prompt`, and the L2
    norm of the loss gradient w.r.t. all parameters with `requires_grad`."""
    full_text = prompt + target
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    prompt_length = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].shape[1]
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100

    model.zero_grad()
    with torch.enable_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
        loss.backward()

        grad_norm_sq = sum(
            p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None
        )

    result = (loss.detach().cpu().item(), perplexity, grad_norm_sq ** 0.5)
    del inputs, labels, outputs, loss
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description="White-box loss/gradient-norm MIA for TrafficLLM.")
    parser.add_argument("--model_name", required=True, help="Base model path/name, e.g. THUDM/chatglm2-6b.")
    parser.add_argument("--ptuning_path", default=None, help="PEFT/P-Tuning checkpoint dir (the target model being attacked).")
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

            torch.cuda.empty_cache()
            loss, perplexity, grad_norm = compute_loss_perplexity_gradnorm(
                model, tokenizer, prompt, target, args.max_length
            )
            print(f"[{i}] loss={loss:.4f}, perplexity={perplexity:.4f}, grad_norm={grad_norm:.4f}")
            rows.append({"line": i, "loss": loss, "perplexity": perplexity, "gradnorm": grad_norm})

    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(f"Saved white-box loss/perplexity/gradient-norm scores to {args.output_csv}")


if __name__ == "__main__":
    main()
