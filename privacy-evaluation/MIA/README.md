# Membership Inference Attacks (MIA) on NetDiffusion, NetSSM, TrafficLLM, and NetShare

This folder documents and provides cleaned, runnable implementations of the
black-box and white-box membership inference attacks used to evaluate the
privacy of four synthetic traffic generators in this repo:
[**NetDiffusion**](../../synthetic-data-generation/NetDiffusion) (Stable
Diffusion + ControlNet + LoRA), [**NetSSM**](../../synthetic-data-generation/NetSSM)
(a fine-tuned Mamba state-space language model),
[**TrafficLLM**](../../synthetic-data-generation/TrafficLLM) (a P-Tuned/LoRA
ChatGLM2 model), and [**NetShare**](../../synthetic-data-generation/NetShare)
(a per-flow DoppelGANger-style GAN, black-box attack only so far).

---

# NetDiffusion

## Attribution

Both scripts here are adapted from the local (unpublished) NetDiffusion
training harness's DreamBooth-based fine-tuning scripts (not included in this repo):
- `run_NetDiffusion_blackbox.py` → [`netdiffusion_blackbox_mia.py`](./netdiffusion_blackbox_mia.py)
- `run_NetDiffusion_whitebox.py` → [`netdiffusion_whitebox_mia.py`](./netdiffusion_whitebox_mia.py)

They've been cleaned up (no hardcoded personal paths, parameterized via
`argparse`) but the attack logic itself is unchanged. The white-box script's
model-loading/LoRA scaffold is itself derived from HuggingFace `diffusers`'
official DreamBooth-LoRA training example
([`examples/dreambooth/train_dreambooth_lora.py`](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py)).
The target model under attack, NetDiffusion, is from
[arXiv:2310.08543](https://arxiv.org/abs/2310.08543) /
[github.com/noise-lab/NetDiffusion](https://github.com/noise-lab/NetDiffusion).

## Threat model

An attacker wants to determine whether a specific network-traffic sample was
part of the dataset used to fine-tune a NetDiffusion LoRA checkpoint. Samples
are scored under three splits:
- **`train`** — samples that *were* used for fine-tuning (ground-truth members).
- **`test`** — held-out samples from the same dataset (ground-truth non-members).
- **`test_otheractivity`** — samples from a different traffic class entirely
  (out-of-distribution negative control).

A successful attack is one where members score systematically differently
(lower reconstruction error / lower gradient norm) than non-members.

## Attack 1: Black-box (reconstruction loss)

**Access assumed:** query-only access to the fine-tuned generation pipeline —
no model weights or gradients.

**Method:**
1. For each candidate sample, take its ground-truth traffic image and extract
   a Canny edge map.
2. Condition the fine-tuned Stable Diffusion + ControlNet-Canny pipeline on
   that edge map and regenerate ("reconstruct") the sample (several candidates
   are sampled and the best one kept as `best_reconstruction.png`; this
   generation step is done upstream by NetDiffusion's own pipeline in
   `../../synthetic-data-generation/NetDiffusion`, not by the script here).
3. Score the reconstruction by pixel-wise MSE against the real `input.png`.

**Signal:** members tend to reconstruct with **lower** MSE than non-members,
because the model has "seen" and fit to that sample during fine-tuning.

**Usage:**
```bash
python netdiffusion_blackbox_mia.py \
  --generated_outputs_dir /path/to/generated_outputs \
  --dataset CIRA-CIC-DoHBrw-2020_multiflow_1 \
  --subsets train test test_otheractivity \
  --output_dir ./results/blackbox
```
Expects `{generated_outputs_dir}/{dataset}/{subset}/*/` folders each
containing `input.png` and `best_reconstruction.png`. Writes one
`reconstruction_loss_{subset}.csv` per subset (columns: `name`, `loss`).

## Attack 2: White-box (per-sample gradient norm)

**Access assumed:** full access to the fine-tuned model's weights and the
ability to compute gradients (e.g., a released LoRA checkpoint).

**Method:**
1. Load the base Stable Diffusion model and attach a LoRA adapter matching
   the target checkpoint's configuration, then load the target checkpoint's
   trained LoRA weights (the highest-numbered `checkpoint-*` directory).
2. For each candidate sample, run a single forward + backward pass of the
   standard diffusion training objective (noise-prediction MSE at a random
   timestep) — **no optimizer step is taken**, so the model's weights never
   actually change. This is purely used to measure the loss landscape at that
   point, not to train further.
3. Record the training loss and the L2 norm of the loss gradient with
   respect to the trainable (LoRA) parameters.

**Signal:** members tend to sit closer to a local minimum of the fine-tuned
model's loss landscape and so produce a **smaller** gradient norm than
held-out or out-of-distribution samples.

**Usage:**
```bash
python netdiffusion_whitebox_mia.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --lora_checkpoint_dir /path/to/lora_MIA_allcheckpoints/<run> \
  --instance_data_dir /path/to/candidate_images \
  --instance_prompt "pixilated network traffic type 0" \
  --rank 4 \
  --mixed_precision fp16 \
  --output_log ./results/whitebox/gradnorm.txt
```
Run once per split (`train`, `test`, `test_otheractivity`) by pointing
`--instance_data_dir` at each split's image folder, and compare the resulting
`loss`/`grad_norm` distributions across splits.

## Evaluating attack success (NetDiffusion)

Neither script computes an attack success metric directly — they only produce
per-sample scores (`loss` in the black-box CSVs, `loss`/`grad_norm` in the
white-box log). To evaluate the attack, compare the score distributions
between `train` (members) and `test`/`test_otheractivity` (non-members) —
e.g., with an ROC-AUC over "is this sample a member" using the (negated)
score as the decision threshold, or a simple distributional comparison
(mean/median difference, KS test).

---

# NetSSM

## Attribution

Both scripts here are adapted from `conduct_blackbox_attack()`,
`conduct_whitebox_attack()`, `calculate_perplexity_on_sample()`, and
`calculate_gradient_loss_NetSSM()` in the local (unpublished) NetSSM training
harness's `generate_old.py` (distinct from the vendored copy at
`../../synthetic-data-generation/NetSSM`; not included in this repo):
- `conduct_blackbox_attack()` → [`netssm_blackbox_mia.py`](./netssm_blackbox_mia.py)
- `conduct_whitebox_attack()` → [`netssm_whitebox_mia.py`](./netssm_whitebox_mia.py)

They've been cleaned up (no hardcoded personal paths, parameterized via
`argparse`, generalized from fixed 1-/3-packet prefixes to an arbitrary list)
but the attack logic itself is unchanged. Both scripts depend on the
`models.mixer_seq_simple.MambaLMHeadModel` class vendored at
[`../../synthetic-data-generation/NetSSM/models`](../../synthetic-data-generation/NetSSM/models)
(run them with that folder's parent on your `PYTHONPATH`, or copy `models/`
alongside these scripts). The target model under attack, NetSSM, is from
[github.com/noise-lab/netssm](https://github.com/noise-lab/netssm).

## Threat model

Same as NetDiffusion above: determine whether a specific network-traffic flow
was part of the dataset used to fine-tune a NetSSM checkpoint, scored across
`train` (members), `test` (held-out non-members), and `otheractivity`
(out-of-distribution negative control) splits. Each candidate flow is
truncated to a short packet-prefix (e.g. the first packet, or the first 3
packets) before scoring, matching how NetSSM itself is trained on packet
sequences delimited by `<|pkt|>`.

## Attack 1: Black-box (perplexity)

**Access assumed:** the ability to score a sequence's likelihood under the
fine-tuned model (a forward pass under `torch.no_grad()`) — no gradients or
weight-update access required.

**Method:** for each candidate sample's packet-prefix, compute its perplexity
under the target model via teacher-forced cross-entropy loss (`exp(loss)`).

**Signal:** members tend to have **lower** perplexity (the model assigns them
higher likelihood) than held-out or out-of-distribution samples.

**Usage:**
```bash
python netssm_blackbox_mia.py \
  --model /path/to/checkpoints/IoT_checkpoints_1epoch_singleflow/checkpoint-412 \
  --tokenizer /path/to/tokenizers/tokenizer_single_flow_IoT \
  --data_jsonl /path/to/preprocessing/tokenized_IoT_train_singleflow.jsonl \
  --packet_prefixes 1 3 \
  --output_csv ./results/netssm_blackbox/IoT_train.csv
```
Run once per split by pointing `--data_jsonl` at each split's JSONL file.
Writes one CSV with per-sample `perplexity_{n}packet` columns for each
requested prefix length.

## Attack 2: White-box (loss + gradient norm)

**Access assumed:** full access to the fine-tuned model's weights and
gradients. Unlike NetDiffusion (LoRA fine-tuned, so only the adapter's
gradient norm is measured), NetSSM is fully fine-tuned, so gradients are
taken with respect to **all** model parameters.

**Method:** for each candidate sample's packet-prefix, run a single forward +
backward pass of the causal-LM objective (teacher-forced next-token
cross-entropy) — **no optimizer step is taken**, so the model's weights never
actually change. Record the loss and the L2 norm of its gradient w.r.t. all
parameters.

**Signal:** members tend to sit closer to a local loss minimum and so
produce a **lower loss and smaller gradient norm** than held-out or
out-of-distribution samples.

**Usage:**
```bash
python netssm_whitebox_mia.py \
  --model /path/to/checkpoints/CIRA-CIC-DoHBrw-2020_checkpoints_20epoch/checkpoint-11700 \
  --tokenizer /path/to/tokenizers/tokenizer_CIRA-CIC-DoHBrw-2020 \
  --data_jsonl /path/to/preprocessing/tokenized_CIRA-CIC-DoHBrw-2020_train_truncated_200.jsonl \
  --packet_prefixes 1 3 \
  --output_csv ./results/netssm_whitebox/CIRA-CIC-DoHBrw-2020_train.csv
```
Run once per split, then compare the resulting `loss_{n}packet` /
`grad_norm_{n}packet` distributions across splits.

## Evaluating attack success (NetSSM)

As with NetDiffusion, these scripts only produce per-sample scores — compare
the `perplexity` / `loss` / `grad_norm` distributions between `train`
(members) and `test`/`otheractivity` (non-members), e.g. via ROC-AUC or a
distributional test (mean/median difference, KS test).

---

# TrafficLLM

## Attribution

Both scripts here are adapted from `evaluation.py` in the local (unpublished)
TrafficLLM training harness (distinct from the vendored copy at
`../../synthetic-data-generation/TrafficLLM`; not included in this repo).
That single file has two attacks tangled into one `compute_metrics_for_input()` function
plus two mostly-commented-out evaluation loops (one saving to a
`blackbox_data/` folder, one to a `whitebox_data/` folder) — this pair of
scripts splits them back out along the actual black-box/white-box access
boundary they were meant to represent:
- The commented-out `torch.no_grad()` variant of `compute_metrics_for_input()`
  + the `blackbox_data` loop → [`trafficllm_blackbox_mia.py`](./trafficllm_blackbox_mia.py)
- The `torch.enable_grad()` variant of `compute_metrics_for_input()`
  + the `whitebox_data` loop → [`trafficllm_whitebox_mia.py`](./trafficllm_whitebox_mia.py)

No hardcoded personal paths remain (everything is `argparse`-parameterized),
and the model-loading logic matches what's already vendored at
[`../../synthetic-data-generation/TrafficLLM`](../../synthetic-data-generation/TrafficLLM)
(see `tutorials/generation.py`'s `load_model()` for the same P-Tuning-checkpoint
loading pattern). The target model under attack, TrafficLLM, is from
[arXiv:2504.04222](https://arxiv.org/abs/2504.04222) /
[github.com/ZGC-LLM-Safety/TrafficLLM](https://github.com/ZGC-LLM-Safety/TrafficLLM),
built on [ChatGLM2](https://github.com/THUDM/ChatGLM2-6B).

## Threat model

Same as NetSSM above: determine whether a specific (instruction, traffic-data)
sample was part of the dataset used to fine-tune a TrafficLLM P-Tuning/LoRA
checkpoint, scored across `train` (members) and `test`/out-of-distribution
splits. Each sample is TrafficLLM's own `{instruction, output}` format, where
`output` for generation tasks is `{header dict} <hex payload>`.

## Attack 1: Black-box (perplexity)

**Access assumed:** the ability to score a sequence's likelihood under the
fine-tuned model (a forward pass under `torch.no_grad()`) — no gradients or
weight-update access required.

**Method:** tokenize `prompt + target`, mask the prompt tokens out of the
loss (label = -100), and compute the target's perplexity under the target
model — once for the full target response, and once for just the hex-payload
portion after the last `}` (TrafficLLM's generation outputs are a header dict
followed by hex bytes, so this separates "did it memorize the header" from
"did it memorize the payload").

**Signal:** members tend to have **lower** perplexity than held-out or
out-of-distribution samples.

**Usage:**
```bash
python trafficllm_blackbox_mia.py \
  --model_name THUDM/chatglm2-6b \
  --ptuning_path /path/to/models/chatglm2/peft/ustc-tfc-2016-generation-packet-header/checkpoint-10000 \
  --test_file /path/to/datasets/ustc-tfc-2016/ustc-tfc-2016_generation_packet_train.json \
  --output_csv ./results/trafficllm_blackbox/ustc-tfc-2016_train.csv
```
Run once per split. Writes a CSV with `loss`, `perplexity`, `loss_hex`,
`perplexity_hex` columns per line.

## Attack 2: White-box (loss + gradient norm)

**Access assumed:** full access to the fine-tuned model's weights and
gradients (e.g., a released P-Tuning/LoRA checkpoint).

**Method:** same prompt/target masking as the black-box attack, but run a
single forward + backward pass (**no optimizer step**) and record the L2 norm
of the loss gradient with respect to every parameter that receives one — in
practice just the trainable prefix-encoder (P-Tuning) parameters, since the
8-bit-loaded base model is frozen.

**Signal:** members tend to sit closer to a local loss minimum and so
produce a **lower loss/perplexity and smaller gradient norm** than held-out
or out-of-distribution samples.

**Usage:**
```bash
python trafficllm_whitebox_mia.py \
  --model_name THUDM/chatglm2-6b \
  --ptuning_path /path/to/models/chatglm2/peft/ustc-tfc-2016-generation-packet-header/checkpoint-10000 \
  --test_file /path/to/datasets/ustc-tfc-2016/ustc-tfc-2016_generation_packet_train.json \
  --output_csv ./results/trafficllm_whitebox/ustc-tfc-2016_train.csv
```
Run once per split. Writes a CSV with `loss`, `perplexity`, `gradnorm`
columns per line.

## Evaluating attack success (TrafficLLM)

As with the other two models, these scripts only produce per-sample scores —
compare the `perplexity` / `loss` / `gradnorm` distributions between `train`
(members) and held-out/out-of-distribution splits, e.g. via ROC-AUC or a
distributional test (mean/median difference, KS test).

---

# NetShare

Only a black-box attack has been recovered for NetShare so far (no equivalent
white-box/gradient-norm script was found in the local experiment code).

## Attribution

Adapted from `extract_features_single_df_NetShare()` and the NetShare-specific
driving loop in `calculate_MIA_attack.py`, in a private internal research
harness (not included in this repo)
→ [`netshare_blackbox_mia.py`](./netshare_blackbox_mia.py). Cleaned up (no
hardcoded personal paths, parameterized via `argparse`) with no functional
change to the feature extraction itself. The target model under attack,
NetShare, is from the SIGCOMM 2022 paper — see
[`../../synthetic-data-generation/NetShare`](../../synthetic-data-generation/NetShare)
for the full attribution and vendored code, including the `examples/pcap/driver.py`
that produces the per-flow output layout this script reads.

## Threat model

NetShare is trained **one model per real flow** (see
`../../synthetic-data-generation/NetShare/examples/pcap/driver.py`), so unlike
the other three models — where "is sample X a member" is answered directly
from a single shared fine-tuned model — here the question becomes: does the
flow-conditioned synthetic output statistically resemble a candidate flow well
enough to identify it? The attacker only has query access to NetShare's
*output* (per-flow synthetic packet-header CSVs), not to model weights or
gradients.

## Attack: Black-box (statistical features)

**Access assumed:** query-only access to the generator's synthetic CSV output
per flow.

**Method:**
1. For each per-flow synthetic CSV (`train`, `test`, and `otheractivity`
   splits), compute a fixed-size statistical feature vector: packet count,
   unique src/dst IP and port counts, TCP/UDP/ICMP protocol fractions,
   packet-length/ToS/TTL mean/std/min/max, and inter-arrival-time mean/std.
2. Label each feature vector with the identity of the original flow it was
   generated from (recovered from the per-flow output folder name).
3. Concatenate into one feature table per split.

**Signal:** this script only produces the feature tables, not a decision —
see "Evaluating attack success" below.

**Usage:**
```bash
python netshare_blackbox_mia.py \
  --generated_data_dir /path/to/NetShare/generated_data_100epoch \
  --dataset servicerecognition \
  --subsets train test otheractivity \
  --epoch 99 \
  --output_dir ./results/netshare_blackbox
```
Expects `{generated_data_dir}/{dataset}/{subset}/{flow_name}/generated_data/sample_len-10/syn_dfs/chunk_id-0/epoch_id-{epoch}.csv`
per flow. Writes one `NetShare_features_{subset}.csv` per subset.

## Evaluating attack success (NetShare)

Unlike the other three models (a single scalar score per sample), NetShare's
attack produces a multi-dimensional feature vector per flow, so a simple
threshold isn't enough — the actual membership decision requires training a
binary classifier (e.g. random forest, or AutoGluon's `TabularPredictor`) to
distinguish `train`-split feature vectors from `test`/`otheractivity`-split
ones, then reporting its held-out ROC-AUC / TPR-at-low-FPR. A private
internal research harness (not included in this repo) has such an evaluator,
`run_autogluon.py` (`MIA_attack()`), reused across all four models' feature/score tables — it
wasn't ported here as-is because its train/test labeling looked like it
conflates the `train` and `test` splits into a single "same-distribution"
class against `otheractivity` alone, rather than testing `train` vs `test`
membership directly; double-check that logic before reusing it.
