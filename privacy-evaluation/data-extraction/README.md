# Data-Extraction Attacks on NetDiffusion, NetSSM, and TrafficLLM

This folder documents and provides cleaned, runnable implementations of the
data-extraction attacks used to check whether these three synthetic traffic
generators leak specific, identifiable values (IP addresses, MAC addresses,
packet-header fingerprints) from their training data through generation.

## Attribution

All four scripts here are adapted from `calculate_memorization_attack.py` in
a private internal research harness (not included in this repo):
- `calculate_cc()` → [`extraction_common.py`](./extraction_common.py)
- `coverage_confidence_NetSSM()` → [`netssm_data_extraction.py`](./netssm_data_extraction.py)
- `coverage_confidence_TrafficLLM()` → [`trafficllm_data_extraction.py`](./trafficllm_data_extraction.py)
- `calculate_memorized_sequence_NetDiffusion()` → [`netdiffusion_data_extraction.py`](./netdiffusion_data_extraction.py)

They've been cleaned up (no hardcoded personal paths, parameterized via
`argparse`) and **the original "coverage"/"confidence" metrics have been
renamed and reframed as recall/precision** (same underlying counts — see
below for the exact correspondence). The three target models are attributed
in full under [`../../synthetic-data-generation/`](../../synthetic-data-generation/)
(NetDiffusion, NetSSM, TrafficLLM); this folder only concerns the attacks
against them.

---

## The attack, in general

A data-extraction attack asks: **can an attacker who only gets to generate
samples from the model recover real, specific values from its training
data** (an IP address, a MAC address, an exact packet structure) — as
opposed to membership inference (see [`../MIA`](../MIA)), which asks whether
one *known* candidate sample was a training member. Here the attacker
doesn't need candidate samples up front: they generate many outputs, extract
one field of interest from each, and check how many of those extracted
values are real training values.

### Precision and recall (replacing "coverage"/"confidence")

Let `original_values` be every occurrence of the target field across the
real training data (a list, repeats included — frequency matters), and
`generated_values` be every occurrence of that field across the generated
outputs.

| Metric | Definition | Original script's name |
|---|---|---|
| **Precision** | (# generated occurrences whose value is a real training value) / (# generated occurrences) — *"of everything the model generated, how much was actually memorized training data?"* | `confidence` |
| **Recall** | (# unique training values reproduced at least once) / (# unique training values) — *"of everything in the training set, how much did the attacker manage to recover?"* | `coverage` |
| **F1** | harmonic mean of precision and recall | *(not previously reported)* |

All three scripts share this scoring logic via
[`extraction_common.precision_recall_extraction()`](./extraction_common.py) —
each model-specific script only differs in **how it extracts the target
field's values** from that model's own data/output format.

---

## Workflow (the same three steps for every model)

1. **Pick a field to attack.** Each script defaults to the field the original
   research code targeted (NetSSM: destination MAC address; TrafficLLM:
   source IP address; NetDiffusion: a packet-header bit-column fingerprint —
   see why below), but all are configurable to attack a different field.
2. **Extract that field's values** from (a) the real training data and
   (b) a batch of generated/reconstructed samples.
3. **Score with precision/recall** via `extraction_common.precision_recall_extraction()`.

Run each script once per dataset/checkpoint you want to evaluate.

---

## NetSSM: MAC-address extraction

NetSSM generates packets as space-separated decimal byte strings
(`<|netflix|> 226 70 154 ... <|pkt|> 00 128 153 ...`). Bytes `[0:6]` of the
first packet are the Ethernet **destination MAC address** (any other fixed
byte range is scoreable via `--byte_start`/`--byte_end`, e.g. `[6:12]` for
the source MAC).

```bash
python netssm_data_extraction.py \
  --original_jsonl /path/to/preprocessing/tokenized_IoT_train_singleflow.jsonl \
  --generated_glob "/path/to/inference/IoT_singleflow_.../*/singleprompt_*.txt" \
  --byte_start 0 --byte_end 6 \
  --field_name dst_mac
```

## TrafficLLM: IP-address extraction

TrafficLLM's generation-task outputs are `{'src': '1.2.3.4', 'dst': ..., ...} <hex payload>`.
This script pulls the `src` (or `dst`) IP string out of that header dict.

```bash
python trafficllm_data_extraction.py \
  --original_jsonl /path/to/output_TrafficLLM/VNAT_final_train/VNAT_final_train_preprocessed_generation_packet_train.json \
  --generated_glob "/path/to/inference_new/VNAT_1epochs/*/*_maxlen_300_temp_0_9.txt" \
  --field src
```

## NetDiffusion: packet-header fingerprint extraction

NetDiffusion's own preprocessing (see `nprint_to_png.py` in
[`../../synthetic-data-generation/NetDiffusion`](../../synthetic-data-generation/NetDiffusion))
**deliberately drops IP-address columns before training**, specifically to
avoid encoding identifiable IPs into the model — so an IP-extraction attack
doesn't apply here by construction. Instead this script fingerprints each
packet's structural header: the values of whichever nprint bit-columns are
shared between the real `.nprint` files and the reconstructed
`best_reconstruction.nprint` files, condensed to one character per bit
(`-1`→`*`, `0`→`0`, else→`1`, per nprint's own convention).

```bash
python netdiffusion_data_extraction.py \
  --original_nprint_glob "/path/to/data_MIA_test/all-labels/IoT_training/train_nprint/*.nprint" \
  --generated_nprint_glob "/path/to/inference/IoT/train/*/best_reconstruction.nprint"
```
Add `--all_packets` to fingerprint every packet in each flow instead of just
the first, or `--columns col1 col2 ...` to restrict the fingerprint to
specific nprint columns instead of every shared one.

---

## Interpreting results

- **High precision** ⇒ almost everything the model generates for that field
  is a real training value — the model has essentially memorized (and is
  freely reproducing) that field.
- **High recall** ⇒ generating enough samples eventually surfaces most of
  the training set's distinct values for that field — even if any single
  generation isn't obviously "the same as training", the training
  vocabulary as a whole is recoverable.
- Both near 0 is the desired outcome for a privacy-preserving generator:
  synthetic field values should look plausible without reproducing real
  ones.
