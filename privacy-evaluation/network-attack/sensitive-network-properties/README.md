# Sensitive-Network-Property Divergence Attacks

This folder documents and provides cleaned, runnable implementations of the
distributional divergence attacks used to check whether NetDiffusion,
NetSSM, TrafficLLM, and NetShare reproduce the real *distribution* of
sensitive or behavioral packet/flow properties (TTL, TCP window size, TCP
flags, packet size, flow byte totals) from their training data.

## Attribution

Adapted from `calculate_divergence.py` in a private internal research
harness (not included in this repo), and merged into a single, section-organized script,
[`divergence_attacks.py`](./divergence_attacks.py):
- The repeated normalized-Wasserstein-distance pattern, `tcp_flags_to_number()`,
  `convert_hex_string()`, `extract_field_from_packet_bytes()`, and the
  nprint bit-decoding logic → the "Shared scoring core" / "Shared
  packet/field-decoding helpers" sections
- `calculate_divergence_NetShare()` → the "NetShare" section (`netshare` subcommand)
- `calculate_divergence_NetSSM()` → the "NetSSM" section (`netssm` subcommand)
- `calculate_divergence_NetDiffusion()` → the "NetDiffusion" section (`netdiffusion` subcommand)
- `calculate_divergence_TrafficLLM()` → not yet ported; the shared byte-decoding
  helpers it needs (`convert_hex_string()`, `extract_field_from_packet_bytes()`)
  are still present in `divergence_attacks.py` for when it is added.

Cleaned up (no hardcoded personal paths, parameterized via `argparse`) with
no functional change to the divergence math itself. The four target models
are attributed in full under
[`../../../synthetic-data-generation/`](../../../synthetic-data-generation/).

`divergence_attacks.py`'s "Feature extraction from pcaps" section
(`extract_packet_features()`, `extract_flow_features()`,
`ensure_features_csv()`, `extract_features_from_pcaps()`, the
`extract-features` subcommand) is adapted, trimmed to just the columns the
NetSSM/NetShare attacks read, from:
- `extract_packet_features_helper()` / `extract_flow_features_helper()` in
  a private internal research harness's `extract_feat_pcaps.py`
- `extract_packet()` / `extract_flow()` in a second private internal
  research harness's `extract_feats.py`

(neither harness is included in this repo)

## The attack

This is a *softer* attack than [`../network-identifiers`](../network-identifiers)
or [`../network-topology`](../network-topology): it doesn't ask whether any
single value or graph edge is exactly reproduced, but whether the overall
**shape of a sensitive field's distribution** leaks through. Fields like TTL
and TCP window size are commonly used for OS/device fingerprinting — if a
generator reproduces their real distribution too faithfully, it leaks
information about the population of devices/OSes in the training data even
without ever emitting an individually identifiable value.

### Scoring: normalized Wasserstein (Earth-Mover's) distance

For a chosen numeric field, compute the Wasserstein distance between the
real and generated value distributions, normalized by the combined value
range so scores are comparable across fields with different units:

```
norm_emd = wasserstein_distance(original, generated) / (max(original ∪ generated) - min(original ∪ generated))
```

**⚠️ Lower `norm_emd` = stronger attack result here** (0 = distributions are
identical, i.e. maximal leakage of the real shape) — this is the *opposite*
direction from the recall/precision/F1 metrics in
[`../network-identifiers`](../network-identifiers) and
[`../../data-extraction`](../../data-extraction), where higher means a
stronger attack. Don't compare the numbers across folders without relabeling.

## Per-model notes

| Model | Default field(s) | Data read |
|---|---|---|
| **NetShare** | `pkt_len` | Tabular flow CSVs (any numeric column works via `--column`). |
| **NetSSM** | `TTL`, `IP ID`, `IP Type of Service`, `TCP Window Size`, `TCP Data Offset`, `Packet Size`, `TCP Flags`, plus flow-level `Total Bytes` | Already-decoded per-packet/per-flow CSVs. |
| **TrafficLLM** | `Window Size` (configurable to any header field) | Hex payload, decoded to raw bytes then parsed via `extract_field_from_packet_bytes()`. |
| **NetDiffusion** | `ipv4_tl` (IPv4 total length; configurable) | nprint bit-columns, decoded via `bits_to_int()`. Works for any numeric field, unlike the identifier/topology attacks, since it doesn't need IP columns (which NetDiffusion's preprocessing drops — see the other two folders' NetDiffusion caveats). |

## Feature extraction from pcaps

NetShare and NetSSM expect pre-extracted feature CSVs (nprint files for
NetDiffusion still need the external `nprint` tool). If you only have raw
`.pcap` captures:

- Point `--original_glob`/`--generated_glob` (NetShare) or
  `--original_packets_glob`/`--generated_packets_glob`/`--*_flows_glob`
  (NetSSM) directly at `.pcap` files — the loaders extract features on
  demand and cache them as a sibling `<name>_packets.csv` /
  `<name>_flows.csv`, reusing the cache on later runs.
- Or extract up front with the `extract-features` subcommand (useful to
  warm the cache, or to inspect the extracted CSV before running an attack).

## Usage

All attacks and the pcap feature extractor live in one script,
`divergence_attacks.py`, as subcommands:

```bash
# Extract packet-level feature CSVs from pcaps up front (optional -- see above)
python divergence_attacks.py extract-features \
  --pcap_glob "/path/to/<dataset>/train/*.pcap" \
  --kind packet

# NetShare
python divergence_attacks.py netshare \
  --original_glob "/path/to/<dataset>/train/*/pre_processed_data/*.csv" \
  --generated_glob "/path/to/<dataset>/train/*/generated_data/sample_len-10/syn_dfs/chunk_id-0/epoch_id-*.csv" \
  --column pkt_len

# NetSSM
python divergence_attacks.py netssm \
  --original_packets_glob "/path/to/data_MIA_new/<dataset>/train/*packets.csv" \
  --generated_packets_glob "/path/to/inference/<dataset>_.../*/singleprompt_*_packets.csv" \
  --original_flows_glob "/path/to/data_MIA_new/<dataset>/train/*flows.csv" \
  --generated_flows_glob "/path/to/inference/<dataset>_.../*/singleprompt_*_flows.csv"

# NetDiffusion
python divergence_attacks.py netdiffusion \
  --original_nprint_glob "/path/to/data_MIA_test/all-labels/<dataset>_training/train_nprint/*.nprint" \
  --generated_nprint_glob "/path/to/inference/<dataset>/train/*/best_reconstruction.nprint" \
  --column_prefix ipv4_tl
```

TrafficLLM is not yet ported to this script (see Attribution above), so
there is no `trafficllm` subcommand.

Run `python divergence_attacks.py <model> --help` for the full flag list of
any subcommand.
