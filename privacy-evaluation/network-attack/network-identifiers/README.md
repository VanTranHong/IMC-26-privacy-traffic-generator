# Network-Identifier Extraction Attacks

This folder documents and provides cleaned, runnable implementations of the
network-identifier extraction attacks used to check whether NetSSM,
TrafficLLM, and NetShare reproduce specific, real IP addresses or MAC
addresses from their training data. Scope is deliberately narrow: identifiers
only, not ports, protocols, or 5-tuple flows (NetDiffusion is the one
exception, noted below, since it has no IP/MAC to attack by construction).

## Attribution

Adapted from `calculate_network_attack.py` in a private internal research
harness (not included in this repo). All three model attacks below (plus the shared extraction/comparison/rate
logic) now live in a single script,
[`network_identifiers.py`](./network_identifiers.py):
- `calculate_cc`-style recall/precision math (bottom of the original file) → shared `compare_identifier_sets()` / `print_report()`
- `count_overlap_NetShare()` → `run_netshare()`
- `count_overlap()` → `run_netssm()` (plus a new `extract_identifiers_from_pcaps()` extraction path,
  adapted from `extract_packet_features_helper()`/`extract_flow_features_helper()` in
  a private internal research harness's `extract_feat_pcaps.py` (not included in this repo),
  for when the real/original side is still raw `.pcap` captures rather than
  already-decoded per-packet/flow CSVs)
- No TrafficLLM/NetDiffusion function existed in the original file — `run_trafficllm()` applies the
  same methodology to that model's own output format (see the module
  docstring for the exact parsing). The four target models are attributed in
  full under [`../../../synthetic-data-generation/`](../../../synthetic-data-generation/).
  (NetDiffusion is documented here for context but has no implementation in
  this folder — see the table below.)
- MAC-address attacks (new): NetSSM's via `run_netssm()` reading `MAC Src`/`MAC Dst`
  from flow-level CSVs (or `extract_identifiers_from_pcaps()`); TrafficLLM's via
  `parse_mac_addresses()`, adapted from the commented-out MAC block in
  `calculate_memorization_attack.py` (same shared attack harness), which decodes
  the destination/source MAC out of the raw Ethernet-frame hex payload that
  follows TrafficLLM's `{header dict}`.

## The attack

Generate many samples, extract one identifier field from each (an IP address
or a MAC address), and compare the **unique set** of extracted values
against the unique set of values in the real training data.

- **Recall** = |overlap| / |unique training values| — of everything in the
  training set, how much did the attacker ever see reproduced in generated
  output?
- **Precision** = |overlap| / |unique generated values| — of everything the
  generator ever produced, how much was a real training value rather than a
  novel one?

This is a *set-membership* attack — it doesn't care how often a value
recurs, just whether it was ever reproduced at all. Contrast with
[`../../data-extraction`](../../data-extraction) (which weights by
occurrence frequency) and [`../../MIA`](../../MIA) (which tests one known
candidate sample at a time rather than free-running generation).

## Per-model notes

| Model | Field(s) attacked | Why |
|---|---|---|
| **NetShare** | `srcip`, `dstip` | Explicit columns in NetShare's tabular output — the most direct case. No MAC: NetShare's preprocessing works at the flow/IP level and never carries an Ethernet frame. |
| **NetSSM** | `Src IP`, `Dst IP`, `MAC Src`/`MAC Dst` | IP read from already-decoded per-flow packet CSVs (byte sequences → structured fields); MAC is a separate flow-level field (`*_flows.csv`), or pulled straight from the Ethernet layer when starting from raw `.pcap`s. |
| **TrafficLLM** | `src`, `dst`, dst/src MAC | IP parsed out of the `{header dict} <hex payload>` generation format via `ast.literal_eval`; MAC is decoded separately from the first 12 bytes of the raw Ethernet frame in the hex payload itself (not in the header dict). |
| **NetDiffusion** *(not implemented here)* | `udp_sport`/`udp_dport` (or `tcp_*`) | The one exception to the identifiers-only scope above: NetDiffusion's own preprocessing **deliberately drops IP columns** before training (see `nprint_to_png.py` in the vendored NetDiffusion folder), so there's no IP or MAC to attack by construction — ports are the only field left to check. |

## Usage

All three implemented attacks are subcommands of the one script,
[`network_identifiers.py`](./network_identifiers.py) (NetDiffusion has no
implementation here, see the note above):

```bash
# NetShare
python network_identifiers.py netshare \
  --original_glob "/path/to/<dataset>/train/*/pre_processed_data/*.csv" \
  --generated_glob "/path/to/<dataset>/train/*/generated_data/sample_len-10/syn_dfs/chunk_id-0/epoch_id-*.csv" \
  --ip_columns_are_integers

# NetSSM (original side already decoded to per-packet CSVs)
python network_identifiers.py netssm \
  --original_glob "/path/to/data_MIA_new/<dataset>/train_features/*_packets.csv" \
  --generated_glob "/path/to/inference/<dataset>_.../*/singleprompt_*_packets.csv"

# NetSSM (original side is still raw .pcap captures -- extracted on the fly via scapy,
# including MAC Src/MAC Dst alongside IP)
python network_identifiers.py netssm \
  --original_pcap_glob "/path/to/data_MIA_new/<dataset>/train/*.pcap" \
  --generated_glob "/path/to/inference/<dataset>_.../*/singleprompt_*_packets.csv" \
  --generated_flow_glob "/path/to/inference/<dataset>_.../*/singleprompt_*_flows.csv"

# NetSSM (both sides already decoded, including the MAC-address comparison via flow-level CSVs)
python network_identifiers.py netssm \
  --original_glob "/path/to/data_MIA_new/<dataset>/train_features/*_packets.csv" \
  --generated_glob "/path/to/inference/<dataset>_.../*/singleprompt_*_packets.csv" \
  --original_flow_glob "/path/to/data_MIA_new/<dataset>/train_features/*_flows.csv" \
  --generated_flow_glob "/path/to/inference/<dataset>_.../*/singleprompt_*_flows.csv"

# TrafficLLM
python network_identifiers.py trafficllm \
  --original_jsonl /path/to/output_TrafficLLM/VNAT_final_train/VNAT_final_train_preprocessed_generation_packet_train.json \
  --generated_glob "/path/to/inference_new/VNAT_1epochs/*/*_maxlen_300_temp_0_9.txt"
```

Each subcommand prints a recall/precision report per field (see
[`print_report()`](./network_identifiers.py) — the module docstring at the
top of that file spells out exactly how identifiers are extracted per model,
how the extracted sets are compared, and how recall/precision are computed).
