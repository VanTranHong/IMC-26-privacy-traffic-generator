# Network-Topology Reconstruction Attacks

This folder documents and provides cleaned, runnable implementations of the
network-topology attacks used to check whether NetSSM and NetShare
reproduce the real *structure* of who-talks-to-whom from their training
data — not just individual field values. (TrafficLLM and NetDiffusion are
documented for context but not currently implemented here — see Attribution
below.)

## Attribution

Adapted from `calculate_topology_divergence.py` in a private internal
research harness (not included in this repo). NetShare's and NetSSM's attacks (plus the shared extraction/comparison/
leakage-scoring logic) now live in a single script,
[`network_topology.py`](./network_topology.py):
- `load_and_process_data()`, `calculate_node_degrees()`,
  `get_undirected_edges()`/`get_undirected_edges_with_count()`, `compare_topo()`
  → shared `top_n_flows()` / `edges_from_flow_series()` / `calculate_node_degrees()` /
  `undirected_edge_counts()` / `compare_topologies()` / `print_report()`
- `calculate_flowsize_NetShare()` → `run_netshare()`
- `calculate_flowsize_NetSSM()` → `run_netssm()`

**TrafficLLM/NetDiffusion are not currently implemented in this folder.**
No TrafficLLM/NetDiffusion function existed in the original file — a
`trafficllm_topology.py` had been written here applying the same
methodology to TrafficLLM's own output format, but it was removed from this
folder (by a separate concurrent process, not this merge) before this file
was combined; `netdiffusion_topology.py` was never written. Re-add a
`run_trafficllm()`/`run_netdiffusion()` to `network_topology.py` following
the same pattern (see the important caveat for NetDiffusion below) if
they're needed again. The four target models are attributed in full under
[`../../../synthetic-data-generation/`](../../../synthetic-data-generation/).

## The attack

1. **Build a communication graph** from each model's top-N heaviest flows
   (by total bytes), reduced to `source -> dest` edges.
2. **Compare structurally**, via [`compare_topologies()`](./network_topology.py):
   - Node overlap (how many real hosts/entities appear in the synthetic graph)
   - Edge overlap (how many real connections appear in the synthetic graph)
   - Top-K hub agreement and top-K edge agreement
   - Degree-distribution similarity (normalized Wasserstein/Earth-Mover's
     distance between the two degree distributions — how similarly "shaped"
     the two graphs are, independent of exact node identity)

High overlap/agreement means the generator isn't just producing plausible
individual flows, it's reproducing the *real graph structure* of the
training data — a stronger and more damaging leak than a single reproduced
identifier.

## Per-model notes

| Model | Flow weight | Nodes |
|---|---|---|
| **NetShare** | `pkt_len` summed per 5-tuple | `srcip` / `dstip` |
| **NetSSM** | `Total Bytes` per flow CSV | `Src IP` / `Dst IP` |
| **TrafficLLM** *(not implemented here)* | hex-payload byte length per sample | `src` / `dst` (parsed from the header dict) |
| **NetDiffusion** *(not implemented here)* | packet length (`ipv4_tl` bit-column) | **port numbers, not hosts** — see caveat below |

### ⚠️ NetDiffusion caveat

NetDiffusion's own preprocessing deliberately drops IP columns before
training, **and** it's fine-tuned one model per individual flow rather than
across many flows at once — so there is no multi-host graph to reconstruct
in the first place, by construction. A `run_netdiffusion()` following this
file's pattern would need to build a port-number graph as a best-effort
structural analog instead, and be explicit in its docstring/report label
that this is *not* an equivalent host-topology attack.

## Usage

Both implemented attacks are subcommands of the one script,
[`network_topology.py`](./network_topology.py) (TrafficLLM/NetDiffusion have
no implementation here, see Attribution above):

```bash
# NetShare
python network_topology.py netshare \
  --original_glob "/path/to/<dataset>/train/*/pre_processed_data/*.csv" \
  --generated_glob "/path/to/<dataset>/train/*/generated_data/sample_len-10/syn_dfs/chunk_id-0/epoch_id-*.csv"

# NetSSM
python network_topology.py netssm \
  --original_glob "/path/to/data_MIA_new/<dataset>/train_features/*_flows.csv" \
  --generated_glob "/path/to/inference/<dataset>_.../*/singleprompt_*_flows.csv"
```

Each subcommand prints a structural comparison report (see
[`print_report()`](./network_topology.py) — the module docstring at the top
of that file spells out exactly how the topology is extracted per model,
how the two graphs are compared, and how each leakage metric is
calculated).
