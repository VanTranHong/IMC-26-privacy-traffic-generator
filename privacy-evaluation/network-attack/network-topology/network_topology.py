"""Network-Topology Reconstruction Attacks (combined single-file implementation).

Adapted from `calculate_topology_divergence.py` in a private internal
research harness (not included in this repo; cleaned up, parameterized,
and merged here). See ./README.md for the
per-model attribution and background. This file merges topology_common.py,
netshare_topology.py, and netssm_topology.py into one, organized as:

    1. EXTRACTION     -- per-model code that reduces a model's raw output
                          into a two-column (source, dest) edge list built
                          from its top-N heaviest flows.
    2. COMPARISON +
       LEAKAGE SCORE   -- `compare_topologies()`, the one function both
                          models funnel into. It builds both communication
                          graphs from those edge lists and computes every
                          topology-leakage metric from them.
    3. CLI             -- one `python network_topology.py <model> ...`
                          entry point per model (netshare / netssm), each
                          ending in the same `print_report()`.

NOTE: trafficllm_topology.py (and the never-implemented
netdiffusion_topology.py) are not included here -- the TrafficLLM script was
removed from this folder by another concurrent session before this merge;
re-add a `run_trafficllm()` following the same pattern once it's back.

## What this attack measures

Real network traffic has *structure*: which hosts talk to which, and how
often. If a generative model has memorized (rather than generalized) its
training flows, the **communication graph** built from its synthetic output
-- which nodes exist, which pairs of nodes are connected, which nodes are
hubs -- will closely mirror the real training graph. This is a stronger and
more damaging leak than a single reproduced identifier (../network-identifiers):
it means the generator reproduced *relationships*, not just values.

## How the topology is EXTRACTED (per model)

Both models reduce to the same three-step recipe, via the shared
`top_n_flows()` + `edges_from_flow_series()` helpers:
  1. Group the raw flow/packet rows by their node-pair columns.
  2. Sum a weight column (bytes) per group and keep the top `--top_n`
     heaviest flows.
  3. Turn that into a `source_ip`/`dest_ip` edge-list DataFrame, dropping
     self-loops (`src == dst`).

- **NetShare**: grouped by the full 5-tuple (`srcip`/`dstip`/`srcport`/
  `dstport`/`proto`) from its tabular CSV output, weighted by summed
  `pkt_len`. Node columns are `srcip`/`dstip`.
- **NetSSM**: grouped by `Src IP`/`Dst IP` from its per-flow CSVs (already
  aggregated to flow level, one row per flow), weighted by `Total Bytes`.

## How the two graphs are COMPARED, and how LEAKAGE is calculated

`compare_topologies()` is the single shared step every model's edge list
funnels into. From each edge list it computes:
  - **Undirected node degree** per host (`calculate_node_degrees()`): how
    many distinct edges touch that node.
  - **Undirected edge counts** (`undirected_edge_counts()`): each edge
    canonicalized as `tuple(sorted((src, dst)))` so `A->B` and `B->A` count
    as the same edge.

From those, the leakage metrics:
  - **Node overlap** = |real nodes ∩ generated nodes|, and
    **node_overlap_rate** = that / |real nodes| -- of every real host, how
    many appeared anywhere in the generated graph?
  - **Edge overlap** = |real edges ∩ generated edges| (same denominator
    pattern for **edge_overlap_rate**) -- of every real connection, how many
    were reproduced?
  - **Top-K hub/edge agreement**: intersect the top `--top_k`
    highest-degree nodes (resp. highest-count edges) on each side --
    do the two graphs agree on who the *hubs* are, not just on any overlap?
  - **Degree-distribution EMD (normalized)**: the Wasserstein/Earth-Mover's
    distance between the real and generated degree-value distributions,
    divided by the full range spanned by both (`0` if every node/edge has
    degree 0 or the two graphs are identical in shape). This captures
    similarity of overall graph *shape* independent of exact node identity
    -- two graphs can score 0 on node/edge overlap (completely different
    hosts) yet still have a similar degree distribution (same shape, e.g.
    both star-shaped or both flat).

High overlap/agreement (and low degree-EMD) means the generator isn't just
producing plausible individual flows -- it's reproducing the real graph
structure of the training data.
"""

import argparse
import glob
import os
from collections import Counter
from typing import Dict, List, Tuple

import pandas as pd
from scipy.stats import wasserstein_distance

# =============================================================================
# 1. EXTRACTION -- shared helpers + per-model edge-list builders
# =============================================================================


def top_n_flows(df: pd.DataFrame, group_cols: List[str], value_col: str, n: int = 100) -> pd.Series:
    """Sum `value_col` (e.g. total bytes) per flow (grouped by `group_cols`,
    e.g. a 5-tuple or `(srcip, dstip)`), and keep the heaviest `n` flows."""
    return df.groupby(group_cols)[value_col].sum().sort_values(ascending=False).head(n)


def edges_from_flow_series(flow_series: pd.Series, src_index: int = 0, dst_index: int = 1) -> pd.DataFrame:
    """A top-flows Series (as returned by `top_n_flows`, indexed by a tuple or
    a `"src-dst"` string) -> a `source_ip`/`dest_ip` edge-list DataFrame,
    dropping self-loops."""
    sources, dests = [], []
    for key in flow_series.index:
        if isinstance(key, str):
            src, dst = key.split("-", 1)
        else:
            src, dst = key[src_index], key[dst_index]
        if src == dst:
            continue
        sources.append(src)
        dests.append(dst)
    return pd.DataFrame({"source_ip": sources, "dest_ip": dests})


def load_concat_csv(csv_glob: str, skip_empty: bool = False, **read_csv_kwargs) -> pd.DataFrame:
    """Read every CSV matching `csv_glob` and concatenate into one dataframe,
    skipping unreadable (and optionally empty) files. Shared extraction step
    for NetShare and NetSSM, whose flow data lives in tabular columns."""
    frames = []
    for fn in glob.glob(csv_glob):
        if skip_empty and (not os.path.exists(fn) or os.path.getsize(fn) == 0):
            continue
        try:
            frames.append(pd.read_csv(fn, **read_csv_kwargs))
        except Exception as e:
            print(f"[skip] {fn}: {e}")
    if not frames:
        raise FileNotFoundError(f"No readable CSVs matched: {csv_glob}")
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# 2. COMPARISON + LEAKAGE SCORE -- the shared graph-comparison step every
#    model's edge list funnels into after extraction
# =============================================================================


def calculate_node_degrees(edges: pd.DataFrame) -> Dict[str, int]:
    """Undirected degree of every node appearing in the edge list, sorted
    descending."""
    nodes = set(edges["source_ip"]) | set(edges["dest_ip"])
    degrees = {
        node: int(((edges["source_ip"] == node) | (edges["dest_ip"] == node)).sum())
        for node in nodes
    }
    return dict(sorted(degrees.items(), key=lambda kv: kv[1], reverse=True))


def undirected_edge_counts(edges: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """Each edge canonicalized as `tuple(sorted((src, dst)))` -- so `A->B`
    and `B->A` are counted as the same undirected edge -- sorted by count
    descending."""
    counter = Counter(tuple(sorted((s, d))) for s, d in zip(edges["source_ip"], edges["dest_ip"]))
    return dict(sorted(counter.items(), key=lambda kv: kv[1], reverse=True))


def compare_topologies(edges_original: pd.DataFrame, edges_generated: pd.DataFrame, top_k: int = 10) -> Dict:
    """Full structural comparison of two communication graphs -- this is
    where every topology-LEAKAGE metric described in the module docstring
    is actually calculated. Returns a report dict; pass it to
    `print_report()` for a human-readable summary.

    COMPARISON: both edge lists are reduced to a node-degree dict and an
    undirected-edge-count dict (see the two functions above), then
    intersected as sets.

    RATES / SCORES:
      node_overlap_rate = |common_nodes| / |nodes_original|
      edge_overlap_rate = |common_edges| / |edges_original|
      top_k_node_overlap = |top-K-degree nodes (original) ∩ top-K (generated)|
      top_k_edge_overlap = |top-K-count edges (original) ∩ top-K (generated)|
      degree_distribution_emd_normalized =
          wasserstein_distance(degrees_original, degrees_generated)
          / (max(all degrees) - min(all degrees))
    """
    degrees_original = calculate_node_degrees(edges_original)
    degrees_generated = calculate_node_degrees(edges_generated)

    edge_counts_original = undirected_edge_counts(edges_original)
    edge_counts_generated = undirected_edge_counts(edges_generated)

    nodes_original, nodes_generated = set(degrees_original), set(degrees_generated)
    edges_original_set, edges_generated_set = set(edge_counts_original), set(edge_counts_generated)

    common_nodes = nodes_original & nodes_generated
    common_edges = edges_original_set & edges_generated_set

    topk_nodes_original = set(list(degrees_original)[:top_k])
    topk_nodes_generated = set(list(degrees_generated)[:top_k])
    topk_edges_original = set(list(edge_counts_original)[:top_k])
    topk_edges_generated = set(list(edge_counts_generated)[:top_k])

    degree_list_original = list(degrees_original.values())
    degree_list_generated = list(degrees_generated.values())
    all_degrees = degree_list_original + degree_list_generated
    degree_range = (max(all_degrees) - min(all_degrees)) if all_degrees else 0
    degree_emd = wasserstein_distance(degree_list_original, degree_list_generated) if all_degrees else 0.0
    degree_emd_norm = (degree_emd / degree_range) if degree_range else 0.0

    return {
        "num_nodes_original": len(nodes_original),
        "num_nodes_generated": len(nodes_generated),
        "node_overlap": len(common_nodes),
        "node_overlap_rate": len(common_nodes) / len(nodes_original) if nodes_original else 0.0,
        "num_edges_original": len(edges_original_set),
        "num_edges_generated": len(edges_generated_set),
        "edge_overlap": len(common_edges),
        "edge_overlap_rate": len(common_edges) / len(edges_original_set) if edges_original_set else 0.0,
        "top_k": top_k,
        "top_k_node_overlap": len(topk_nodes_original & topk_nodes_generated),
        "top_k_edge_overlap": len(topk_edges_original & topk_edges_generated),
        "degree_distribution_emd_normalized": degree_emd_norm,
        "degrees_original": degrees_original,
        "degrees_generated": degrees_generated,
    }


def print_report(title: str, report: Dict) -> None:
    print(f"--- Network-topology comparison report: {title} ---")
    print(f"  Nodes:  original={report['num_nodes_original']}, generated={report['num_nodes_generated']}, "
          f"overlap={report['node_overlap']} ({report['node_overlap_rate']:.1%})")
    print(f"  Edges:  original={report['num_edges_original']}, generated={report['num_edges_generated']}, "
          f"overlap={report['edge_overlap']} ({report['edge_overlap_rate']:.1%})")
    k = report["top_k"]
    print(f"  Top-{k} hub agreement:  {report['top_k_node_overlap']}/{k}")
    print(f"  Top-{k} edge agreement: {report['top_k_edge_overlap']}/{k}")
    print(f"  Degree-distribution EMD (normalized, lower = more similar shape): {report['degree_distribution_emd_normalized']:.4f}")


# =============================================================================
# NetShare
# =============================================================================
# EXTRACTION: NetShare's CSV output already has explicit srcip/dstip/srcport/
# dstport/proto columns and a per-packet pkt_len -- group by the full
# 5-tuple, sum pkt_len, keep the top-N heaviest flows, reduce to srcip/dstip
# edges.
# COMPARISON + LEAKAGE: delegated to compare_topologies() / print_report()
# above.


def run_netshare(args: argparse.Namespace) -> None:
    df_original = load_concat_csv(args.original_glob, encoding="utf-8", encoding_errors="ignore")
    df_generated = load_concat_csv(args.generated_glob, encoding="utf-8", encoding_errors="ignore")

    tuple_cols = ["srcip", "dstip", "srcport", "dstport", "proto"]
    flows_original = top_n_flows(df_original, tuple_cols, "pkt_len", args.top_n)
    flows_generated = top_n_flows(df_generated, tuple_cols, "pkt_len", args.top_n)

    edges_original = edges_from_flow_series(flows_original, src_index=0, dst_index=1)
    edges_generated = edges_from_flow_series(flows_generated, src_index=0, dst_index=1)

    report = compare_topologies(edges_original, edges_generated, top_k=args.top_k)
    print_report("NetShare", report)


# =============================================================================
# NetSSM
# =============================================================================
# EXTRACTION: reads already-decoded, already flow-aggregated per-flow CSVs
# (Src IP/Dst IP/Total Bytes columns, one row per flow) -- group by
# Src IP/Dst IP, sum Total Bytes, keep the top-N heaviest flows (see
# ../../MIA/netssm_blackbox_mia.py for reading NetSSM's raw generated byte
# sequences instead).
# COMPARISON + LEAKAGE: same shared functions as NetShare.


def run_netssm(args: argparse.Namespace) -> None:
    df_original = load_concat_csv(args.original_glob, skip_empty=True, encoding="utf-8", encoding_errors="ignore")
    df_generated = load_concat_csv(args.generated_glob, skip_empty=True, encoding="utf-8", encoding_errors="ignore")

    flow_cols = ["Src IP", "Dst IP"]
    flows_original = top_n_flows(df_original, flow_cols, "Total Bytes", args.top_n)
    flows_generated = top_n_flows(df_generated, flow_cols, "Total Bytes", args.top_n)

    edges_original = edges_from_flow_series(flows_original, src_index=0, dst_index=1)
    edges_generated = edges_from_flow_series(flows_generated, src_index=0, dst_index=1)

    report = compare_topologies(edges_original, edges_generated, top_k=args.top_k)
    print_report("NetSSM", report)


# =============================================================================
# 3. CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Network-topology reconstruction attacks (NetShare / NetSSM) in one script. "
        "See the module docstring for how the topology is extracted, compared, and how leakage is scored."
    )
    subparsers = parser.add_subparsers(dest="model", required=True)

    p_netshare = subparsers.add_parser("netshare", help="Run the attack against NetShare's tabular CSV output.")
    p_netshare.add_argument("--original_glob", required=True, help="Glob for real flow CSVs (pre_processed_data/*.csv).")
    p_netshare.add_argument("--generated_glob", required=True, help="Glob for synthetic flow CSVs (generated_data/.../epoch_id-*.csv).")
    p_netshare.add_argument("--top_n", type=int, default=100, help="How many heaviest flows to build the graph from.")
    p_netshare.add_argument("--top_k", type=int, default=10, help="Top-K hubs/edges to report agreement over.")
    p_netshare.set_defaults(func=run_netshare)

    p_netssm = subparsers.add_parser("netssm", help="Run the attack against NetSSM's flow-aggregated CSVs.")
    p_netssm.add_argument("--original_glob", required=True, help="Glob for real per-flow CSVs, e.g. '.../train_features/*_flows.csv'.")
    p_netssm.add_argument("--generated_glob", required=True, help="Glob for generated per-flow CSVs, e.g. '.../inference/.../*_flows.csv'.")
    p_netssm.add_argument("--top_n", type=int, default=100, help="How many heaviest flows to build the graph from.")
    p_netssm.add_argument("--top_k", type=int, default=10, help="Top-K hubs/edges to report agreement over.")
    p_netssm.set_defaults(func=run_netssm)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
