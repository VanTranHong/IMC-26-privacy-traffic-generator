#!/usr/bin/env python3
"""Utility evaluation for synthetic network-traffic generators.

Checks whether NetShare/NetSSM-style synthetic traffic is still *useful*:
does it statistically resemble the real data it was trained on (fidelity),
and can a model trained on it substitute for real training data on a
downstream classification task (utility)? This is the counterpart to
../privacy-evaluation: high fidelity / high downstream utility is desirable
here, whereas ../privacy-evaluation/network-attack looks for the same kind
of distributional closeness as evidence of *privacy leakage*. A generator
can score well on one axis and poorly on the other -- they measure different
things and should not be combined into a single number.



## Fidelity: categorical-field distribution divergence

For a set of categorical fields (e.g. source/destination IP, source/
destination port, protocol), compares the real and generated value
distributions with two divergence measures:

  - **Jensen-Shannon divergence** (`scipy.spatial.distance.jensenshannon`):
    a bounded, symmetric divergence between two probability distributions.
  - **Normalized Earth Mover's Distance**: a positional/cumulative-mass
    distance (`sum(|cumsum(p) - cumsum(q)|) / (n - 1)`), computed over the
    two value-count distributions *after* aligning them to a shared,
    lexicographically-sorted category index. Unlike
    ../privacy-evaluation/network-attack/sensitive-network-properties (which
    uses `scipy.stats.wasserstein_distance` on numeric values, where
    "distance between categories" is well-defined), this treats the sorted
    category *rank* as if it were a numeric axis -- reasonable for a small
    set of IPs/ports where you mainly care that the same items get similar
    mass, but it does depend on the (arbitrary) sort order, unlike JSD. Kept
    faithful to the original for continuity with prior results.

Both are `>= 0`, and **lower is better here** (0 = identical distributions,
i.e. maximal fidelity) -- the opposite direction from the same-shaped metric
used as an attack score in
../privacy-evaluation/network-attack/sensitive-network-properties.

## Downstream task: TRTR vs. TSTR classification accuracy

Trains a classifier on **real** training data and evaluates it on real test
data (TRTR: Train-Real-Test-Real, an upper-bound reference), then trains an
identical classifier on **synthetic** data and evaluates it on the same real
test data (TSTR: Train-Synthetic-Test-Real). TSTR accuracy close to TRTR
accuracy means the synthetic data preserves enough task-relevant signal to
substitute for real training data; a large gap means it doesn't. Two input
feature representations are supported (`--feature_mode`):

  - `nprint` (the representation actually exercised in the original
    script): the first few packets' nprint bit-columns per flow, flattened
    into one row.
  - `flow_stats`: per-flow summary statistics (packet/byte counts, IAT
    stats, rates) -- present in the original as unused helper functions,
    exposed here as a lighter-weight alternative that doesn't require an
    nprint export.

## Usage

    # Fidelity: NetShare (tabular flow CSVs, columns srcip/dstip/srcport/dstport/proto)
    python utility_evaluation.py fidelity-netshare \\
        --original_glob "/path/to/<dataset>/train/*_merged/pre_processed_data/*.csv" \\
        --generated_glob "/path/to/<dataset>/train/*_merged/generated_data/*/*.csv" \\
        --dataset <dataset>

    # Fidelity: NetSSM (per-packet CSVs, columns Src/Dst IP/Port, Protocol)
    python utility_evaluation.py fidelity-netssm \\
        --original_packets_glob "/path/to/data_MIA_new/single_<dataset>/new_data/train/*_packets.csv" \\
        --generated_packets_glob "/path/to/inference/<dataset>_singleflow_.../*/singleprompt_*_packets.csv" \\
        --dataset <dataset>

    # Downstream task: TRTR vs. TSTR classification accuracy
    python utility_evaluation.py downstream-task \\
        --train_real_folder "/path/to/data_MIA_new/single_<dataset>/new_data/train/" \\
        --test_real_folder "/path/to/data_MIA_new/single_<dataset>/new_data/test/" \\
        --synthetic_glob "/path/to/inference/<dataset>_singleflow_.../*/singleprompt_*.nprint" \\
        --feature_mode nprint --classifier xgboost --model_name NetSSM --dataset <dataset>

Each subcommand's flags are also documented under `--help`, e.g.
`python utility_evaluation.py downstream-task --help`.
"""

import argparse
import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import classification_report


# =============================================================================
# Fidelity: shared distribution-divergence core
# =============================================================================

def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two aligned probability vectors."""
    return float(jensenshannon(np.asarray(p, dtype=float), np.asarray(q, dtype=float)))


def normalized_earth_movers_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Cumulative-mass distance between two aligned probability vectors,
    normalized by the maximum possible distance (n - 1). See module
    docstring for the category-ordering caveat."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    emd = np.sum(np.abs(np.cumsum(p) - np.cumsum(q)))
    n = len(p)
    max_emd = n - 1
    return float(emd / max_emd) if max_emd > 0 else 0.0


def compare_categorical_distributions(original_df: pd.DataFrame, generated_df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
    """For each column, aligns the real/generated value-count distributions
    to a shared category index (missing categories filled with 0 mass) and
    scores JSD + normalized EMD between them."""
    results = {}
    for column in columns:
        o_counts = original_df[column].value_counts(normalize=True)
        g_counts = generated_df[column].value_counts(normalize=True)
        o_counts, g_counts = o_counts.align(g_counts, fill_value=0.0)
        results[column] = {
            "jsd": jensen_shannon_divergence(o_counts.values, g_counts.values),
            "normalized_emd": normalized_earth_movers_distance(o_counts.values, g_counts.values),
        }
    return results


def print_fidelity_report(model_name: str, dataset: str, metrics: Dict[str, Dict[str, float]]) -> None:
    print(f"--- Fidelity report: {model_name} / {dataset} ---")
    for column, m in metrics.items():
        print(f"  {column:14s}  JSD: {m['jsd']:.4f}   Normalized EMD: {m['normalized_emd']:.4f}  (lower = higher fidelity)")
    overall = float(np.mean([m["normalized_emd"] for m in metrics.values()]))
    print(f"  Overall average normalized EMD: {overall:.4f}")


# =============================================================================
# Fidelity: NetShare
#
# Adapted from `fidelity_NetShare()`. Compares real vs. synthetic tabular
# flow CSVs on the categorical fields srcip/dstip/srcport/dstport/proto.
# =============================================================================

NETSHARE_FIDELITY_COLUMNS = ["srcip", "dstip", "srcport", "dstport", "proto"]


def load_csvs(csv_glob: str) -> pd.DataFrame:
    df_lst = []
    for fn in glob.glob(csv_glob):
        try:
            with open(fn, "r", encoding="utf-8", errors="ignore") as f:
                df_lst.append(pd.read_csv(f))
        except Exception as e:
            print(f"[skip] {fn}: {e}")
    if not df_lst:
        raise ValueError(f"No usable CSVs matched {csv_glob!r}")
    return pd.concat(df_lst, ignore_index=True)


def run_fidelity_netshare(args: argparse.Namespace) -> None:
    original_df = load_csvs(args.original_glob)
    generated_df = load_csvs(args.generated_glob)
    metrics = compare_categorical_distributions(original_df, generated_df, args.columns)
    print_fidelity_report("NetShare", args.dataset, metrics)


def _add_fidelity_netshare_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("fidelity-netshare", help="Categorical-field distribution fidelity for NetShare.")
    parser.add_argument("--original_glob", required=True, help="Glob for real flow CSVs (pre_processed_data/*.csv).")
    parser.add_argument("--generated_glob", required=True, help="Glob for synthetic flow CSVs (generated_data/*/*.csv).")
    parser.add_argument("--columns", nargs="+", default=NETSHARE_FIDELITY_COLUMNS, help=f"Categorical columns to compare. Default: {NETSHARE_FIDELITY_COLUMNS}")
    parser.add_argument("--dataset", default="(unlabeled)", help="Dataset name, used only for the report header.")
    parser.set_defaults(func=run_fidelity_netshare)


# =============================================================================
# Fidelity: NetSSM
#
# Adapted from `fidelity_NetSSM()`. Compares real vs. synthetic per-packet
# CSVs on the categorical fields Src/Dst IP, Src/Dst Port, Protocol, using
# the first `n_head` packets of each flow.
# =============================================================================

NETSSM_FIDELITY_COLUMNS = ["Src IP", "Dst IP", "Src Port", "Dst Port", "Protocol"]


def read_csv_safe(filepath: str) -> "pd.DataFrame | None":
    """Reads a CSV, returning None (with a printed warning) instead of
    raising on an empty file, an unparsable file, or a file with no rows."""
    if os.path.getsize(filepath) == 0:
        print(f"[skip] {filepath}: empty file")
        return None
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            df = pd.read_csv(f)
    except pd.errors.EmptyDataError:
        print(f"[skip] {filepath}: no columns to parse")
        return None
    except Exception as e:
        print(f"[skip] {filepath}: {e}")
        return None
    if df.empty or len(df.columns) == 0:
        print(f"[skip] {filepath}: no data")
        return None
    return df


def load_packet_csvs(csv_glob: str, n_head: int = 200) -> pd.DataFrame:
    df_lst = []
    for fn in glob.glob(csv_glob):
        df = read_csv_safe(fn)
        if df is not None:
            df_lst.append(df.iloc[:n_head])
    if not df_lst:
        raise ValueError(f"No usable CSVs matched {csv_glob!r}")
    return pd.concat(df_lst, ignore_index=True)


def run_fidelity_netssm(args: argparse.Namespace) -> None:
    original_df = load_packet_csvs(args.original_packets_glob, args.n_head)
    generated_df = load_packet_csvs(args.generated_packets_glob, args.n_head)
    metrics = compare_categorical_distributions(original_df, generated_df, args.columns)
    print_fidelity_report("NetSSM", args.dataset, metrics)


def _add_fidelity_netssm_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("fidelity-netssm", help="Categorical-field distribution fidelity for NetSSM.")
    parser.add_argument("--original_packets_glob", required=True, help="Glob for real per-packet CSVs, e.g. '.../train/*_packets.csv'.")
    parser.add_argument("--generated_packets_glob", required=True, help="Glob for generated per-packet CSVs.")
    parser.add_argument("--columns", nargs="+", default=NETSSM_FIDELITY_COLUMNS, help=f"Categorical columns to compare. Default: {NETSSM_FIDELITY_COLUMNS}")
    parser.add_argument("--n_head", type=int, default=200, help="Only use the first N packets of each flow (matches the original script's behavior).")
    parser.add_argument("--dataset", default="(unlabeled)", help="Dataset name, used only for the report header.")
    parser.set_defaults(func=run_fidelity_netssm)


# =============================================================================
# Downstream task: TRTR vs. TSTR classification accuracy
#
# Adapted from downstream_task_performance.py. Two feature representations
# are supported (--feature_mode): the nprint-flattening path that was
# actually exercised in the original script's live (uncommented) code, and
# the flow-level-statistics path that existed there only as unused helper
# functions (prepare_real_stats/prepare_synthetic_stats).
# =============================================================================

NPRINT_HEAD_ROWS = 3
FLOW_STAT_COLUMNS = [
    "Total Packets", "Total Bytes", "Fwd Packets", "Bwd Packets", "Fwd Bytes", "Bwd Bytes",
    "Flow Duration", "Avg Packet Size", "Std Packet Size", "Flow IAT Mean", "Flow IAT Std",
    "Flow IAT Max", "Flow IAT Min", "Packet Rate", "Byte Rate",
]


def _flatten_nprint(df: pd.DataFrame, n_head: int = NPRINT_HEAD_ROWS) -> pd.DataFrame:
    """First `n_head` packets of one flow's nprint bit-columns -> a single
    flattened row (`{original_column}_{packet_index}` per cell), dropping
    the identifying `src_ip` and any `*prt*` (port) columns."""
    df = df.iloc[:n_head]
    drop_columns = ["src_ip"] + [c for c in df.columns if "prt" in c]
    df = df.drop(columns=drop_columns, errors="ignore")
    return pd.DataFrame([df.values.flatten()], columns=[f"{col}_{i}" for i in range(len(df)) for col in df.columns])


def load_nprint_features_labeled(folder: str) -> pd.DataFrame:
    """Loads one flattened-nprint row per flow for a labeled real split.
    `folder` must contain `labels.csv` (columns `File`, `Label`) and, for
    each `File` entry, a same-named `.nprint` file (`.pcap` extension
    swapped for `.nprint`)."""
    df_labels = pd.read_csv(os.path.join(folder, "labels.csv"))
    rows, labels = [], []
    for _, row in df_labels.iterrows():
        nprint_fn = os.path.join(folder, row["File"].replace(".pcap", ".nprint"))
        try:
            rows.append(_flatten_nprint(pd.read_csv(nprint_fn)))
            labels.append(row["Label"])
        except Exception as e:
            print(f"[skip] {nprint_fn}: {e}")
    df_all = pd.concat(rows, ignore_index=True)
    df_all["label"] = labels
    return df_all


def load_nprint_features_glob(nprint_glob: str) -> pd.DataFrame:
    """Loads one flattened-nprint row per matched file for synthetic data,
    where there's no labels.csv -- the label is taken from the parent
    directory name's suffix after the last underscore (matches the
    `<dataset>_<label>/` folder-naming convention used by the generators)."""
    rows, labels = [], []
    for fn in glob.glob(nprint_glob):
        try:
            rows.append(_flatten_nprint(pd.read_csv(fn)))
            labels.append(fn.split("/")[-2].split("_")[-1])
        except Exception as e:
            print(f"[skip] {fn}: {e}")
    df_all = pd.concat(rows, ignore_index=True)
    df_all["label"] = labels
    return df_all


def load_flow_stats_labeled(folder: str) -> pd.DataFrame:
    """Loads one flow-statistics row per flow for a labeled real split, from
    `labels.csv` (columns `File`, `Label`) plus each `File`'s `_flows.csv`
    (`.pcap` extension swapped for `_flows.csv`)."""
    df_labels = pd.read_csv(os.path.join(folder, "labels.csv"))
    rows, labels = [], []
    for _, row in df_labels.iterrows():
        flow_fn = os.path.join(folder, row["File"].replace(".pcap", "_flows.csv"))
        try:
            rows.append(pd.read_csv(flow_fn)[FLOW_STAT_COLUMNS].iloc[:1])
            labels.append(row["Label"])
        except Exception as e:
            print(f"[skip] {flow_fn}: {e}")
    df_all = pd.concat(rows, ignore_index=True)
    df_all["label"] = labels
    return df_all


def load_flow_stats_glob(flows_glob: str) -> pd.DataFrame:
    """Loads one flow-statistics row per matched `_flows.csv` for synthetic
    data; label taken the same way as `load_nprint_features_glob()`."""
    rows, labels = [], []
    for fn in glob.glob(flows_glob):
        try:
            rows.append(pd.read_csv(fn)[FLOW_STAT_COLUMNS].iloc[:1])
            labels.append(fn.split("/")[-2].split("_")[-1])
        except Exception as e:
            print(f"[skip] {fn}: {e}")
    df_all = pd.concat(rows, ignore_index=True)
    df_all["label"] = labels
    return df_all


def make_classifier(name: str):
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(eval_metric="mlogloss")
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, random_state=42)
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, loss_function="MultiClass", verbose=False, random_seed=42)
    raise ValueError(f"Unknown classifier {name!r}; choose xgboost, random_forest, or catboost.")


def evaluate_downstream_task(train_real_df: pd.DataFrame, test_real_df: pd.DataFrame, train_synthetic_df: pd.DataFrame, classifier_name: str = "xgboost") -> Dict[str, Dict[str, float]]:
    """TRTR vs. TSTR accuracy/macro-F1 for a classification downstream
    task. See module docstring for interpretation (closer TSTR-to-TRTR =
    more downstream utility preserved)."""
    labels_missing = [l for l in train_synthetic_df["label"].unique() if l not in test_real_df["label"].unique()]
    if labels_missing:
        print(f"[note] dropping synthetic rows with labels absent from real test data: {labels_missing}")
        train_synthetic_df = train_synthetic_df[~train_synthetic_df["label"].isin(labels_missing)]

    labels_all = sorted(set(test_real_df["label"]) | set(train_synthetic_df["label"]))
    label_to_idx = {label: idx for idx, label in enumerate(labels_all)}

    def encode(df: pd.DataFrame):
        df = df.copy()
        df["label"] = df["label"].map(label_to_idx)
        return df.drop(columns=["label"]), df["label"]

    X_train_real, y_train_real = encode(train_real_df)
    X_test_real, y_test_real = encode(test_real_df)
    X_train_synthetic, y_train_synthetic = encode(train_synthetic_df)

    def fit_eval(X_train, y_train):
        clf = make_classifier(classifier_name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test_real)
        report = classification_report(y_test_real, y_pred, output_dict=True, zero_division=0)
        return {"accuracy": report["accuracy"], "macro_f1": report["macro avg"]["f1-score"]}

    return {
        "trtr": fit_eval(X_train_real, y_train_real),
        "tstr": fit_eval(X_train_synthetic, y_train_synthetic),
    }


def print_downstream_report(model_name: str, dataset: str, results: Dict[str, Dict[str, float]]) -> None:
    print(f"--- Downstream-task report: {model_name} / {dataset} ---")
    print(f"  Train-real  / Test-real  (TRTR, upper bound):    accuracy={results['trtr']['accuracy']:.4f}  macro-F1={results['trtr']['macro_f1']:.4f}")
    print(f"  Train-synth / Test-real  (TSTR, utility score):  accuracy={results['tstr']['accuracy']:.4f}  macro-F1={results['tstr']['macro_f1']:.4f}")
    gap = results["trtr"]["accuracy"] - results["tstr"]["accuracy"]
    print(f"  Utility gap (TRTR - TSTR accuracy): {gap:+.4f}  (closer to 0 = synthetic data preserves more task utility)")


def run_downstream_task(args: argparse.Namespace) -> None:
    if args.feature_mode == "nprint":
        train_real_df = load_nprint_features_labeled(args.train_real_folder)
        test_real_df = load_nprint_features_labeled(args.test_real_folder)
        synthetic_df = load_nprint_features_glob(args.synthetic_glob)
    else:
        train_real_df = load_flow_stats_labeled(args.train_real_folder)
        test_real_df = load_flow_stats_labeled(args.test_real_folder)
        synthetic_df = load_flow_stats_glob(args.synthetic_glob)

    print("Train-real labels:  ", train_real_df["label"].value_counts().to_dict())
    print("Test-real labels:   ", test_real_df["label"].value_counts().to_dict())
    print("Synthetic labels:   ", synthetic_df["label"].value_counts().to_dict())

    results = evaluate_downstream_task(train_real_df, test_real_df, synthetic_df, classifier_name=args.classifier)
    print_downstream_report(args.model_name, args.dataset, results)


def _add_downstream_task_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("downstream-task", help="TRTR vs. TSTR classification accuracy/F1 (downstream-task utility).")
    parser.add_argument("--train_real_folder", required=True, help="Folder with real training data: labels.csv (File, Label) plus per-file .nprint or _flows.csv.")
    parser.add_argument("--test_real_folder", required=True, help="Folder with real test data, same layout as --train_real_folder.")
    parser.add_argument("--synthetic_glob", required=True, help="Glob for synthetic .nprint or _flows.csv files (label read from the parent directory name).")
    parser.add_argument("--feature_mode", choices=["nprint", "flow_stats"], default="nprint", help="Feature representation to classify on. See module docstring.")
    parser.add_argument("--classifier", choices=["xgboost", "random_forest", "catboost"], default="xgboost", help="Classifier to train for both TRTR and TSTR.")
    parser.add_argument("--model_name", default="(unlabeled)", help="Generator name, used only for the report header.")
    parser.add_argument("--dataset", default="(unlabeled)", help="Dataset name, used only for the report header.")
    parser.set_defaults(func=run_downstream_task)


# =============================================================================
# CLI entry point
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utility evaluation for synthetic network-traffic generators: fidelity and downstream-task metrics.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Evaluation to run.")
    _add_fidelity_netshare_subparser(subparsers)
    _add_fidelity_netssm_subparser(subparsers)
    _add_downstream_task_subparser(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
