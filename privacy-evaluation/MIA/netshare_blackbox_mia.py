"""Black-box (statistical-feature) membership inference attack on NetShare.



Threat model: the attacker only has query access to the fine-tuned NetShare
generator's *output* -- per-flow synthetic packet-header CSVs -- no access to
model weights or gradients. NetShare is trained one model per real flow (see
../../synthetic-data-generation/NetShare's `examples/pcap/driver.py`), so
"is flow X a member" reduces to: does the flow-conditioned synthetic output
statistically resemble flow X well enough to identify it? Unlike NetDiffusion
(reconstruction MSE) or NetSSM/TrafficLLM (perplexity), NetShare's output is
tabular, not an image or text sequence, so the attacker instead:
  1. For each per-flow synthetic CSV, computes a fixed-size statistical
     feature vector summarizing that flow (packet count, unique src/dst
     IP/port counts, protocol distribution, packet-length/ToS/TTL summary
     stats, inter-arrival-time mean/std).
  2. Labels each feature vector with the identity of the original flow it was
     generated from (recovered from the per-flow output folder name).
  3. Repeats for the `train` (members), `test` (held-out non-members), and
     `otheractivity` (out-of-distribution) splits, producing one feature
     table per split.

These per-split feature tables are the attack's raw signal: a downstream
binary classifier (e.g. AutoGluon/random forest) trained to distinguish
`train`-split feature vectors from `test`/`otheractivity`-split ones is the
actual membership decision -- see "Evaluating attack success" in ./README.md.
"""

import argparse
import glob
import os

import pandas as pd


def extract_features_single_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed-size statistical feature vector summarizing one synthetic flow's
    packet-header CSV (as produced by NetShare)."""
    feats = pd.DataFrame()

    feats["packet_count"] = [len(df)]
    feats["unique_srcip"] = [df["srcip"].nunique()]
    feats["unique_dstip"] = [df["dstip"].nunique()]
    feats["unique_srcport"] = [df["srcport"].nunique()]
    feats["unique_dstport"] = [df["dstport"].nunique()]

    proto_counts = df["proto"].value_counts(normalize=True).to_dict()
    for p in ["TCP", "UDP", "ICMP"]:
        feats[f"proto_{p}"] = [proto_counts.get(p, 0.0)]

    for col in ["pkt_len", "tos", "ttl"]:
        feats[f"{col}_mean"] = [df[col].mean()]
        feats[f"{col}_std"] = [df[col].std()]
        feats[f"{col}_min"] = [df[col].min()]
        feats[f"{col}_max"] = [df[col].max()]

    if "time" in df.columns:
        feats["duration"] = [df["time"].max() - df["time"].min()]
        if len(df) > 1:
            inter_arrival = df["time"].sort_values().diff().dropna()
            feats["iat_mean"] = [inter_arrival.mean()]
            feats["iat_std"] = [inter_arrival.std()]
        else:
            feats["iat_mean"] = [0]
            feats["iat_std"] = [0]

    return feats


def score_subset(
    generated_data_dir: str,
    dataset: str,
    subset: str,
    epoch: int,
    output_dir: str,
    csv_glob_suffix: str = "*/generated_data/sample_len-10/syn_dfs/chunk_id-0/epoch_id-{epoch}.csv",
) -> str:
    """Extract per-flow feature vectors for every synthetic flow CSV in one
    subset. Expects the NetShare output layout produced by
    ../../synthetic-data-generation/NetShare's examples/pcap/driver.py:
    `{generated_data_dir}/{dataset}/{subset}/{flow_name}/{csv_glob_suffix}`.
    """
    pattern = os.path.join(generated_data_dir, dataset, subset, csv_glob_suffix.format(epoch=epoch))
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    for fn in glob.glob(pattern):
        try:
            df = pd.read_csv(fn).dropna().reset_index(drop=True)
            features = extract_features_single_df(df)
            if len(features) == 0:
                continue
            # Recovers the original flow's identity/label from its per-flow
            # output folder name -- see driver.py's `work_folder` naming.
            # Index depends on csv_glob_suffix's depth below {subset}/{flow_name}/...
            flow_name = fn.split("/")[-6]
            features["Label"] = [flow_name.split("_")[0]]
            rows.append(features)
        except Exception as e:
            print(f"[skip] {fn}: {e}")

    out_path = os.path.join(output_dir, f"NetShare_features_{subset}.csv")
    if rows:
        pd.concat(rows, axis=0).to_csv(out_path, index=False)
    else:
        pd.DataFrame().to_csv(out_path, index=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Black-box statistical-feature MIA for NetShare.")
    parser.add_argument(
        "--generated_data_dir",
        required=True,
        help="Root folder containing {dataset}/{subset}/{flow_name}/generated_data/... synthetic flow CSVs.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. servicerecognition.")
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["train", "test", "otheractivity"],
        help="Subsets to score. 'train' = members, 'test' = held-out non-members, "
        "'otheractivity' = out-of-distribution negative control.",
    )
    parser.add_argument("--epoch", type=int, required=True, help="Training epoch of the synthetic CSVs to read.")
    parser.add_argument("--output_dir", required=True, help="Where to write NetShare_features_{subset}.csv files.")
    args = parser.parse_args()

    for subset in args.subsets:
        out_path = score_subset(args.generated_data_dir, args.dataset, subset, args.epoch, args.output_dir)
        print(f"[{subset}] saved feature table to {out_path}")


if __name__ == "__main__":
    main()
