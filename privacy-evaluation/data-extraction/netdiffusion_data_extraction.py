"""Data-extraction attack on NetDiffusion.

Adapted from the nprint column-overlap / bitstring-fingerprint logic in
`calculate_memorized_sequence_NetDiffusion()` in a private internal research
harness's `calculate_memorization_attack.py` (not included in this repo;
cleaned up, parameterized, and recast into the same precision/recall framing
used by the other two scripts in this folder -- the original NetDiffusion
function didn't call `calculate_cc()` directly, but performed the same kind
of "does the generated fingerprint reproduce a real one" comparison by hand).
See ./README.md for the overall workflow, and
../../synthetic-data-generation/NetDiffusion for the vendored NetDiffusion
code (`scripts/mass_reconstruction.py`, etc.) that produces the
`best_reconstruction.nprint` files this script reads.

## What this script does, and why it's not IP-address based

Unlike NetSSM/TrafficLLM, NetDiffusion's own preprocessing (see
`nprint_to_png.py` in the vendored NetDiffusion folder) **deliberately drops
IP-address columns** before training, specifically to avoid encoding
identifiable IPs into the model. So an IP-extraction attack doesn't apply
here by construction. Instead, this script extracts a **structural
fingerprint** of each packet's header: the values of a chosen set of nprint
bit-columns (e.g. protocol/flags/TTL bits), turned into a single string per
packet row, using nprint's own convention (`-1` = missing bit -> `*`,
`0` -> `0`, anything else -> `1`).

Workflow:
  1. For each real training-set flow's `.nprint` file and each reconstructed
     flow's `best_reconstruction.nprint` file (produced by NetDiffusion's own
     generation pipeline), read the *first packet's* fingerprint over
     whichever bit-columns are present in both files (`--all_packets` scores
     every packet instead of just the first).
  2. Score with `extraction_common.precision_recall_extraction()`:
     precision = fraction of generated fingerprints that exactly match a
     real, memorized training packet's fingerprint; recall = fraction of the
     training set's unique fingerprints that show up somewhere in the
     reconstructed output.
"""

import argparse
import glob

import pandas as pd

from extraction_common import precision_recall_extraction, print_report

DEFAULT_DROP_COLUMNS = ["src_ip", "dst_ip"]


def row_to_fingerprint(row: pd.Series) -> str:
    """nprint's own missing/0/1 convention, condensed to one character per bit."""
    return "".join("*" if x == -1 else "0" if x == 0 else "1" for x in row.tolist())


def extract_fingerprints(nprint_path: str, columns, all_packets: bool) -> list:
    df = pd.read_csv(nprint_path)
    df = df.drop(columns=[c for c in DEFAULT_DROP_COLUMNS if c in df.columns])
    df = df[columns]
    if len(df) == 0:
        return []
    if not all_packets:
        return [row_to_fingerprint(df.iloc[0])]
    return [row_to_fingerprint(df.iloc[i]) for i in range(len(df))]


def collect_fingerprints(nprint_glob: str, columns, all_packets: bool) -> list:
    values = []
    for fn in glob.glob(nprint_glob):
        try:
            values.extend(extract_fingerprints(fn, columns, all_packets))
        except Exception as e:
            print(f"[skip] {fn}: {e}")
    return values


def resolve_shared_columns(original_glob: str, generated_glob: str):
    """The two nprint layouts (raw training nprint vs. reconstructed nprint)
    may not have identical columns, so only the columns present in *both* (in
    the original's column order) are used for the fingerprint -- mirroring
    the original script's `overlap_columns` logic."""
    original_sample = next(iter(glob.glob(original_glob)), None)
    generated_sample = next(iter(glob.glob(generated_glob)), None)
    if original_sample is None or generated_sample is None:
        raise FileNotFoundError("Could not find a sample file to resolve shared nprint columns from.")

    original_cols = pd.read_csv(original_sample, nrows=0).columns
    original_cols = [c for c in original_cols if c not in DEFAULT_DROP_COLUMNS]
    generated_cols = set(pd.read_csv(generated_sample, nrows=0).columns)

    return [c for c in original_cols if c in generated_cols]


def main():
    parser = argparse.ArgumentParser(description="Data-extraction attack (precision/recall) for NetDiffusion.")
    parser.add_argument(
        "--original_nprint_glob",
        required=True,
        help="Glob for real training-flow .nprint files, e.g. '.../data_MIA_test/all-labels/<dataset>_training/train_nprint/*.nprint'.",
    )
    parser.add_argument(
        "--generated_nprint_glob",
        required=True,
        help="Glob for reconstructed .nprint files, e.g. '.../inference/<dataset>/train/*/best_reconstruction.nprint'.",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        default=None,
        help="nprint bit-columns to fingerprint on. Defaults to every column shared by both globs (minus IP columns).",
    )
    parser.add_argument(
        "--all_packets",
        action="store_true",
        help="Fingerprint every packet in each flow instead of just the first.",
    )
    args = parser.parse_args()

    columns = args.columns or resolve_shared_columns(args.original_nprint_glob, args.generated_nprint_glob)
    print(f"Using {len(columns)} shared nprint columns for the fingerprint.")

    original_values = collect_fingerprints(args.original_nprint_glob, columns, args.all_packets)
    generated_values = collect_fingerprints(args.generated_nprint_glob, columns, args.all_packets)

    metrics = precision_recall_extraction(original_values, generated_values)
    print_report("NetDiffusion / packet-header fingerprint", metrics)


if __name__ == "__main__":
    main()
