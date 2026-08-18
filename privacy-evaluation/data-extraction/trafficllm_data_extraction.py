"""Data-extraction attack on TrafficLLM.

Adapted from the IP-address extraction logic in `coverage_confidence_TrafficLLM()`
in a private internal research harness's `calculate_memorization_attack.py`
(not included in this repo; cleaned up, parameterized here). See ./README.md for the overall workflow,
and ../../synthetic-data-generation/TrafficLLM for the vendored TrafficLLM
code that produces the outputs this script reads.

## What this script does

TrafficLLM's traffic-generation task outputs a Python-dict-formatted packet
header followed by a hex payload, e.g.
`{'src': '147.32.84.165', 'dst': '212.117.171.138', 'proto': 6, ...} <hex bytes>`.
This script pulls the `src` (or `dst`) IP address string out of that header.

Workflow:
  1. Read every training sample's target IP field from the original
     `{instruction, output}` JSONL dataset -> `original_values`.
  2. Read the same field from every model-generated `.txt` sample -> `generated_values`.
  3. Score with `extraction_common.precision_recall_extraction()`:
     precision = fraction of generated IP addresses that are real, memorized
     training addresses; recall = fraction of the training set's unique IP
     addresses that show up somewhere in the generated output.

Run once per split you want to compare (e.g. run against the `train` JSON to
measure how much of the fine-tuning data leaks out).
"""

import argparse
import glob
import json

from extraction_common import precision_recall_extraction, print_report


def extract_ip_field(text: str, field: str) -> str:
    """Pull `'src': '1.2.3.4'` (or `'dst': ...`) out of a TrafficLLM header
    dict string. Returns None if the field isn't present in `text`."""
    marker = f"'{field}': "
    if marker not in text:
        return None
    try:
        return text.split(marker)[-1].split("'")[1]
    except IndexError:
        return None


def extract_field_from_jsonl(jsonl_path: str, field: str) -> list:
    """One IP value per sample, across a TrafficLLM `{instruction, output}`
    JSONL training/test dataset."""
    values = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            output = json.loads(line)["output"]
            value = extract_ip_field(output, field)
            if value is not None:
                values.append(value)
    return values


def extract_field_from_generated(txt_glob: str, field: str) -> list:
    """One IP value per generated sample, across every `.txt` file matching
    `txt_glob` (TrafficLLM's `evaluation.py`/`inference.py` output format)."""
    values = []
    for fn in glob.glob(txt_glob):
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                value = extract_ip_field(line, field)
                if value is not None:
                    values.append(value)
    return values


def main():
    parser = argparse.ArgumentParser(description="Data-extraction attack (precision/recall) for TrafficLLM.")
    parser.add_argument(
        "--original_jsonl",
        required=True,
        help="TrafficLLM-format training JSONL, e.g. "
        ".../output_TrafficLLM/<dataset>_final_train/<dataset>_final_train_preprocessed_generation_packet_train.json.",
    )
    parser.add_argument(
        "--generated_glob",
        required=True,
        help="Glob for generated .txt files, e.g. '.../inference_new/<dataset>_<epoch>epochs/*/*_maxlen_300_temp_0_9.txt'.",
    )
    parser.add_argument("--field", choices=["src", "dst"], default="src", help="Which IP field to extract.")
    args = parser.parse_args()

    original_values = extract_field_from_jsonl(args.original_jsonl, args.field)
    generated_values = extract_field_from_generated(args.generated_glob, args.field)

    metrics = precision_recall_extraction(original_values, generated_values)
    print_report(f"TrafficLLM / {args.field}_ip", metrics)


if __name__ == "__main__":
    main()
