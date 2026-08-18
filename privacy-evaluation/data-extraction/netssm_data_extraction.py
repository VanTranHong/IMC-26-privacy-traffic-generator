"""Data-extraction attack on NetSSM.

Adapted from the MAC-address extraction logic in `coverage_confidence_NetSSM()`
in a private internal research harness's `calculate_memorization_attack.py`
(not included in this repo; cleaned up, parameterized, and generalized here). See ./README.md for the
overall workflow, and ../../synthetic-data-generation/NetSSM for the vendored
NetSSM code that produces the packet sequences this script reads.

## What this script does

NetSSM generates packets as space-separated decimal byte strings, e.g.
`<|netflix|> 226 70 154 78 108 47 ... <|pkt|> 00 128 153 192 ... <|pkt|>`.
The first 6 bytes of the first packet in a flow are the Ethernet
**destination MAC address**; bytes 6-11 are the **source MAC address** (any
other fixed byte range/header field can be scored the same way via
`--byte_start`/`--byte_end`).

Workflow:
  1. Read every training-set packet's target byte range from the original
     JSONL dataset -> `original_values` (a MAC address string per packet).
  2. Read the same byte range from every model-generated packet -> `generated_values`.
  3. Score with `extraction_common.precision_recall_extraction()`:
     precision = fraction of generated MAC addresses that are real,
     memorized training addresses; recall = fraction of the training set's
     unique MAC addresses that show up somewhere in the generated output.

Run once per split you want to compare (e.g. run against the `train` JSONL
to measure how much of the *fine-tuning* data leaks out).
"""

import argparse
import glob
import json
import os

from extraction_common import precision_recall_extraction, print_report


def extract_byte_field(packet: str, byte_start: int, byte_end: int, is_first_packet: bool) -> str:
    """Extract bytes [byte_start:byte_end] from one packet's space-separated
    byte string. The first packet in a flow is prefixed with the `<|label|>`
    token, which occupies byte position 0, so the requested range is shifted
    by one there to line back up with actual packet bytes."""
    tokens = packet.strip().split(" ")
    offset = 1 if is_first_packet else 0
    return " ".join(tokens[offset + byte_start: offset + byte_end])


def extract_field_from_jsonl(jsonl_path: str, byte_start: int, byte_end: int) -> list:
    """One field value per packet, across every flow in a NetSSM-format JSONL
    dataset (one `{"Data": "<|label|> ... <|pkt|> ..."}` record per line)."""
    values = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)["Data"]
            packets = data.split("<|pkt|>")
            for i, packet in enumerate(packets):
                if packet.strip():
                    values.append(extract_byte_field(packet, byte_start, byte_end, is_first_packet=(i == 0)))
    return values


def extract_field_from_generated(txt_glob: str, byte_start: int, byte_end: int) -> list:
    """One field value per packet, across every generated `.txt` sample
    matching `txt_glob` (NetSSM's `generation/generate.py` output format)."""
    values = []
    for fn in glob.glob(txt_glob):
        with open(fn, "r", encoding="utf-8") as f:
            for line in f:
                packets = line.split("<|pkt|>")
                for i, packet in enumerate(packets):
                    if packet.strip():
                        values.append(extract_byte_field(packet, byte_start, byte_end, is_first_packet=(i == 0)))
    return values


def main():
    parser = argparse.ArgumentParser(description="Data-extraction attack (precision/recall) for NetSSM.")
    parser.add_argument(
        "--original_jsonl",
        required=True,
        help="NetSSM-format training JSONL, e.g. ../../synthetic-data-generation/NetSSM preprocessing output.",
    )
    parser.add_argument(
        "--generated_glob",
        required=True,
        help="Glob for generated .txt files, e.g. '.../inference/<run>/*/singleprompt_*.txt'.",
    )
    parser.add_argument("--byte_start", type=int, default=0, help="Start byte offset of the field (default: dst MAC).")
    parser.add_argument("--byte_end", type=int, default=6, help="End byte offset (exclusive) of the field.")
    parser.add_argument("--field_name", default="dst_mac", help="Label for the field being scored, for the report.")
    args = parser.parse_args()

    original_values = extract_field_from_jsonl(args.original_jsonl, args.byte_start, args.byte_end)
    generated_values = extract_field_from_generated(args.generated_glob, args.byte_start, args.byte_end)

    metrics = precision_recall_extraction(original_values, generated_values)
    print_report(f"NetSSM / {args.field_name}", metrics)


if __name__ == "__main__":
    main()
