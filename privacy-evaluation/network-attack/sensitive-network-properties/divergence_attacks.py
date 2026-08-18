#!/usr/bin/env python3
"""Sensitive-network-property divergence attacks.

Checks whether NetShare, NetSSM, and NetDiffusion reproduce the real
*distribution* of a sensitive or behavioral packet/flow property (TTL, TCP
window size, TCP flags, packet size, flow byte totals, ...) from their
training data closely enough to leak information about it.

Adapted from `calculate_divergence.py` in a private internal research
harness (not included in this repo; cleaned up, parameterized here, and
merged into one file). See README.md for
narrative documentation of the attack methodology; the four attributed
`calculate_divergence_<Model>()` originals map onto the sections of this file
below (a `trafficllm` subcommand is not implemented here -- see README.md).

## What this attack measures

Unlike ../network-identifiers (does a *specific* value reappear?) or
../network-topology (does the *graph structure* reappear?), this attack asks
a softer, distributional question: does the **distribution** of a sensitive
or behavioral packet/flow property in the synthetic data match the real
training distribution closely enough to leak information about it? A
generator that reproduces the training distribution's exact shape (rather
than a smoothed or intentionally-perturbed approximation) is leaking
behavioral/fingerprinting information even when no single generated value is
individually identifiable (e.g. TTL/window-size distributions are commonly
used for OS/device fingerprinting).

## Scoring: normalized Wasserstein (Earth-Mover's) distance

For a chosen field, given `original_values` and `generated_values` (numeric,
one value per packet/flow):

  1. `emd = wasserstein_distance(original_values, generated_values)` -- the
     minimum "work" needed to reshape one distribution into the other.
  2. Normalize by the combined value range so the score is comparable across
     fields with different units/scales: `norm_emd = emd / (max - min)`.

`norm_emd` is in `[0, 1]` for well-behaved inputs (0 = identical
distributions, larger = more divergent) -- **lower is a stronger attack
result** here (it means the generator leaked the real distribution's shape),
which is the opposite direction from the recall/precision/F1 metrics in
../network-identifiers and ../../data-extraction, where higher means a
stronger attack. Don't mix the two without relabeling.

## Feature extraction from pcaps

NetShare and NetSSM expect pre-extracted feature CSVs. If you only have raw
`.pcap` captures, either extract them up front with the `extract-features`
subcommand, or just point `--*_glob` at the `.pcap` files directly -- the
NetShare/NetSSM loaders extract (and cache, as a sibling `_packets.csv` /
`_flows.csv`) automatically the first time, and reuse the cached CSV after
that. See the "Feature extraction from pcaps" section below
(`extract_packet_features()`, `extract_flow_features()`,
`ensure_features_csv()`) -- adapted from the local
`extract_feat_pcaps.py`/`extract_feats.py` harnesses. NetDiffusion is not
covered: it needs `.nprint`-format input from the external `nprint` tool.

## Usage

    # Extract packet-level feature CSVs from pcaps up front (optional --
    # netshare/netssm also do this automatically if pointed at .pcap files).
    python divergence_attacks.py extract-features \\
        --pcap_glob "/path/to/<dataset>/train/*.pcap" \\
        --kind packet

    python divergence_attacks.py netshare \\
        --original_glob "/path/to/<dataset>/train/*/pre_processed_data/*.csv" \\
        --generated_glob "/path/to/<dataset>/train/*/generated_data/sample_len-10/syn_dfs/chunk_id-0/epoch_id-*.csv" \\
        --column pkt_len

    python divergence_attacks.py netssm \\
        --original_packets_glob "/path/to/data_MIA_new/<dataset>/train/*packets.csv" \\
        --generated_packets_glob "/path/to/inference/<dataset>_.../*/singleprompt_*_packets.csv" \\
        --original_flows_glob "/path/to/data_MIA_new/<dataset>/train/*flows.csv" \\
        --generated_flows_glob "/path/to/inference/<dataset>_.../*/singleprompt_*_flows.csv"

    python divergence_attacks.py netdiffusion \\
        --original_nprint_glob "/path/to/data_MIA_test/all-labels/<dataset>_training/train_nprint/*.nprint" \\
        --generated_nprint_glob "/path/to/inference/<dataset>/train/*/best_reconstruction.nprint" \\
        --column_prefix ipv4_tl

Each subcommand's flags are also documented under `--help`, e.g.
`python divergence_attacks.py netssm --help`.
"""

import argparse
import binascii
import glob
import os
import socket
import struct
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


# =============================================================================
# Shared scoring core
# =============================================================================

def normalized_wasserstein(original_values: List[float], generated_values: List[float]) -> Dict[str, float]:
    """Normalized Earth-Mover's distance between two numeric distributions.
    See module docstring for interpretation (lower = more divergence leaked)."""
    original_values = [v for v in original_values if v is not None]
    generated_values = [v for v in generated_values if v is not None]
    if not original_values or not generated_values:
        raise ValueError("Both original_values and generated_values must be non-empty after dropping Nones.")

    emd = wasserstein_distance(original_values, generated_values)
    data_min = min(np.min(original_values), np.min(generated_values))
    data_max = max(np.max(original_values), np.max(generated_values))
    norm_factor = data_max - data_min
    norm_emd = (emd / norm_factor) if norm_factor else 0.0

    return {
        "num_original": len(original_values),
        "num_generated": len(generated_values),
        "emd": float(emd),
        "normalized_emd": float(norm_emd),
    }


def print_report(field_name: str, metrics: Dict[str, float]) -> None:
    print(f"--- Sensitive-property divergence report: {field_name} ---")
    print(f"  Original samples:  {metrics['num_original']}")
    print(f"  Generated samples: {metrics['num_generated']}")
    print(f"  Raw EMD:                {metrics['emd']:.4f}")
    print(f"  Normalized EMD [0, 1]:  {metrics['normalized_emd']:.4f}  (lower = distributions leak more of the real shape)")


# =============================================================================
# Shared packet/field-decoding helpers
#
# Used by the NetSSM (tcp_flags_to_number) and NetDiffusion (bits_to_int)
# attacks below. convert_hex_string / extract_field_from_packet_bytes decode
# raw packet bytes and are kept here (unused by the subcommands implemented
# in this file) because they're shared infrastructure for a TrafficLLM-style
# attack -- see README.md.
# =============================================================================

def tcp_flags_to_number(flag_str: str) -> int:
    """TCP flag string (e.g. 'SA' for SYN+ACK) -> a single bitmask integer,
    so flag combinations can be compared numerically."""
    flag_mapping = {"F": 0, "S": 1, "R": 2, "P": 3, "A": 4, "U": 5, "E": 6, "C": 7, "N": 8}
    if flag_str in (None, "0"):
        return 0
    return sum(2 ** flag_mapping[c] for c in str(flag_str) if c in flag_mapping)


def convert_hex_string(hex_string: str) -> bytes:
    """Best-effort hex-string -> raw bytes, discarding any non-hex characters
    and an odd trailing nibble (matches TrafficLLM/NetDiffusion generation
    output, which occasionally has stray characters at truncation points)."""
    valid_hex = set("0123456789abcdefABCDEF")
    hex_string = "".join(c for c in hex_string if c in valid_hex)
    if len(hex_string) % 2 == 1:
        hex_string = hex_string[:-1]
    return binascii.unhexlify(hex_string)


# byte-offset field map for a raw Ethernet+IPv4(+TCP) packet, as produced by
# TrafficLLM's generation output after hex-decoding.
_SIMPLE_FIELDS = {
    "Src MAC": (0, 5), "Dst MAC": (6, 11), "EtherType": (12, 13),
    "IP Version": 14, "IP Header Length": 14,
    "Type of Service": 15, "ToS": 15,
    "Total Length": (16, 17), "Identification": (18, 19), "ID": (18, 19),
    "Flags": (20, 21), "Fragment Offset": (20, 21),
    "TTL": 22, "Protocol": 23, "IP Checksum": (24, 25),
    "Src IP": (26, 29), "Dst IP": (30, 33),
}
_TCP_FIELDS = {
    "Src Port", "Dst Port", "Sequence Number", "Seq Num", "Acknowledgment Number", "Ack Num",
    "TCP Data Offset", "Control Flags", "TCP Flags", "Window Size", "TCP Checksum", "Urgent Pointer",
}


def extract_field_from_packet_bytes(packet_bytes: bytes, field: str) -> Optional[int]:
    """Extract one header field from raw packet bytes (Ethernet + IPv4,
    optionally + TCP). Returns None if the packet is too short for the
    requested field. See TrafficLLM's own `tutorials/generation.py` for the
    inverse operation (building bytes from a header dict)."""
    if field in _SIMPLE_FIELDS:
        position = _SIMPLE_FIELDS[field]
        if isinstance(position, int):
            if position >= len(packet_bytes):
                return None
            value = packet_bytes[position]
            if field == "IP Version":
                return value >> 4
            if field == "IP Header Length":
                return (value & 0x0F) * 4
            return value
        start, end = position
        data = packet_bytes[start:end + 1]
        if len(data) != (end - start + 1):
            return None
        if field in ("Src MAC", "Dst MAC"):
            return ":".join(f"{b:02x}" for b in data)
        if field in ("Src IP", "Dst IP"):
            return socket.inet_ntoa(data)
        if field == "EtherType":
            return struct.unpack("!H", data)[0]
        if field in ("Total Length", "Identification", "ID", "IP Checksum"):
            return struct.unpack("!H", data)[0]
        if field in ("Flags", "Fragment Offset"):
            combined = struct.unpack("!H", data)[0]
            return (combined >> 13) if field == "Flags" else (combined & 0x1FFF)

    elif field in _TCP_FIELDS:
        if len(packet_bytes) < 15:
            return None
        ip_header_length = (packet_bytes[14] & 0x0F) * 4
        tcp_start = 14 + ip_header_length
        if len(packet_bytes) < tcp_start + 20:
            return None
        tcp_fields = {
            "Src Port": (tcp_start, tcp_start + 1), "Dst Port": (tcp_start + 2, tcp_start + 3),
            "Sequence Number": (tcp_start + 4, tcp_start + 7), "Seq Num": (tcp_start + 4, tcp_start + 7),
            "Acknowledgment Number": (tcp_start + 8, tcp_start + 11), "Ack Num": (tcp_start + 8, tcp_start + 11),
            "TCP Data Offset": tcp_start + 12, "Control Flags": tcp_start + 13, "TCP Flags": tcp_start + 13,
            "Window Size": (tcp_start + 14, tcp_start + 15), "TCP Checksum": (tcp_start + 16, tcp_start + 17),
            "Urgent Pointer": (tcp_start + 18, tcp_start + 19),
        }
        position = tcp_fields[field]
        if isinstance(position, int):
            value = packet_bytes[position]
            return (value >> 4) * 4 if field == "TCP Data Offset" else value
        start, end = position
        data = packet_bytes[start:end + 1]
        if field in ("Src Port", "Dst Port", "Window Size", "TCP Checksum", "Urgent Pointer"):
            return struct.unpack("!H", data)[0]
        if field in ("Sequence Number", "Seq Num", "Acknowledgment Number", "Ack Num"):
            return struct.unpack("!I", data)[0]

    elif field == "Packet Size":
        return len(packet_bytes)

    else:
        raise ValueError(f"Field '{field}' not recognized.")


def bits_to_int(bits) -> int:
    """MSB-first bit list -> integer, for decoding nprint bit-columns."""
    value = 0
    for bit in bits:
        value = value * 2 + (0 if bit <= 0 else 1)  # nprint uses -1 for missing bits
    return value


# =============================================================================
# Feature extraction from pcaps
#
# Adapted from extract_packet_features_helper()/extract_flow_features_helper()
# and extract_packet()/extract_flow() in two private internal research
# harnesses' extract_feat_pcaps.py / extract_feats.py (not included in this
# repo), trimmed down to just the columns the NetSSM and NetShare loaders below read (TTL, IP
# ID, IP Type of Service, TCP Window Size, TCP Data Offset, Packet Size,
# TCP Flags, pkt_len, Total Bytes). Requires scapy (imported lazily so the
# rest of this script works without it if pcap extraction is never invoked).
#
# NetDiffusion is not covered here: its `.nprint` inputs come from the
# external `nprint` CLI tool, not from parsing pcaps directly.
# =============================================================================

def extract_packet_features(pcap_file: str) -> pd.DataFrame:
    """Per-packet features from one pcap: the IP/TCP header fields the
    NetSSM attack reads (TTL, IP ID, IP Type of Service, TCP Window Size,
    TCP Data Offset, Packet Size, TCP Flags), plus `pkt_len` (NetShare's
    column name for the same packet-length value) and flow-identifying
    columns (Src/Dst IP/Port, Protocol, Timestamp). One row per IP packet;
    non-IP packets are skipped."""
    from scapy.all import IP, TCP, UDP, rdpcap

    rows = []
    for pkt in rdpcap(pcap_file):
        if IP not in pkt:
            continue
        ip_layer = pkt[IP]
        if TCP in pkt:
            proto, l4 = "TCP", pkt[TCP]
        elif UDP in pkt:
            proto, l4 = "UDP", pkt[UDP]
        else:
            proto, l4 = None, None

        rows.append({
            "Timestamp": float(pkt.time),
            "Src IP": ip_layer.src,
            "Dst IP": ip_layer.dst,
            "Src Port": getattr(l4, "sport", None),
            "Dst Port": getattr(l4, "dport", None),
            "Protocol": proto,
            "Packet Size": len(pkt),
            "pkt_len": len(pkt),
            "TTL": ip_layer.ttl,
            "IP ID": ip_layer.id,
            "IP Type of Service": ip_layer.tos,
            "TCP Window Size": l4.window if proto == "TCP" else None,
            "TCP Data Offset": l4.dataofs if proto == "TCP" else None,
            "TCP Flags": str(l4.flags) if proto == "TCP" else None,
        })
    return pd.DataFrame(rows)


def extract_flow_features(pcap_file: str) -> pd.DataFrame:
    """Per-flow features from one pcap, keyed by (src ip, dst ip, src port,
    dst port, protocol): packet/byte totals, used here for NetSSM's
    flow-level `Total Bytes` field."""
    from scapy.all import IP, TCP, UDP, rdpcap

    flows: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {"Total Packets": 0, "Total Bytes": 0})
    for pkt in rdpcap(pcap_file):
        if IP not in pkt:
            continue
        ip_layer = pkt[IP]
        if TCP in pkt:
            proto, l4 = "TCP", pkt[TCP]
        elif UDP in pkt:
            proto, l4 = "UDP", pkt[UDP]
        else:
            continue
        flow_key = (ip_layer.src, ip_layer.dst, l4.sport, l4.dport, proto)
        flows[flow_key]["Total Packets"] += 1
        flows[flow_key]["Total Bytes"] += len(pkt)

    rows = []
    for (src_ip, dst_ip, src_port, dst_port, proto), stats in flows.items():
        rows.append({
            "Src IP": src_ip, "Dst IP": dst_ip, "Src Port": src_port, "Dst Port": dst_port,
            "Protocol": proto, "Total Packets": stats["Total Packets"], "Total Bytes": stats["Total Bytes"],
        })
    return pd.DataFrame(rows)


def ensure_features_csv(pcap_file: str, kind: str = "packet", overwrite: bool = False) -> str:
    """Extracts `kind` ("packet" or "flow") features from `pcap_file` into a
    sibling CSV (`<name>_packets.csv` / `<name>_flows.csv`, next to the
    pcap) if that CSV doesn't already exist, and returns its path. No-ops
    (just returns the existing path) when the CSV is already there, unless
    `overwrite=True` -- mirrors the `if not os.path.exists(...)` caching
    pattern in the reference extraction scripts."""
    if kind not in ("packet", "flow"):
        raise ValueError(f"kind must be 'packet' or 'flow', got {kind!r}")
    suffix = "_packets.csv" if kind == "packet" else "_flows.csv"
    extractor = extract_packet_features if kind == "packet" else extract_flow_features

    base, _ = os.path.splitext(pcap_file)
    csv_path = base + suffix

    if overwrite or not os.path.exists(csv_path):
        extractor(pcap_file).to_csv(csv_path, index=False)
    return csv_path


def extract_features_from_pcaps(pcap_glob: str, kind: str = "packet", overwrite: bool = False) -> List[str]:
    """Batch entry point: for every file matching `pcap_glob`, ensure a
    features CSV exists (extracting from the pcap if not already present),
    and return the list of CSV paths produced/found."""
    csv_paths = []
    for pcap_file in sorted(glob.glob(pcap_glob)):
        try:
            csv_paths.append(ensure_features_csv(pcap_file, kind=kind, overwrite=overwrite))
        except Exception as e:
            print(f"[skip] {pcap_file}: {e}")
    return csv_paths


# =============================================================================
# NetShare
#
# Adapted from `calculate_divergence_NetShare()`. Compares the distribution
# of a chosen numeric column (default: `pkt_len`) between real and synthetic
# flow data. See ../../../synthetic-data-generation/NetShare for the
# vendored NetShare code that produces the CSVs read here.
# =============================================================================

def _netshare_load_column(csv_glob: str, column: str) -> list:
    """Reads `column` from every CSV matching `csv_glob`. If `csv_glob`
    matches `.pcap` files instead (no pre-extracted CSV available), each
    pcap's packet-feature CSV is extracted on demand via
    `ensure_features_csv()` and read from there."""
    values = []
    for fn in glob.glob(csv_glob):
        try:
            if fn.endswith(".pcap"):
                fn = ensure_features_csv(fn, kind="packet")
            df = pd.read_csv(fn, encoding="utf-8", encoding_errors="ignore")
            values.extend(df[column].tolist())
        except Exception as e:
            print(f"[skip] {fn}: {e}")
    return values


def run_netshare(args: argparse.Namespace) -> None:
    original_values = _netshare_load_column(args.original_glob, args.column)
    generated_values = _netshare_load_column(args.generated_glob, args.column)

    metrics = normalized_wasserstein(original_values, generated_values)
    print_report(f"NetShare / {args.column}", metrics)


def _add_netshare_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("netshare", help="Sensitive-property divergence attack for NetShare.")
    parser.add_argument("--original_glob", required=True, help="Glob for real flow CSVs (pre_processed_data/*.csv).")
    parser.add_argument("--generated_glob", required=True, help="Glob for synthetic flow CSVs (generated_data/.../epoch_id-*.csv).")
    parser.add_argument("--column", default="pkt_len", help="Numeric column to compare, e.g. pkt_len, tos, ttl.")
    parser.set_defaults(func=run_netshare)


# =============================================================================
# NetSSM
#
# Adapted from `calculate_divergence_NetSSM()`. Reads already-decoded
# per-packet CSVs (TTL, IP ID, IP Type of Service, TCP Window Size, TCP Data
# Offset, Packet Size, TCP Flags) and per-flow CSVs (Total Bytes), and
# compares their distributions between real and generated data. See
# ../../../synthetic-data-generation/NetSSM for the vendored NetSSM code.
# =============================================================================

NETSSM_PACKET_FIELDS = ["TTL", "IP ID", "IP Type of Service", "TCP Window Size", "TCP Data Offset", "Packet Size", "TCP Flags"]


def _netssm_load_column(csv_glob: str, column: str, n_head: int = 200) -> list:
    """Reads `column` from every CSV matching `csv_glob`, keeping only the
    first `n_head` rows of each (matches the original script's behavior of
    only using the first N packets of each flow). If `csv_glob` matches
    `.pcap` files instead (no pre-extracted CSV available), each pcap's
    features are extracted on demand via `ensure_features_csv()` -- as
    packet-level fields, or as the flow-level `Total Bytes` field."""
    values = []
    for fn in glob.glob(csv_glob):
        try:
            if fn.endswith(".pcap"):
                fn = ensure_features_csv(fn, kind="flow" if column == "Total Bytes" else "packet")
            df = pd.read_csv(fn).iloc[:n_head]
        except Exception as e:
            print(f"[skip] {fn}: {e}")
            continue
        if df is None or len(df) == 0 or column not in df.columns:
            continue
        col = df[column].dropna().tolist()
        values.extend(col)
    return values


def run_netssm(args: argparse.Namespace) -> None:
    for field in args.fields:
        original_values = _netssm_load_column(args.original_packets_glob, field)
        generated_values = _netssm_load_column(args.generated_packets_glob, field)
        if field == "TCP Flags":
            original_values = [tcp_flags_to_number(v) for v in original_values]
            generated_values = [tcp_flags_to_number(v) for v in generated_values]
        else:
            original_values = [int(v) for v in original_values]
            generated_values = [int(v) for v in generated_values]
        if not set(original_values) or not set(generated_values):
            print(f"[skip] {field}: no values on one side")
            continue
        metrics = normalized_wasserstein(original_values, generated_values)
        print_report(f"NetSSM / {field}", metrics)

    if args.original_flows_glob and args.generated_flows_glob:
        original_values = [float(v) for v in _netssm_load_column(args.original_flows_glob, "Total Bytes")]
        generated_values = [float(v) for v in _netssm_load_column(args.generated_flows_glob, "Total Bytes")]
        if set(original_values) and set(generated_values):
            metrics = normalized_wasserstein(original_values, generated_values)
            print_report("NetSSM / Total Bytes (flow-level)", metrics)


def _add_netssm_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("netssm", help="Sensitive-property divergence attack for NetSSM.")
    parser.add_argument("--original_packets_glob", required=True, help="Glob for real per-packet CSVs, e.g. '.../train/*packets.csv'.")
    parser.add_argument("--generated_packets_glob", required=True, help="Glob for generated per-packet CSVs.")
    parser.add_argument("--original_flows_glob", default=None, help="Glob for real per-flow CSVs (for the 'Total Bytes' field), e.g. '.../train/*flows.csv'.")
    parser.add_argument("--generated_flows_glob", default=None, help="Glob for generated per-flow CSVs.")
    parser.add_argument(
        "--fields",
        nargs="+",
        default=NETSSM_PACKET_FIELDS,
        help=f"Packet-level fields to compare. Default: {NETSSM_PACKET_FIELDS}",
    )
    parser.set_defaults(func=run_netssm)


# =============================================================================
# NetDiffusion
#
# Adapted from `calculate_divergence_NetDiffusion()`. Decodes a chosen
# nprint bit-column group (default: `ipv4_tl`, IPv4 total length) back into
# an integer per packet, for both the real training `.nprint` and the
# reconstructed `best_reconstruction.nprint`, and compares their
# distributions. Unlike ../network-identifiers/netdiffusion_identifiers.py
# and ../network-topology/netdiffusion_topology.py, this attack works on
# *any* numeric field NetDiffusion generates -- including TTL, window size,
# or TCP flags -- since it doesn't need identifiable IP columns (which
# NetDiffusion's own preprocessing drops before training; see those two
# scripts' docstrings for the same caveat). See
# ../../../synthetic-data-generation/NetDiffusion for the vendored
# NetDiffusion code that produces the `.nprint` files read here.
# =============================================================================

def _netdiffusion_extract_field(nprint_glob: str, column_prefix: str) -> list:
    values = []
    for fn in glob.glob(nprint_glob):
        try:
            df = pd.read_csv(fn)
        except Exception as e:
            print(f"[skip] {fn}: {e}")
            continue
        cols = sorted(
            [c for c in df.columns if c.startswith(f"{column_prefix}_")],
            key=lambda c: int(c.rsplit("_", 1)[-1]),
        )
        if not cols:
            continue
        for _, row in df[cols].iterrows():
            values.append(bits_to_int(row.tolist()))
    return values


def run_netdiffusion(args: argparse.Namespace) -> None:
    original_values = _netdiffusion_extract_field(args.original_nprint_glob, args.column_prefix)
    generated_values = _netdiffusion_extract_field(args.generated_nprint_glob, args.column_prefix)

    metrics = normalized_wasserstein(original_values, generated_values)
    print_report(f"NetDiffusion / {args.column_prefix}", metrics)


def _add_netdiffusion_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("netdiffusion", help="Sensitive-property divergence attack for NetDiffusion.")
    parser.add_argument("--original_nprint_glob", required=True, help="Glob for real training-flow .nprint files.")
    parser.add_argument(
        "--generated_nprint_glob",
        required=True,
        help="Glob for reconstructed .nprint files, e.g. '.../inference/<dataset>/train/*/best_reconstruction.nprint'.",
    )
    parser.add_argument(
        "--column_prefix",
        default="ipv4_tl",
        help="nprint bit-column group to decode, e.g. ipv4_tl (total length), ipv4_ttl, tcp_wsize.",
    )
    parser.set_defaults(func=run_netdiffusion)


# =============================================================================
# CLI entry point
# =============================================================================

def run_extract_features(args: argparse.Namespace) -> None:
    csv_paths = extract_features_from_pcaps(args.pcap_glob, kind=args.kind, overwrite=args.overwrite)
    print(f"Extracted/verified {len(csv_paths)} {args.kind}-level feature CSV(s) from '{args.pcap_glob}'.")


def _add_extract_features_subparser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "extract-features",
        help="Extract packet/flow feature CSVs from raw .pcap files, for use as --*_glob input to the other subcommands.",
    )
    parser.add_argument("--pcap_glob", required=True, help="Glob for .pcap files to extract from, e.g. '/path/to/train/*.pcap'.")
    parser.add_argument(
        "--kind", choices=["packet", "flow"], default="packet",
        help="'packet': per-packet fields (TTL, TCP Window Size, TCP Flags, ...). 'flow': per-flow fields (Total Bytes, ...).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if a features CSV already exists next to the pcap.")
    parser.set_defaults(func=run_extract_features)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sensitive-network-property divergence attacks (NetShare, NetSSM, NetDiffusion).",
    )
    subparsers = parser.add_subparsers(dest="model", required=True, help="Target model to attack.")
    _add_netshare_subparser(subparsers)
    _add_netssm_subparser(subparsers)
    _add_netdiffusion_subparser(subparsers)
    _add_extract_features_subparser(subparsers)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
